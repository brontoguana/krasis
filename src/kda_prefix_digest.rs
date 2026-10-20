// Actual leased owner images, before and after an ordinary producer/recurrence
// span. No state mutation, numerical kernel, or diagnostic GPU allocation.
fn capture_selected_state_sha256(
    owner: &PreparedOwner,
    layer: usize,
    expected_absolute: usize,
    phase: &str,
    target: &crate::matched_probe::Target,
    maximum: usize,
    stream: sys::CUstream,
) -> Result<(), String> {
    let mut registry = crate::session_cache::SequenceStateRegistry::default();
    owner.register(&mut registry, layer)?;
    let mut scalars = std::collections::BTreeMap::new();
    for role in ["absolute", "pending", "error"] {
        let allocation = registry
            .allocations()
            .iter()
            .find(|a| a.name.rsplit('.').next() == Some(role))
            .ok_or("digest scalar registry allocation missing")?;
        let data = target.read_digest_source(
            role,
            allocation.ptr,
            allocation.storage_bytes,
            maximum,
            stream,
        )?;
        if data.len() != if role == "absolute" { 8 } else { 4 } {
            return Err("digest scalar extent mismatch".into());
        }
        scalars.insert(role, data);
    }
    let absolute = u64::from_le_bytes(
        scalars["absolute"]
            .as_slice()
            .try_into()
            .map_err(|_| "digest absolute extent")?,
    );
    let pending = i32::from_le_bytes(
        scalars["pending"]
            .as_slice()
            .try_into()
            .map_err(|_| "digest pending extent")?,
    );
    let error = i32::from_le_bytes(
        scalars["error"]
            .as_slice()
            .try_into()
            .map_err(|_| "digest error extent")?,
    );
    if absolute != u64::try_from(expected_absolute).map_err(|_| "digest absolute exceeds u64")?
        || pending < 0
        || error != 0
    {
        return Err(format!("actual digest cursor/poison differs: absolute={absolute} expected={expected_absolute} pending={pending} error={error}"));
    }
    let semantic = absolute
        .checked_sub(pending as u64)
        .ok_or("digest semantic cursor underflow")?;
    let mut entries = Vec::new();
    let mut source_bytes = 0usize;
    let mut retained_bytes = 0usize;
    for (index, allocation) in registry.allocations().iter().enumerate() {
        let role = allocation
            .name
            .rsplit('.')
            .next()
            .ok_or("digest registry role absent")?;
        let data = if let Some(s) = scalars.get(role) {
            s.clone()
        } else {
            target.read_digest_source(
                &allocation.name,
                allocation.ptr,
                allocation.storage_bytes,
                maximum,
                stream,
            )?
        };
        source_bytes = source_bytes
            .checked_add(data.len())
            .ok_or("state digest source accounting overflow")?;
        let file = format!("allocation-{index}.sha256");
        target.write(&file, &Sha256::digest(&data))?;
        retained_bytes = retained_bytes
            .checked_add(32)
            .ok_or("state digest retained accounting overflow")?;
        let scalar_file = if scalars.contains_key(role) {
            let file = format!("allocation-{index}.bin");
            target.write(&file, &data)?;
            retained_bytes = retained_bytes
                .checked_add(data.len())
                .ok_or("scalar retained accounting overflow")?;
            Some(file)
        } else {
            None
        };
        let mut valid_pending_file = None;
        let mut valid_pending_rows = None;
        if role.starts_with("raw") {
            let capacity = *allocation
                .shape
                .first()
                .ok_or("pending digest missing capacity")?;
            let row_bytes = *allocation
                .strides_bytes
                .first()
                .ok_or("pending digest missing row stride")?;
            if pending as usize >= capacity
                || row_bytes == 0
                || capacity.checked_mul(row_bytes) != Some(data.len())
            {
                return Err(
                    "pending digest geometry/cursor exceeds actual contiguous owner".into(),
                );
            }
            let bytes = (pending as usize)
                .checked_mul(row_bytes)
                .ok_or("valid pending digest overflow")?;
            let hashes = if pending == 0 {
                Vec::new()
            } else {
                crate::matched_probe::prefix_row_sha256(&data[..bytes], row_bytes)?
            };
            let file = format!("allocation-{index}-valid-pending.sha256");
            target.write(&file, &hashes)?;
            retained_bytes = retained_bytes
                .checked_add(hashes.len())
                .ok_or("valid pending retained accounting overflow")?;
            valid_pending_file = Some(file);
            valid_pending_rows = Some(pending as usize);
        }
        entries.push(serde_json::json!({"index":index,"name":allocation.name,"dtype":allocation.dtype,"source_bytes_read":data.len(),"storage_bytes":allocation.storage_bytes,"shape":allocation.shape,"strides_bytes":allocation.strides_bytes,"actual_ptr":allocation.ptr,"device_ordinal":allocation.device_ordinal,"file":file,"encoding":"SHA256 of the complete actual allocation bytes","scalar_file":scalar_file,"valid_pending_file":valid_pending_file,"valid_pending_rows":valid_pending_rows,"unused_pending_capacity_is_not_causal_evidence":role.starts_with("raw")}));
    }
    target.write("descriptor.json",&serde_json::to_vec_pretty(&serde_json::json!({"phase":phase,"actual_state_at_observation":true,"absolute":absolute,"pending":pending,"error":error,"complete_semantic_position":semantic,"expected_absolute":expected_absolute,"layer":layer,"heads":owner.heads,"dim":owner.dim,"arithmetic_sha256":owner.arithmetic_sha256,"source_bytes_read":source_bytes,"retained_digest_and_scalar_bytes":retained_bytes,"entries":entries,"diagnostic_gpu_allocation_bytes":0,"scope":"actual leased state image; only complete/latest semantic positions and valid pending rows support chronology; scratch/unused storage hashes are descriptive"})).map_err(|e|e.to_string())?)
}
