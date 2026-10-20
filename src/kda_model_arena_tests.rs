fn model_arena_admission_controls(
    root: &Path,
    device: &Arc<CudaDevice>,
    bindings: &[ModelFamilyBinding<'_>],
    stream: &NativeStream,
    identity: &Value,
    shared_a: &PreparedOwner,
    shared_b: &PreparedOwner,
    producer: &KdaRawProducer,
) -> Result<(), String> {
    let flags = || -> Result<Vec<(u64, bool, bool)>, String> {
        bindings
            .iter()
            .map(|b| {
                let slot = b
                    .owner
                    .state
                    .timeline
                    .try_lock()
                    .map_err(|_| "arena control family locked")?;
                let t = slot.as_ref().ok_or("arena control timeline missing")?;
                Ok((t.phases, t.recorded, t.poisoned))
            })
            .collect()
    };
    let before = flags()?;
    let proof = ModelBindingProof::checked(bindings, identity)?;
    {
        let tx = ModelArenaAdmission::from_proof(&proof, stream)?;
        if tx.arenas.len() != bindings.len() || tx.arenas.len() != 2 {
            return Err("distinct recurrence arenas not completely admitted".into());
        }
        for b in bindings {
            if b.owner.arena.lease().is_ok() {
                return Err("complete admission did not retain every arena".into());
            }
        }
    }
    let shared = [
        ModelFamilyBinding {
            layer: 0,
            owner: shared_a,
            producer,
        },
        ModelFamilyBinding {
            layer: 1,
            owner: shared_b,
            producer,
        },
    ];
    let shared_identity = model_family_identity(&shared)?;
    {
        let shared_proof = ModelBindingProof::checked(&shared, &shared_identity)?;
        let tx = ModelArenaAdmission::from_proof(&shared_proof, stream)?;
        if tx.arenas.len() != 1 {
            return Err("shared recurrence arena not deduplicated".into());
        }
    }
    let locked = bindings[1].owner.arena.lease()?;
    let (error, worker) = request_worker(
        device,
        "second-arena-complete-admission-rejection",
        || -> Result<String, String> {
            match ModelArenaAdmission::from_proof(&proof, stream) {
                Err(error) if error == "arena enqueue already leased/poisoned" => Ok(error),
                Err(error) => Err(format!("unexpected second arena rejection: {error}")),
                Ok(_) => Err("leased second recurrence arena admitted".into()),
            }
        },
    )?;
    // The worker acquired the first arena before rejection. It must have
    // released it and every family guard without entering execution.
    drop(bindings[0].owner.arena.lease()?);
    drop(locked);
    let after = flags()?;
    if after != before {
        return Err("rejected arena admission changed family timeline".into());
    }
    {
        let tx = ModelArenaAdmission::from_proof(&proof, stream)?;
        if tx.arenas.len() != 2 {
            return Err("arena admission after rejection incomplete".into());
        }
    }
    owner_save(root, "complete-arena-guards.json", &serde_json::to_vec_pretty(&json!({
        "scope":"recurrence arenas only; producer workspace and cross-family GPU reuse not qualified",
        "distinct_arena_count":2,"shared_arena_count":1,
        "all_distinct_guards_retained":true,"partial_rejection_releases_first":true,
        "execution_callback_entered":false,"gpu_submission_after_rejected_admission":false,
        "before":before,"after":after,"rejection":error,"worker":worker,
        "arenas":bindings.iter().map(|b| Arc::as_ptr(&b.owner.arena) as usize).collect::<Vec<_>>()
    })).map_err(|e|e.to_string())?)?;
    Ok(())
}
