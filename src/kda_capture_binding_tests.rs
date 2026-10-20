fn capture_storage_identity_controls(
    root: &Path,
    device: &Arc<CudaDevice>,
    bindings: &[ModelFamilyBinding<'_>],
    stream: &NativeStream,
    expected: &Value,
    twin: &KdaRawProducer,
    rebound: &PreparedOwner,
) -> Result<(), String> {
    let flags = || -> Result<Vec<(u64, bool, bool)>, String> {
        bindings
            .iter()
            .map(|b| {
                let s = b
                    .owner
                    .state
                    .timeline
                    .try_lock()
                    .map_err(|_| "binding control family locked")?;
                let t = s.as_ref().ok_or("binding control timeline missing")?;
                Ok((t.phases, t.recorded, t.poisoned))
            })
            .collect()
    };
    let before = flags()?;
    let original = bindings[0].producer;
    if original.arithmetic_sha256 != twin.arithmetic_sha256
        || original.raw.inventory() != twin.raw.inventory()
        || original.prepared.schema != twin.prepared.schema
        || original.prepared.addresses() == twin.prepared.addresses()
        || pointer(&original.a_log) == pointer(&twin.a_log)
        || pointer(&original.bias) == pointer(&twin.bias)
        || !Arc::ptr_eq(&rebound.state, &bindings[1].owner.state)
        || Arc::ptr_eq(&rebound.arena, &bindings[1].owner.arena)
    {
        return Err("capture binding discriminator actual storage not distinct/compatible".into());
    }
    let old_expected = model_family_identity_frozen307(bindings)?;
    let substituted_producer = [
        ModelFamilyBinding {
            layer: bindings[0].layer,
            owner: bindings[0].owner,
            producer: twin,
        },
        ModelFamilyBinding {
            layer: bindings[1].layer,
            owner: bindings[1].owner,
            producer: twin,
        },
    ];
    let substituted_arena = [
        ModelFamilyBinding {
            layer: bindings[0].layer,
            owner: bindings[0].owner,
            producer: original,
        },
        ModelFamilyBinding {
            layer: bindings[1].layer,
            owner: rebound,
            producer: bindings[1].producer,
        },
    ];
    let mut cases = Vec::new();
    for (label, other) in [
        (
            "different prepared/parameter storage",
            &substituted_producer[..],
        ),
        (
            "same state different nonprimary arena",
            &substituted_arena[..],
        ),
    ] {
        let old_actual = model_family_identity_frozen307(other)?;
        if old_actual != old_expected {
            return Err("frozen307 storage identity mechanism not reproduced".into());
        }
        // Exact retained307 checked method must actually acquire/release every
        // family guard, not just compare two reconstructed JSON values.
        drop(ModelFamilyAdmission::checked_frozen307(
            other,
            stream,
            &old_expected,
        )?);
        let current = model_family_identity(other)?;
        if &current == expected {
            return Err("new capture identity omitted substituted workspace".into());
        }
        let proof_error = ModelBindingProof::checked(other, expected)
            .err()
            .ok_or("substituted immutable binding proof admitted")?;
        if proof_error != "complete graph family identity or inventory incomplete" {
            return Err("unexpected immutable proof rejection".into());
        }
        let (error, worker) = request_worker(
            device,
            "captured-storage-substitution-rejection",
            || -> Result<String, String> {
                match ModelArenaAdmission::checked(other, stream, expected) {
                    Err(e) if e == "complete graph family identity or inventory incomplete" => {
                        Ok(e)
                    }
                    Err(e) => Err(format!("unexpected storage rejection: {e}")),
                    Ok(_) => Err("substituted captured storage admitted".into()),
                }
            },
        )?;
        cases.push(json!({"label":label,"old_identity_exact":true,"old_complete_admission_succeeded":true,
            "new_identity_different":true,"new_admission_rejected_before_callback":true,
            "error":error,"worker":worker,"proof_construction_rejected":true,"proof_error":proof_error,"substituted_identity":current}));
    }
    let after = flags()?;
    if before != after {
        return Err("capture substitution changed committed timeline".into());
    }
    drop(ModelArenaAdmission::checked(bindings, stream, expected)?);
    owner_save(root,"capture-storage-guards.json",&serde_json::to_vec_pretty(&json!({
        "scope":"capture pointer/layout binding, not producer concurrency or model attribution",
        "original_identity":expected,"frozen307_identity":old_expected,"cases":cases,"before":before,"after":after,
        "ordinary_binding_readmitted":true,"gpu_replay_of_substituted_binding":false,
        "original_producer":model_workspace_identity(&bindings[0])?["producer"],
        "twin_producer":model_workspace_identity(&substituted_producer[0])?["producer"]
    })).map_err(|e|e.to_string())?)?;
    Ok(())
}
