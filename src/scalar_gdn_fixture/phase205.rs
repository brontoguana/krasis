// Additional controls for the newly connected host-position, health, checkpoint
// and speculative APIs. Retained prepared operands and original references only.
fn revision205_controls(
    p:&Prepared,storage:&Storage,registry:&SequenceStateRegistry,
    source:&InputStorage,base:&Snapshot,full:&Snapshot,d:&Arc<CudaDevice>,root:&Path,
)->Result<(),String>{
    use crate::scalar_gdn_owner::DecodePosition;
    let row_bytes=p.h.checked_mul(p.v).and_then(|n|n.checked_mul(2)).ok_or("205 row extent overflow")?;
    base.restore(storage,registry,None)?;
    let mut output=Vec::new();
    for row in p.base..p.rows {
        let emitted=unsafe{storage.with_decode(||{
            let emitted=storage.decode_emitted(0,&source.view(row,1)?,&DecodePosition::Host(row as u64))?;
            storage.require_active_health()?;
            Ok(emitted)
        })?};
        storage.require_publication_health()?;
        output.extend(completed_output(storage,d,row+1,emitted.plane()?.pointer(),row_bytes)?);
    }
    record_output(root,"205-host-position-suffix.bf16",&output,&p.data["actual-output.bf16"][p.base*row_bytes..])?;
    let state=Snapshot::capture(storage,registry,p.rows,d)?;
    state.save(&root.join("205-host-position-state"))?;same_state(&state,full)?;
    base.restore(storage,registry,None)?;
    // Real checkpoint boundary while the outer prefill timeline is held. This
    // must neither recurse into the timeline nor expose unfinished GPU state.
    let pointer=unsafe{storage.with_prefill(||{
        storage.snapshot_boundary(p.base,||{
            let region=registry.allocations().iter().find(|r|r.name=="latest-state.f32").ok_or("205 latest region absent")?;
            let mut raw=vec![0u8;region.storage_bytes];
            check(sys::lib().cuMemcpyDtoH_v2(raw.as_mut_ptr().cast(),region.ptr,raw.len()),"205 active checkpoint read")?;
            compare("205 active boundary",&raw,base.plane("latest-state.f32")?)
        })?;
        storage.prefill_output(0,p.base as u64,p.rows-p.base,&source.view(p.base,p.rows-p.base)?)
    })?};
    let output=completed_output(storage,d,p.rows,pointer,(p.rows-p.base)*row_bytes)?;
    record_output(root,"205-active-boundary-suffix.bf16",&output,&p.data["actual-output.bf16"][p.base*row_bytes..])?;
    same_state(&Snapshot::capture(storage,registry,p.rows,d)?,full)?;
    base.restore(storage,registry,None)?;
    let backup_bytes=storage.configure_speculative_backup()?;
    let expected_bytes=storage.registry().allocations().iter().try_fold(0usize,|n,r|n.checked_add(r.storage_bytes).ok_or("205 backup extent overflow"))?;
    if backup_bytes!=expected_bytes{return Err("205 measured native backup bytes differ from actual registry".into());}
    let legacy=registry.allocations().iter().filter(|r|matches!(r.kind.as_str(),"linear_attention_conv_state"|"linear_attention_recurrent_state")).map(|r|Ok((r.clone(),d.alloc_zeros::<u8>(r.storage_bytes).map_err(fail)?))).collect::<Result<Vec<_>,String>>()?;
    let copy_legacy=|restore:bool,inject:bool|->Result<(),String>{
        for (region,buffer) in &legacy {
            let(to,from)=if restore{(region.ptr,*buffer.device_ptr())}else{(*buffer.device_ptr(),region.ptr)};
            check(unsafe{sys::lib().cuMemcpyDtoDAsync_v2(to,from,region.storage_bytes,*d.cu_stream())},"205 actual legacy backup")?;
            if inject{return Err("205 explicit error after actual legacy restore enqueue".into());}
        }
        Ok(())
    };
    unsafe{storage.save_speculative(p.base,||copy_legacy(false,false))?;}
    unsafe{storage.with_prefill(||storage.prefill_output(0,p.base as u64,p.rows-p.base,&source.view(p.base,p.rows-p.base)?))?;}
    unsafe{storage.restore_speculative(||copy_legacy(true,false))?;}
    let restored=Snapshot::capture(storage,registry,p.base,d)?;
    restored.save(&root.join("205-spec-restored-state"))?;
    for (region,_) in &base.regions{compare("205 exact speculative restore",restored.plane(&region.name)?,base.plane(&region.name)?)?;}
    let rejected=unsafe{storage.restore_speculative(||copy_legacy(true,true))}.err().ok_or("205 partial speculative restore accepted")?;
    if rejected!="205 explicit error after actual legacy restore enqueue"{return Err(rejected);}
    require_poison(storage,p.base)?;
    unsafe{storage.restore_speculative(||copy_legacy(true,false))?;}
    storage.require_publication_health()?;
    // A wrong actual eager position is a GPU semantic error, not an injected
    // CPU failure. Its sticky health must reject output before publication.
    let failure=unsafe{storage.with_decode(||{
        storage.decode_emitted(0,&source.view(p.base,1)?,&DecodePosition::Host((p.base+1)as u64))?;
        storage.require_active_health()
    })}.err().ok_or("205 wrong position reached CPU publication")?;
    if !failure.contains("device semantic state is invalid"){return Err(failure);}
    if storage.require_publication_health().is_ok(){return Err("205 poisoned phase published logits".into());}
    require_poison(storage,p.base)?;
    base.restore(storage,registry,None)?;
    let health=registry.allocations().iter().find(|r|r.kind=="scalar_gdn_native64_health").ok_or("205 actual health region absent")?;
    check(unsafe{sys::lib().cuMemsetD32Async(health.ptr,1,1,*d.cu_stream())},"205 explicit sticky-health control")?;
    if storage.require_publication_health().is_ok(){return Err("205 sticky health fault was published".into());}
    require_poison(storage,p.base)?;
    base.restore(storage,registry,None)?;
    storage.require_publication_health()?;
    let mut stale_entered=false;
    let stale=unsafe{storage.restore_speculative(||{stale_entered=true;Ok(())})};
    if stale.is_ok()||stale_entered{return Err("205 unrelated restore left a speculative backup eligible".into());}
    require_poison(storage,p.base)?;
    base.restore(storage,registry,None)?;
    Snapshot::capture(storage,registry,p.base,d)?.save(&root.join("205-after-health-recovery-state"))?;
    fs::write(root.join("NATIVE205_CONTROLS.json"),serde_json::to_vec_pretty(&json!({
        "passed":true,"ordinary_serving_qualified":false,"host_position_suffix_exact":true,
        "active_prefill_checkpoint_and_continuation_exact":true,"native_backup_bytes":backup_bytes,
        "complete_speculative_restore_exact":true,"partial_speculative_restore_poisoned":true,
        "wrong_device_semantic_position_rejected_before_publication":true,
        "sticky_health_rejected_before_publication":true,"complete_restore_recovers_health":true,
        "unrelated_restore_invalidates_speculative_backup":true,
        "scope":"Actual205Storage with retained193 prepared operands and independent same-device native references. Does not qualify Store producer integration, cross-device handoff, model serving, Witness or speed."
    })).map_err(fail)?).map_err(fail)?;
    Ok(())
}
