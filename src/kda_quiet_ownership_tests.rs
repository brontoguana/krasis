// Private quiet ownership-cost fixture; zero production selector.
struct QuietBurstEvents { start:sys::CUevent, end:sys::CUevent }
impl QuietBurstEvents {
    fn new()->Result<Self,String>{
        let mut s=Self{start:std::ptr::null_mut(),end:std::ptr::null_mut()};
        check(unsafe{sys::lib().cuEventCreate(&mut s.start,0)},"quiet start event")?;
        check(unsafe{sys::lib().cuEventCreate(&mut s.end,0)},"quiet end event")?;Ok(s)
    }
}
impl Drop for QuietBurstEvents {fn drop(&mut self){unsafe{for event in [self.start,self.end]{if !event.is_null(){let rc=sys::lib().cuEventDestroy_v2(event);assert_eq!(rc,sys::CUresult::CUDA_SUCCESS,"quiet timing event destroy");}}}}}
fn capture_quiet_model_families<'a>(bindings:&'a[ModelFamilyBinding<'a>],emissions:&'a[ModelFamilyEmission],stream:&NativeStream)->Result<ModelFamilyGraph<'a>,String>{
    if emissions.len()!=bindings.len(){return Err("quiet emission family count".into());}
    let identity=model_family_identity(bindings)?;
    let mut graph=OwnerGraph{owner:bindings.first().ok_or("quiet empty family")?.owner,graph:std::ptr::null_mut(),exec:std::ptr::null_mut()};
    let mut admission=ModelFamilyAdmission::checked(bindings,stream,&identity)?;
    admission.capture_only(||{
        check(unsafe{sys::lib().cuStreamBeginCapture_v2(stream.handle,sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)},"quiet capture begin")?;
        let run=(||->Result<(),String>{for (b,out) in bindings.iter().zip(emissions){
            if b.owner.stream.handle!=stream.handle{return Err("quiet capture owner stream".into());}
            let lease=b.owner.arena.lease()?;b.owner.validate_lease(&lease)?;
            b.producer.enqueue(0,1,b.owner)?;b.owner.decode_capture_in(&lease,&b.producer.view(1)?)?;
            let hd=b.owner.heads.checked_mul(b.owner.dim).ok_or("quiet output geometry")?;
            if out.bf16.len()!=hd{return Err("quiet BF16 extent".into());}
            timeline_copy(stream,pointer(&out.bf16),pointer(&b.owner.arena.bf16_emitted),hd.checked_mul(2).ok_or("quiet output byte extent")?)?;
        }Ok(())})();
        let end=check(unsafe{sys::lib().cuStreamEndCapture(stream.handle,&mut graph.graph)},"quiet capture end");run?;end?;
        check(unsafe{sys::lib().cuGraphInstantiateWithFlags(&mut graph.exec,graph.graph,0)},"quiet instantiate")
    })?;Ok(ModelFamilyGraph{inner:graph,bindings,emissions,identity})
}
fn quiet_ownership_compare(
    root:&Path, bindings:&[ModelFamilyBinding<'_>],graph:&ModelFamilyGraph<'_>,events:&QuietBurstEvents,
    stream:&NativeStream,prefill:&NativeStream,batches:[&PreparedOwner;2],producer:&KdaRawProducer,
    output:[&GpuBuf<u8>;2],registries:[&crate::session_cache::SequenceStateRegistry;2],
    start:usize,t:usize,capacity:usize,expected_bf:&[u8],want:&OwnerSnapshot,reserve:usize,
)->Result<Value,String>{
    if bindings.len()!=2||graph.bindings.len()!=2||t<=start{return Err("quiet family/suffix scope".into());}
    stream.validate()?;
    if crate::vram_monitor::native_optional_monitor_active(){return Err("quiet diagnostic sampler still active".into());}
    let hd=bindings[0].owner.heads.checked_mul(bindings[0].owner.dim).ok_or("quiet geometry")?;
    let bytes=hd.checked_mul(2).ok_or("quiet output stride")?;
    let source=producer.raw.inventory();let destination=bindings[0].producer.raw.inventory();
    if source.len()!=destination.len(){return Err("quiet raw inventory extent".into());}
    let mut copies=Vec::new();for ((src_desc,src),(dst_desc,dst)) in source.into_iter().zip(destination){
        let stride=src_desc.strides_bytes[0];if stride!=dst_desc.storage_bytes{return Err("quiet raw stride".into());}
        native_pointer_admission(stream,src,src_desc.storage_bytes)?;native_pointer_admission(stream,dst,stride)?;copies.push((src,dst,stride));
    }
    for (dst,emission) in output.iter().zip(graph.emissions){native_pointer_admission(stream,pointer(*dst),dst.len())?;native_pointer_admission(stream,pointer(&emission.bf16),bytes)?;}
    let proof=ModelBindingProof::checked(bindings,&graph.identity)?;
    let mut rows=Vec::new();
    for pair in 0..6 {let order=if pair%2==0{["per_replay","held"]}else{["held","per_replay"]};for mode in order {
        for batch in batches{batch.ordered_phase(prefill,|phase|phase.reset())?;}
        let mut at=0;while at<start{let count=capacity.min(start-at);for batch in batches{batch.ordered_phase(prefill,|phase|phase.prefill(producer,at,count))?;}at+=count;}
        prefill.drain()?; // Prefix completion outside the measured suffix interval.
        let whole_start=std::time::Instant::now();
        let admission_start=std::time::Instant::now();
        let mut held=if mode=="held"{Some(ModelArenaAdmission::from_proof(&proof,stream)?)}else{None};
        let admission_ns=if held.is_some(){admission_start.elapsed().as_nanos()}else{0};
        check(unsafe{sys::lib().cuEventRecord(events.start,stream.handle)},"quiet start record")?;
        let submit_start=std::time::Instant::now();
        for row in start..t {
            let work=|leases:&[EnqueueLease<'_>]|->Result<(),String>{
                for &(src,dst,stride) in &copies{check(unsafe{sys::lib().cuMemcpyDtoDAsync_v2(dst,src+(row*stride)as u64,stride,stream.handle)},"quiet required input")?;}
                bindings[0].owner.ordered_graph_in_lease(&StatePhase{owner:bindings[0].owner,stream},&graph.inner,model_arena_lease(bindings[0].owner,leases)?)?;
                for (dst,emission) in output.iter().zip(graph.emissions){check(unsafe{sys::lib().cuMemcpyDtoDAsync_v2(pointer(*dst)+(row*bytes)as u64,pointer(&emission.bf16),bytes,stream.handle)},"quiet required BF16 delivery")?;}Ok(())
            };
            if let Some(tx)=held.as_mut(){tx.execute(work)?;}else{ModelArenaAdmission::from_proof(&proof,stream)?.execute(work)?;}
        }
        let submit_ns=submit_start.elapsed().as_nanos();
        check(unsafe{sys::lib().cuEventRecord(events.end,stream.handle)},"quiet end record")?;
        let completion_start=std::time::Instant::now();check(unsafe{sys::lib().cuEventSynchronize(events.end)},"quiet required complete suffix")?;
        let completion_ns=completion_start.elapsed().as_nanos();
        let mut gpu_ms=0f32;check(unsafe{sys::lib().cuEventElapsedTime(&mut gpu_ms,events.start,events.end)},"quiet GPU elapsed")?;
        drop(held);
        let host_complete_ns=whole_start.elapsed().as_nanos(); // Includes initial admission, event timing calls, final wait and held lock release.
        // Validation uses state registry only after exclusive admission release.
        let dir=root.join(format!("quiet-pair-{pair}-{mode}"));fs::create_dir(&dir).map_err(|e|e.to_string())?;
        let mut retained=Vec::new();for (family,((b,dst),registry)) in bindings.iter().zip(output).zip(registries).enumerate(){
            let mut bf=vec![0u8;(t-start)*bytes];check(unsafe{sys::lib().cuMemcpyDtoH_v2(bf.as_mut_ptr().cast(),pointer(dst)+(start*bytes)as u64,bf.len())},"quiet validation suffix")?;
            owner_compare(&bf,&expected_bf[start*bytes..t*bytes],"quiet complete suffix")?;
            let actual=capture_owner(b.owner,registry)?;
            if actual.layout!=want.layout{return Err("quiet complete final schema".into());}
            owner_compare(&serde_json::to_vec(&actual.blobs).map_err(|e|e.to_string())?,&serde_json::to_vec(&want.blobs).map_err(|e|e.to_string())?,"quiet complete final state")?;
            let suffix=owner_save(&dir,&format!("family-{family}-suffix.bf16"),&bf)?;
            let mut state=Vec::new();let mut schema=Vec::new();for blob in actual.blobs {let offset=state.len();let extent=blob.bytes.len();state.extend_from_slice(&blob.bytes);let mut meta=serde_json::to_value(&blob).map_err(|e|e.to_string())?;meta.as_object_mut().ok_or("quiet blob metadata")?.remove("bytes");schema.push(json!({"offset":offset,"bytes":extent,"metadata":meta}));}
            let state=owner_save(&dir,&format!("family-{family}-final-state.bin"),&state)?;
            let schema=owner_save(&dir,&format!("family-{family}-state-schema.json"),&serde_json::to_vec(&json!({"layout":actual.layout,"arithmetic":actual.arithmetic_sha256,"blobs":schema})).map_err(|e|e.to_string())?)?;
            retained.push(json!({"suffix":suffix,"state":state,"schema":schema}));long_check_cursor(b.owner,t)?;
        }
        let free=memory()?[0];if free<reserve{return Err("quiet postcompletion reserve violated".into());}
        rows.push(json!({"pair":pair,"warmup":pair==0,"mode":mode,"suffix_rows":t-start,"initial_admission_ns":admission_ns,"cpu_submission_ns":submit_ns,"cpu_submission_with_admission_ns":submit_ns+admission_ns,"host_burst_through_completion_and_release_ns":host_complete_ns,"required_completion_wait_ns":completion_ns,"gpu_event_span_ms":gpu_ms,"postcompletion_free_bytes":free,"retained":retained}));
    }}
    let v=json!({"optional_instrumentation":false,"external_sampler":false,"native_sampler_stopped":true,"forensic_copies_in_interval":false,"per_token_timers":false,"prefix_batched":true,"prefix_rows":start,"suffix_rows":t-start,"mode_scope":"complete checked immutable storage proof; fresh per-replay locks versus held locks, no identity reconstruction per token; private cost control only","gpu_span_includes_submission_gaps":true,"rows":rows});
    owner_save(root,"quiet-performance.json",&serde_json::to_vec_pretty(&v).map_err(|e|e.to_string())?)?;Ok(v)
}
