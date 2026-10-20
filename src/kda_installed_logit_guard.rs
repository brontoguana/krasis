// Startup-only actual-model binding qualification. No logits enter sampling.
// Allocation/pool admission belongs outside stream capture. This immutable
// binding retains the exact checked backing and stream until graph teardown.
struct InstalledLogitBinding<'a> {
    guard: &'a PreparedLogitGuard,
    _logits: &'a CudaSlice<f32>,
    stream: Arc<NativeStream>,
    admission: NativePointerAdmission,
    owners: i32,
}
impl<'a> InstalledLogitBinding<'a> {
    fn checked(
        guard: &'a PreparedLogitGuard,
        stream: &Arc<NativeStream>,
        logits: &'a CudaSlice<f32>,
    ) -> Result<Self, String> {
        if !guard.streams.iter().any(|x| Arc::ptr_eq(x, stream))
            || !Arc::ptr_eq(&stream.device, &guard.device)
            || logits.len() == 0
        {
            return Err("installed guard unadmitted typed logits/stream".into());
        }
        stream.validate()?;
        let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        check(
            unsafe { sys::lib().cuStreamIsCapturing(stream.handle, &mut status) },
            "installed logit admission capture status",
        )?;
        if status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            return Err("installed logits must be admitted before capture".into());
        }
        let bytes = logits
            .len()
            .checked_mul(4)
            .ok_or("installed logits extent overflow")?;
        let admission = native_pointer_admission(stream, logits.address(), bytes)?;
        let owners =
            i32::try_from(guard.states.len()).map_err(|_| "installed guard count overflow")?;
        if owners == 0
            || !guard
                .kernel_bank
                .functions
                .contains_key("kda_prepared_logits_guard")
        {
            return Err("installed guard binding kernel/owners absent".into());
        }
        Ok(Self {
            guard,
            _logits: logits,
            stream: stream.clone(),
            admission,
            owners,
        })
    }
    // SAFETY: caller owns the complete current phase; this binding borrows
    // actual logits/guard owners and retains the admitted stream. No driver
    // allocation/pool query may be introduced between begin/end capture.
    unsafe fn launch(&self) -> Result<(), String> {
        invoke(
            &self.guard.device,
            self.stream.handle,
            self.guard
                .kernel_bank
                .functions
                .get("kda_prepared_logits_guard")
                .ok_or("installed guard kernel absent")?,
            (1, 1, 1),
            1,
            0,
            &mut [
                Parameter::Ptr(pointer(&self.guard.table)),
                Parameter::Ptr(self.admission.ptr),
                Parameter::Int(self.owners),
            ],
        )
    }
}
struct InstalledGuardGraph<'a> {
    binding: InstalledLogitBinding<'a>,
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
}
impl<'a> InstalledGuardGraph<'a> {
    unsafe fn capture(binding: InstalledLogitBinding<'a>) -> Result<Self, String> {
        let mut result = Self {
            binding,
            graph: std::ptr::null_mut(),
            exec: std::ptr::null_mut(),
        };
        let stream = &result.binding.stream;
        stream.drain()?;
        check(
            sys::lib().cuStreamBeginCapture_v2(
                stream.handle,
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            ),
            "installed guard begin capture",
        )?;
        let launch = result.binding.launch();
        let end = check(
            sys::lib().cuStreamEndCapture(stream.handle, &mut result.graph),
            "installed guard end capture",
        );
        launch?;
        end?;
        let mut count = 0usize;
        check(
            sys::lib().cuGraphGetNodes(result.graph, std::ptr::null_mut(), &mut count),
            "installed guard node count",
        )?;
        if count != 1 {
            return Err("installed guard graph must contain one kernel".into());
        }
        let mut node = std::ptr::null_mut();
        check(
            sys::lib().cuGraphGetNodes(result.graph, &mut node, &mut count),
            "installed guard node",
        )?;
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        check(
            sys::lib().cuGraphNodeGetType(node, &mut kind),
            "installed guard kind",
        )?;
        if kind != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
            return Err("installed guard graph nonkernel".into());
        }
        check(
            sys::lib().cuGraphInstantiateWithFlags(&mut result.exec, result.graph, 0),
            "installed guard instantiate",
        )?;
        Ok(result)
    }
    unsafe fn replay(&self, stream: &Arc<NativeStream>) -> Result<(), String> {
        if !Arc::ptr_eq(&self.binding.stream, stream) {
            return Err("installed guard replay stream not retained".into());
        }
        stream.validate()?;
        check(
            sys::lib().cuGraphLaunch(self.exec, stream.handle),
            "installed guard replay",
        )
    }
}
impl Drop for InstalledGuardGraph<'_> {
    fn drop(&mut self) {
        for s in &self.binding.guard.streams {
            s.drain()
                .expect("installed guard graph teardown must complete");
        }
        unsafe {
            if !self.exec.is_null() {
                check(
                    sys::lib().cuGraphExecDestroy(self.exec),
                    "installed guard exec teardown",
                )
                .expect("installed guard exec destruction");
            }
            if !self.graph.is_null() {
                check(
                    sys::lib().cuGraphDestroy(self.graph),
                    "installed guard graph teardown",
                )
                .expect("installed guard graph destruction");
            }
        }
    }
}
impl KdaEngineRequestStorage {
    pub(crate) fn diagnostic_installed_logit_guard(
        &self,
        store: &mut crate::gpu_decode::GpuDecodeStore,
        admission: &KdaDiagnosticStateAdmission,
        output: &std::path::Path,
    ) -> Result<serde_json::Value, String> {
        let (registry, before) = self.audit()?;
        let original_before = store.diagnostic_sequence_state_hashes_rust()?;
        let added_before = crate::gpu_decode::cold_kda_complete_state_hashes(&registry)?;
        let admission_before = admission.audit(store.sequence_state_registry_rust(), &registry)?;
        let (records,epoch)=self.current_sequence_transfer(
            &before["current_identity"],before["domain_generation"].as_u64().ok_or("installed guard generation")?,
            before["live_sequence_epoch"].as_u64().ok_or("installed guard epoch")?,&mut[&mut *store],
            |life,stores|{
                let owners=current_sequence_owners(life)?;
                let selected:Vec<_>=owners.iter().map(|(p,_)|*p).collect();
                admission.guard.validate_owners(&selected)?;
                if selected.len()!=self.layers().len(){return Err("installed guard layer/owner inventory changed".into());}
                let guard=&admission.guard;
                let stream=guard.streams.first().ok_or("installed guard stream missing")?;
                let logits=stores[0].diagnostic_model_logits_rust()?;
                let bytes=logits.len().checked_mul(4).ok_or("installed guard complete logits extent")?;
                let baseline=shadow_read(logits.address(),bytes)?;
                let words:Vec<f32>=baseline.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect();
                crate::prefix_recurrence::check_prefix_logits(&words)?;
                for owner in &selected{
                    if shadow_read(pointer(&owner.error),4)?!=[0,0,0,0]{return Err("installed guard requires canonical healthy owner".into());}
                }
                std::fs::create_dir(output).map_err(|e|format!("installed guard fresh evidence:{e}"))?;
                let baseline_id=shadow_save(output,"actual-calibrated-logits.f32",&baseline)?;
                let binding=InstalledLogitBinding::checked(guard,stream,logits)?;
                let binding_admission=serde_json::to_value(&binding.admission).map_err(|e|e.to_string())?;
                let graph=unsafe{InstalledGuardGraph::capture(binding)}?;
                let mut cases=Vec::new();
                // Each real owner is tested, with a healthy boundary case at both ends.
                for selected_index in std::iter::once(None).chain((0..selected.len()).map(Some)).chain(std::iter::once(None)){
                    stream.drain()?;
                    if let Some(i)=selected_index{
                        check(unsafe{sys::lib().cuMemsetD32Async(pointer(&selected[i].error),1,1,stream.handle)},"installed guard inject live owner failure")?;
                    }
                    for mode in ["eager","captured"]{
                        // Each mode must independently replace a healthy baseline marker.
                        stream.drain()?;
                        check(unsafe{sys::lib().cuMemcpyHtoD_v2(logits.address(),baseline.as_ptr().cast(),bytes)},"installed guard independent actual logits baseline")?;
                        if mode=="eager"{unsafe{graph.binding.launch()}?;}else{unsafe{graph.replay(stream)}?;}
                        stream.drain()?;
                        let actual=shadow_read(logits.address(),bytes)?;
                        let mut expected=baseline.clone();
                        if selected_index.is_some(){expected[..4].copy_from_slice(&crate::prefix_recurrence::PREFIX_FAILURE_MARKER.to_le_bytes());}
                        let observed_id=shadow_save(output,&format!("case{}-{mode}.f32",cases.len()),&actual)?;
                        if actual!=expected{return Err(format!("installed guard complete logits mismatch case{} {mode}",cases.len()));}
                        let values:Vec<f32>=actual.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect();
                        let rejected=crate::prefix_recurrence::check_prefix_logits(&values).is_err();
                        if rejected!=selected_index.is_some(){return Err("installed guard marker consumer disagreement".into());}
                        cases.push(serde_json::json!({"mode":mode,"owner":selected_index,"layer":selected_index.map(|i|self.layers()[i]),"error_pointer":selected_index.map(|i|pointer(&selected[i].error)),"failure_rejected_before_sampling":rejected,"only_word0_marker_changed":selected_index.is_some(),"complete_vector":observed_id}));
                    }
                    if let Some(i)=selected_index{
                        check(unsafe{sys::lib().cuMemsetD32Async(pointer(&selected[i].error),0,1,stream.handle)},"installed guard clear live failure")?;
                    }
                }
                stream.drain()?;
                check(unsafe{sys::lib().cuMemcpyHtoD_v2(logits.address(),baseline.as_ptr().cast(),bytes)},"installed guard restore calibrated logits on successful completion")?;
                stream.drain()?;
                if shadow_read(logits.address(),bytes)?!=baseline{return Err("installed guard calibrated logits restoration mismatch".into());}
                drop(graph);
                Ok(serde_json::json!({"passed":true,"cases":cases,"actual_logit_elements":logits.len(),"baseline":baseline_id,"all_actual_error_owners":selected.len(),"guard_table_bytes":guard.allocation_bytes,"actual_borrowed_model_logits_no_gpu_clone":true,"captured_kernel_count":1,"retained_logits_admission":binding_admission,"allocation_admitted_before_capture":true}))
            })?;
        let (_, after) = self.audit()?;
        let original_after = store.diagnostic_sequence_state_hashes_rust()?;
        let added_after = crate::gpu_decode::cold_kda_complete_state_hashes(&registry)?;
        let admission_after = admission.audit(store.sequence_state_registry_rust(), &registry)?;
        if original_before != original_after
            || added_before != added_after
            || admission_before != admission_after
        {
            return Err(
                "installed guard changed complete original/current state or admitted inventory"
                    .into(),
            );
        }
        for key in [
            "domain_generation",
            "static_identity",
            "current_identity",
            "events",
            "prefill_writer_phases",
            "shadow_records",
        ] {
            if before[key] != after[key] {
                return Err(format!("installed guard changed source identity:{key}"));
            }
        }
        if after["live_sequence_epoch"].as_u64() != Some(epoch)
            || before["live_sequence_epoch"]
                .as_u64()
                .and_then(|x| x.checked_add(1))
                != Some(epoch)
        {
            return Err("installed guard completed phase epoch".into());
        }
        let old = before["resource_domain"]["groups"]
            .as_array()
            .ok_or("installed guard initial resource groups")?;
        let new = after["resource_domain"]["groups"]
            .as_array()
            .ok_or("installed guard final resource groups")?;
        if old.len() != new.len() {
            return Err("installed guard resource coverage changed".into());
        }
        for (x, y) in old.iter().zip(new) {
            let mut expected = x.clone();
            expected["phases"] = serde_json::json!(x["phases"]
                .as_u64()
                .and_then(|v| v.checked_add(1))
                .ok_or("installed guard resource phase overflow")?);
            expected["recorded"] = serde_json::json!(false);
            if expected != *y {
                return Err("installed guard complete resource phases changed".into());
            }
        }
        Ok(
            serde_json::json!({"passed":true,"control":records,"before":before,"after":after,"original_before":original_before,"original_after":original_after,"new_before":added_before,"new_after":added_after,"admission_before":admission_before,"admission_after":admission_after,"all_streams_completed_under_whole_current_guards":true,"selected_for_generation":false,"whole_model_graph_ordering_qualified":false}),
        )
    }
}
