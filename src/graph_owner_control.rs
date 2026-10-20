// Explicit diagnostic build only. Uses the actual configured Store containers
// and their actual replacement/invalidation/destruction paths. Empty graphs
// isolate handle ownership; they make no configured-model capture claim.
impl GpuDecodeStore {
    #[cfg(feature = "scalar-gdn-test")]
    fn install_graph_owner_control_rust(&mut self) -> Result<String, String> {
        let graph = self
            .graph
            .as_ref()
            .ok_or("owner control requires configure")?;
        if graph.num_layers != 0
            || !graph.weights.is_empty()
            || !graph.per_layer_graphs.is_empty()
            || graph.hot_hcs_full_graph.is_some()
        {
            return Err(
                "owner control requires an empty, zero-layer Store; never a loaded model".into(),
            );
        }
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        let mut owners = Vec::new();
        for _ in 0..2 {
            let mut graph = OwnedCaptureGraph {
                pointer: std::ptr::null_mut(),
                device: self.device.clone(),
            };
            unsafe { cuda_sys::lib().cuGraphCreate(&mut graph.pointer, 0) }
                .result()
                .map_err(|e| format!("owner control graph:{e}"))?;
            let mut node = std::ptr::null_mut();
            unsafe {
                cuda_sys::lib().cuGraphAddEmptyNode(&mut node, graph.pointer, std::ptr::null(), 0)
            }
            .result()
            .map_err(|e| format!("owner control node:{e}"))?;
            let mut exec = OwnedCaptureExec {
                pointer: std::ptr::null_mut(),
                device: self.device.clone(),
            };
            let status = unsafe {
                cuda_sys::lib().cuGraphInstantiateWithFlags(&mut exec.pointer, graph.pointer, 0)
            };
            if status != cuda_sys::CUresult::CUDA_SUCCESS {
                if !exec.pointer.is_null() {
                    crate::vram_monitor::fatal_cuda_context_error(
                        "owner control partial executable",
                        &format!("{status:?}"),
                    );
                }
                return Err(format!("owner control instantiate:{status:?}"));
            }
            if exec.pointer.is_null() {
                return Err("owner control null executable".into());
            }
            owners.push(exec.into_store());
        }
        let full = owners.pop().ok_or("owner control full executable absent")?;
        let segment = owners.pop().ok_or("owner control segment absent")?;
        let handles = [format!("{:p}", segment.0), format!("{:p}", full.0)];
        // Real queued launches on the Store's actual stream. Store/graph Drop
        // must complete before releasing the executable/program containers.
        for exec in [&segment, &full] {
            unsafe { cuda_sys::lib().cuGraphLaunch(exec.0, self.copy_stream.0) }
                .result()
                .map_err(|e| format!("owner control launch:{e}"))?;
        }
        let graph = self.graph.as_mut().ok_or("owner control graph removed")?;
        graph.per_layer_graphs.push(segment);
        graph.hot_hcs_full_graph = Some(full);
        graph.per_layer_graphs_valid = true;
        Ok(serde_json::json!({"handles":handles,"device":self.device.ordinal(),"actual_store_containers":true,"model_layers":0,"scope":"Actual Store executable ownership only, not configured-model capture or queued-buffer safety qualification."}).to_string())
    }
}
