// Current retained admission interfaces; real owners and CUDA completion.
impl GpuDecodeStore {
    pub(crate) fn session_dma_device_identity_rust(&self) -> Result<(usize, usize), String> {
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        Ok((
            self.device.ordinal(),
            *self.device.cu_primary_ctx() as usize,
        ))
    }
    pub(crate) fn session_dma_matches_device_rust(
        &self,
        ordinal: usize,
        context: usize,
    ) -> Result<bool, String> {
        let actual = self.session_dma_device_identity_rust()?;
        Ok(actual == (ordinal, context))
    }
    pub(crate) fn complete_session_dma_stream_rust(&self) -> Result<(), String> {
        if let Err(error) = self.device.bind_to_thread() {
            crate::vram_monitor::fatal_cuda_context_error(
                "current sequence DMA completion bind",
                &error.to_string(),
            )
        }
        let rc = unsafe { cuda_sys::lib().cuStreamSynchronize(self.copy_stream.0) };
        if let Err(error) = rc.result() {
            crate::vram_monitor::fatal_cuda_context_error(
                "current sequence DMA completion",
                &error.to_string(),
            )
        }
        Ok(())
    }
    pub(crate) fn sequence_state_registry_rust(
        &self,
    ) -> &crate::session_cache::SequenceStateRegistry {
        &self.sequence_state_registry
    }
    pub(crate) fn diagnostic_sequence_state_hashes_rust(
        &self,
    ) -> Result<Vec<serde_json::Value>, String> {
        cold_kda_complete_state_hashes(&self.sequence_state_registry)
    }
    pub(crate) fn diagnostic_model_logits_rust(
        &self,
    ) -> Result<&cudarc::driver::CudaSlice<f32>, String> {
        let graph = self
            .graph
            .as_ref()
            .ok_or("installed guard actual model graph absent")?;
        if graph.vocab_size == 0 || graph.d_logits.len() != graph.vocab_size {
            return Err("installed guard actual vocab/logit extent changed".into());
        }
        Ok(&graph.d_logits)
    }
    pub(crate) fn diagnostic_capture_installed_graphs_rust(
        &mut self,
    ) -> Result<serde_json::Value, String> {
        if crate::prefix_recurrence::configured_plan()?.is_some() {
            return Err("installed graph diagnostic excludes selected original carry".into());
        }
        let graph = self
            .graph
            .as_ref()
            .ok_or("installed graph configuration absent")?;
        if !graph.graphs_ever_captured
            || graph.per_layer_graphs_valid
            || !graph.per_layer_graphs.is_empty()
            || graph.hot_hcs_full_graph.is_some()
        {
            return Err("installed graph admission requires retired calibrated graphs".into());
        }
        if graph.num_layers != graph.layers.len() || graph.segment_layer_start != 0 {
            return Err("installed graph admission requires a complete configured store".into());
        }
        let position = graph
            .captured_decode_start_position
            .ok_or("installed graph calibrated capture position missing")?;
        let before = serde_json::json!({"position":graph.kv_current_pos,"rope":graph.rope_position_delta,"captured_start":position,"graphs_ever_captured":graph.graphs_ever_captured,"per_layer_count":graph.per_layer_graphs.len(),"per_layer_valid":graph.per_layer_graphs_valid,"hot_graph_present":graph.hot_hcs_full_graph.is_some()});
        self.capture_per_layer_graphs(position, None, true, true, 0)?;
        self.device.synchronize().map_err(|e| e.to_string())?;
        self.complete_session_dma_stream_rust()?;
        let graph = self.graph.as_ref().ok_or("installed captured graph lost")?;
        if !graph.per_layer_graphs_valid
            || graph.per_layer_graphs.is_empty()
            || !graph.graphs_ever_captured
            || graph.captured_decode_start_position != Some(position)
            || before["position"].as_u64() != Some(graph.kv_current_pos as u64)
            || before["rope"] != serde_json::json!(graph.rope_position_delta)
        {
            return Err("installed graph capture metadata incomplete".into());
        }
        Ok(
            serde_json::json!({"passed":true,"before":before,"after":{"position":graph.kv_current_pos,"rope":graph.rope_position_delta,"captured_start":graph.captured_decode_start_position,"graphs_ever_captured":graph.graphs_ever_captured,"per_layer_count":graph.per_layer_graphs.len(),"per_layer_valid":graph.per_layer_graphs_valid,"hot_graph_present":graph.hot_hcs_full_graph.is_some()},"capture_existing_implementation":true,"graph_replay_qualified":false}),
        )
    }
}
pub(crate) fn cold_kda_complete_state_hashes(
    registry: &crate::session_cache::SequenceStateRegistry,
) -> Result<Vec<serde_json::Value>, String> {
    use sha2::Digest;
    registry.allocations().iter().map(|allocation|{
 let mut bytes=vec![0u8;allocation.storage_bytes];
 let rc=unsafe{cuda_sys::lib().cuMemcpyDtoH_v2(bytes.as_mut_ptr().cast(),allocation.ptr,bytes.len())};
 if rc!=cuda_sys::CUresult::CUDA_SUCCESS{return Err(format!("cold complete persistent read:{}:{rc:?}",allocation.name));}
 Ok(serde_json::json!({"allocation":allocation,"bytes":bytes.len(),"sha256":format!("{:x}",sha2::Sha256::digest(&bytes))}))
 }).collect()
}
