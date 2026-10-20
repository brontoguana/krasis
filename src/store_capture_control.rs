// Explicit cold diagnostic build only. Hooks are thread-local to the serialized
// real Store invocation; they never substitute a segment or a configured graph.
#[derive(Clone, Copy)]
struct StoreCaptureFailure {
    mode: &'static str,
    pointer: u64,
    bytes: usize,
    event: cuda_sys::CUevent,
    graph_index: usize,
}
std::thread_local! {
    static STORE_CAPTURE_FAILURE: std::cell::Cell<Option<StoreCaptureFailure>> = const { std::cell::Cell::new(None) };
}
struct StoreCaptureFailureReset;
impl Drop for StoreCaptureFailureReset {
    fn drop(&mut self) {
        STORE_CAPTURE_FAILURE.with(|mode| mode.set(None));
    }
}
impl GpuDecodeStore {
    fn capture_store_control_point(&self, point: &str, graph_index: usize) -> Result<(), String> {
        STORE_CAPTURE_FAILURE.with(|mode| {
            let Some(control) = mode.get() else {
                return Ok(());
            };
            let selected = control.mode;
            let hit = selected == point
                || point == "during-capture"
                    && matches!(selected, "unwind" | "invalidated-capture");
            if !hit || point == "during-capture" && graph_index != control.graph_index {
                return Ok(());
            }
            // Queue at the actual failure boundary, after any successful setup
            // synchronization. Test owners remain retained by the caller.
            unsafe {
                cuda_sys::lib()
                    .cuMemsetD8Async(control.pointer, 37, control.bytes, self.copy_stream.0)
                    .result()
                    .map_err(|e| e.to_string())?;
                cuda_sys::lib()
                    .cuEventRecord(control.event, self.copy_stream.0)
                    .result()
                    .map_err(|e| e.to_string())?;
            }
            if selected == point {
                return Err(format!("207 explicit Store failure:{selected}"));
            }
            if point == "during-capture" {
                if selected == "unwind" {
                    panic!("207 explicit actual Store capture unwind");
                }
                if selected == "invalidated-capture" {
                    let rc =
                        unsafe { cuda_sys::lib().cuStreamSynchronize(*self.device.cu_stream()) };
                    if rc != cuda_sys::CUresult::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED {
                        return Err(format!(
                            "207 invalidation returned unexpected status:{rc:?}"
                        ));
                    }
                    return Err("207 explicit Store failure:invalidated-capture".into());
                }
            }
            Ok(())
        })
    }
    /// Called only by the existing serialized model worker, after an ordinary
    /// request has populated the actual model/cache/autotune state. The genuine
    /// full state snapshot is restored before each recovery capture.
    pub(crate) fn installed_store_capture_control_rust(
        &mut self,
        expected_logical_tokens: usize,
    ) -> Result<serde_json::Value, String> {
        if STORE_CAPTURE_FAILURE.with(|m| m.get().is_some()) {
            return Err("nested Store capture control".into());
        }
        let _reset = StoreCaptureFailureReset;
        let saved_position = self.sequence_position_rust()?;
        let tokens = saved_position.kv_absolute_position;
        if tokens != expected_logical_tokens {
            return Err(format!("Store capture requires the actual completed cursor:requested={expected_logical_tokens},actual={tokens}"));
        }
        if self.last_stream_failure.is_some() {
            return Err("Store capture cannot snapshot a failed generation".into());
        }
        if tokens == 0 {
            return Err("Store control requires actual completed nonempty model request".into());
        }
        let (saved, _) = self.snapshot_sequence_state_rust(tokens)?;
        let original = *self.device.cu_stream();
        let mut original_blas = std::ptr::null_mut();
        let rc = unsafe {
            cublas_sys::lib().cublasGetStream_v2(*self.blas.handle(), &mut original_blas)
        };
        if rc != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(format!("actual original Store cuBLAS stream:{rc:?}"));
        }
        let (position, range, embedding, final_layer, offset) = {
            let g = self.graph.as_ref().ok_or("actual Store graph missing")?;
            if !g.per_layer_graphs_valid || g.per_layer_graphs.is_empty() {
                return Err("Store control requires actual qualified captured model graphs".into());
            }
            let range = if g.decode_layer_start == 0 && g.decode_layer_end == g.num_layers {
                None
            } else {
                Some((g.decode_layer_start, g.decode_layer_end))
            };
            (
                g.captured_decode_start_position
                    .ok_or("actual capture start absent")?,
                range,
                g.decode_layer_start == 0,
                g.decode_layer_end == g.num_layers,
                g.segment_gqa_cache_offset,
            )
        };
        let final_graph_index = self.graph.as_ref().unwrap().per_layer_graphs.len() - 1;
        let graph_address = &**self.graph.as_ref().unwrap() as *const GpuDecodeGraph as usize;
        let mut cases = Vec::new();
        for mode in [
            "before-capture",
            "after-enqueue",
            "during-capture",
            "unwind",
            "invalidated-capture",
        ] {
            self.restore_sequence_state_rust(
                tokens,
                &saved.iter().collect::<Vec<_>>(),
                saved_position.rope_position_delta,
            )?;
            // Independent queued work on the actual Store's copy stream tests
            // completion, not merely success of a later synchronizing read.
            let sentinel = self
                .device
                .alloc_zeros::<u8>(4096)
                .map_err(|e| e.to_string())?;
            let event = ScopedCudaEvent::new("207 completion sentinel")?;
            let initialized = ScopedCudaEvent::new("213 sentinel allocation and initialization")?;
            let completion_device = self.device.clone();
            let copy_stream = self.copy_stream.0;
            let retained = (&sentinel, &event, &initialized);
            unsafe {
                crate::sequence_copy_completion::with_copy_completion(
                    &completion_device,
                    copy_stream,
                    &retained,
                    |_| {
                        // alloc_zeros is asynchronous on the actual device stream. Order
                        // allocation/zero completion before the later copy-stream37 write.
                        // That write still occurs only at the injected failure boundary;
                        // the post-return event query precedes any synchronizing read.
                        unsafe {
                            cuda_sys::lib()
                                .cuEventRecord(initialized.0, *self.device.cu_stream())
                                .result()
                                .map_err(|e| format!("sentinel initialization record:{e}"))?;
                            cuda_sys::lib()
                                .cuStreamWaitEvent(copy_stream, initialized.0, 0)
                                .result()
                                .map_err(|e| format!("sentinel initialization dependency:{e}"))?;
                        }
                        STORE_CAPTURE_FAILURE.with(|m| {
                            m.set(Some(StoreCaptureFailure {
                                mode,
                                pointer: *sentinel.device_ptr(),
                                bytes: sentinel.len(),
                                event: event.0,
                                graph_index: final_graph_index,
                            }))
                        });
                        let failure =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                self.capture_per_layer_graphs(
                                    position,
                                    range,
                                    embedding,
                                    final_layer,
                                    offset,
                                )
                            }));
                        STORE_CAPTURE_FAILURE.with(|m| m.set(None));
                        match (mode, failure) {
                            ("unwind", Err(_)) => {}
                            (_, Ok(Err(error)))
                                if error == format!("207 explicit Store failure:{mode}") => {}
                            (_, other) => {
                                return Err(format!(
                                    "Store injection {mode} did not reach expected exit:{other:?}"
                                ))
                            }
                        }
                        if self
                            .graph
                            .as_ref()
                            .map(|g| &**g as *const GpuDecodeGraph as usize)
                            != Some(graph_address)
                        {
                            return Err(format!("Store lost actual graph owner:{mode}"));
                        }
                        // Query before read/copy/restore/sync: must already have completed.
                        unsafe { cuda_sys::lib().cuEventQuery(event.0) }
                            .result()
                            .map_err(|e| {
                                format!("Store released before queued completion:{mode}:{e}")
                            })?;
                        if *self.device.cu_stream() != original {
                            return Err(format!("Store device stream not restored:{mode}"));
                        }
                        let mut stream = std::ptr::null_mut();
                        let rc = unsafe {
                            cublas_sys::lib().cublasGetStream_v2(*self.blas.handle(), &mut stream)
                        };
                        if rc != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                            || stream != original_blas
                        {
                            return Err(format!("Store cuBLAS stream not restored:{mode}"));
                        }
                        let mut status =
                            cuda_sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
                        unsafe { cuda_sys::lib().cuStreamIsCapturing(original, &mut status) }
                            .result()
                            .map_err(|e| e.to_string())?;
                        if status != cuda_sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
                        {
                            return Err(format!("Store stream still capturing:{mode}"));
                        }
                        let actual = self
                            .device
                            .dtoh_sync_copy(&sentinel)
                            .map_err(|e| e.to_string())?;
                        if actual.iter().any(|&x| x != 37) {
                            return Err(format!("Store queued sentinel differs:mode={mode},zero_bytes={},sentinel_bytes={},total={}",actual.iter().filter(|&&x|x==0).count(),actual.iter().filter(|&&x|x==37).count(),actual.len()));
                        }
                        Ok(())
                    },
                )?;
            }
            drop(sentinel);
            self.restore_sequence_state_rust(
                tokens,
                &saved.iter().collect::<Vec<_>>(),
                saved_position.rope_position_delta,
            )?;
            self.capture_per_layer_graphs(position, range, embedding, final_layer, offset)?;
            if !self.graph.as_ref().unwrap().per_layer_graphs_valid {
                return Err(format!("Store recovery capture invalid:{mode}"));
            }
            cases.push(serde_json::json!({"mode":mode,"completed_before_sentinel_release":true,"same_graph_owner":true,"device_and_blas_restored":true,"recovery_capture":true}));
        }
        self.restore_sequence_state_rust(
            tokens,
            &saved.iter().collect::<Vec<_>>(),
            saved_position.rope_position_delta,
        )?;
        Ok(
            serde_json::json!({"device":self.device.ordinal(),"logical_tokens":tokens,"cases":cases,"scope":"Actual configured Store capture and recovery; ordinary HTTP generation recovery and complete numerical/state equality require subsequent gates."}),
        )
    }
}
