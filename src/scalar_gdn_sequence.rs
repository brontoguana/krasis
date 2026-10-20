// Whole-family sequence operations retain the existing installed copy-completion
// implementation. Only explicit native storage enters its owned timeline.
impl GpuDecodeStore {
    pub fn snapshot_sequence_state_rust(
        &mut self,
        logical_tokens: usize,
    ) -> Result<(Vec<crate::session_cache::SequenceStateBlob>, f64), String> {
        #[cfg(feature = "scalar-gdn-test")]
        if let Some(storage) = self.selected_scalar_gdn.clone() {
            return unsafe {
                storage.snapshot_boundary(logical_tokens, || {
                    self.snapshot_sequence_state_inner_rust(logical_tokens)
                })
            };
        }
        self.snapshot_sequence_state_inner_rust(logical_tokens)
    }
    pub fn snapshot_sequence_state_incremental_rust(
        &mut self,
        previous_logical_tokens: usize,
        logical_tokens: usize,
        previous_blobs: &[&crate::session_cache::SequenceStateBlob],
    ) -> Result<(Vec<crate::session_cache::SequenceStateBlob>, f64), String> {
        #[cfg(feature = "scalar-gdn-test")]
        if let Some(storage) = self.selected_scalar_gdn.clone() {
            return unsafe {
                storage.snapshot_boundary(logical_tokens, || {
                    self.snapshot_sequence_state_incremental_inner_rust(
                        previous_logical_tokens,
                        logical_tokens,
                        previous_blobs,
                    )
                })
            };
        }
        self.snapshot_sequence_state_incremental_inner_rust(
            previous_logical_tokens,
            logical_tokens,
            previous_blobs,
        )
    }
    pub fn restore_sequence_state_rust(
        &mut self,
        logical_tokens: usize,
        blobs: &[&crate::session_cache::SequenceStateBlob],
        rope_position_delta: i32,
    ) -> Result<f64, String> {
        #[cfg(feature = "scalar-gdn-test")]
        if let Some(storage) = self.selected_scalar_gdn.clone() {
            // Inner validation checks the complete actual registry before any
            // mutation; recovery stays poisoned if copy or metadata fails.
            return unsafe {
                storage.restore(logical_tokens, || {
                    self.restore_sequence_state_inner_rust(
                        logical_tokens,
                        blobs,
                        rope_position_delta,
                    )
                })
            };
        }
        self.restore_sequence_state_inner_rust(logical_tokens, blobs, rope_position_delta)
    }
    pub fn reset_sequence_state_for_diagnostic_rust(&mut self) -> Result<usize, String> {
        #[cfg(feature = "scalar-gdn-test")]
        if let Some(storage) = self.selected_scalar_gdn.clone() {
            return unsafe {
                storage.restore(0, || self.reset_sequence_state_for_diagnostic_inner_rust())
            };
        }
        self.reset_sequence_state_for_diagnostic_inner_rust()
    }
}
impl GpuDecodeStore {
    fn save_la_states(&self, cursor: usize) -> Result<(), String> {
        #[cfg(feature = "scalar-gdn-test")]
        if let Some(storage) = &self.selected_scalar_gdn {
            return unsafe { storage.save_speculative(cursor, || self.save_la_states_inner()) };
        }
        #[cfg(not(feature = "scalar-gdn-test"))]
        let _ = cursor;
        self.save_la_states_inner()
    }
    fn restore_la_states(&self) -> Result<(), String> {
        #[cfg(feature = "scalar-gdn-test")]
        if let Some(storage) = &self.selected_scalar_gdn {
            return unsafe { storage.restore_speculative(|| self.restore_la_states_inner()) };
        }
        self.restore_la_states_inner()
    }
    unsafe fn copy_la_backup_region(
        &self,
        destination: u64,
        source: u64,
        bytes: usize,
    ) -> cuda_sys::CUresult {
        if self.scalar_gdn_requested {
            cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                destination,
                source,
                bytes,
                *self.device.cu_stream(),
            )
        } else {
            cuda_sys::lib().cuMemcpyDtoD_v2(destination, source, bytes)
        }
    }
}
impl GpuDecodeStore {
    pub fn capture_pending_active_sequence_checkpoint_rust(
        &mut self,
        logical_tokens: usize,
    ) -> Result<(usize, f64), String> {
        #[cfg(feature = "scalar-gdn-test")]
        if let Some(storage) = self.selected_scalar_gdn.clone() {
            return unsafe {
                storage.snapshot_boundary(logical_tokens, || {
                    self.capture_pending_active_sequence_checkpoint_inner_rust(logical_tokens)
                })
            };
        }
        self.capture_pending_active_sequence_checkpoint_inner_rust(logical_tokens)
    }
    pub fn restore_active_sequence_checkpoint_rust(
        &mut self,
        logical_tokens: usize,
    ) -> Result<f64, String> {
        #[cfg(feature = "scalar-gdn-test")]
        if let Some(storage) = self.selected_scalar_gdn.clone() {
            return unsafe {
                storage.restore(logical_tokens, || {
                    self.restore_active_sequence_checkpoint_inner_rust(logical_tokens)
                })
            };
        }
        self.restore_active_sequence_checkpoint_inner_rust(logical_tokens)
    }
}
