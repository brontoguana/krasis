// Model compatibility is stable across ordinary resets; request epochs are separate.
impl RetainedKdaSessionModel {
    fn validate_live_sequence_epoch(&self, expected: u64) -> Result<(), String> {
        if expected != self.live_sequence_epoch {
            return Err("stale retained live sequence epoch".into());
        }
        Ok(())
    }
    fn reset_sequence_quiescent(
        &mut self,
        current: &serde_json::Value,
        expected_live_epoch: u64,
        stores: &mut [&mut crate::gpu_decode::GpuDecodeStore],
    ) -> Result<u64, String> {
        self.validate_live_sequence_epoch(expected_live_epoch)?;
        let next = expected_live_epoch
            .checked_add(1)
            .ok_or("live sequence epoch overflow")?;
        self.session_transfer(current, stores, |_| {
            // Complete inventory/context/family/arena admission precedes every write.
            for family in &self.families {
                family
                    .prefill
                    .device
                    .bind_to_thread()
                    .map_err(|e| e.to_string())?;
                family.prefill.reset()?;
            }
            Ok(())
        })?;
        // All actual contexts completed resets; only now retire old request admission.
        self.live_sequence_epoch = next;
        Ok(next)
    }
}
