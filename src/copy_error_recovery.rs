// Explicit installed diagnostic only. No ordinary request or token path uses it.
thread_local! {
    static COPY_RECOVERY: std::cell::RefCell<std::collections::HashMap<usize, (usize, i32, Vec<crate::session_cache::SequenceStateBlob>)>> = std::cell::RefCell::new(std::collections::HashMap::new());
}
impl GpuDecodeStore {
    pub(crate) fn installed_copy_error_recovery_rust(
        &mut self,
        action: &str,
    ) -> Result<serde_json::Value, String> {
        use crate::copy_fault::{self, Fault};
        let key = self as *mut Self as usize;
        match action {
            "poison" => {
                if COPY_RECOVERY.with(|m| m.borrow().contains_key(&key)) {
                    return Err("diagnostic recovery already pending".into());
                }
                let position = self.sequence_position_rust()?;
                let tokens = position.kv_absolute_position;
                let original = self.snapshot_sequence_state_rust(tokens)?.0;
                let refs = original.iter().collect::<Vec<_>>();
                copy_fault::arm(0, Fault::EventError);
                let result =
                    self.restore_sequence_state_rust(tokens, &refs, position.rope_position_delta);
                let (enqueues, before) = copy_fault::take();
                let after = unsafe { cuda_sys::lib().cuStreamQuery(self.copy_stream.0) };
                let expected =
                    matches!(&result,Err(error) if error.contains("post-enqueue event error"));
                // Save before evaluating assertions so a failed diagnostic cannot lose recovery.
                COPY_RECOVERY.with(|m| {
                    m.borrow_mut()
                        .insert(key, (tokens, position.rope_position_delta, original))
                });
                if !expected
                    || enqueues != 1
                    || after != cuda_sys::CUresult::CUDA_SUCCESS
                    || self.require_sequence_state_ready_rust().is_ok()
                {
                    return Err(format!("installed poison control failed: result={result:?},enqueues={enqueues},after={after:?}"));
                }
                Ok(
                    serde_json::json!({"poisoned":true,"enqueues":enqueues,"query_at_injection":before,"query_after_return":format!("{after:?}"),"tokens":tokens}),
                )
            }
            "recover" => {
                let value=COPY_RECOVERY.with(|m| -> Result<serde_json::Value,String> {
                    let m=m.borrow();let (tokens,rope,original)=m.get(&key).ok_or("no diagnostic recovery pending")?;
                    self.restore_sequence_state_rust(*tokens,&original.iter().collect::<Vec<_>>(),*rope)?;
                    self.require_sequence_state_ready_rust()?;
                    let recovered=self.snapshot_sequence_state_rust(*tokens)?.0;
                    if recovered.len()!=original.len() || !recovered.iter().zip(original).all(|(a,b)|a.allocation_name==b.allocation_name && a.bytes==b.bytes) {
                        return Err("diagnostic explicit recovery changed original bytes".into());
                    }
                    Ok(serde_json::json!({"recovered":true,"exact_original_bytes":true,"tokens":tokens}))
                })?;
                COPY_RECOVERY.with(|m| m.borrow_mut().remove(&key));
                Ok(value)
            }
            _ => Err("copy error recovery action must be poison or recover".into()),
        }
    }
}
