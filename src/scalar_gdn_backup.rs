// Persistent speculative backups are allocated by the existing measured batch
// setup. These are the additional real native regions; legacy conv/latest
// backups remain owned by the actual graph and join the same execution phase.
struct Backup {
    regions: Vec<crate::session_cache::SequenceStateAllocation>,
    buffers: Vec<CudaSlice<u8>>,
    cursor: Option<usize>,
}
impl Backup {
    unsafe fn enqueue(&self, device: &Arc<CudaDevice>, restore: bool) -> Result<(), String> {
        for (region, buffer) in self.regions.iter().zip(&self.buffers) {
            if buffer.len() != region.storage_bytes {
                return Err("scalar backup actual allocation extent changed".into());
            }
            let (to, from) = if restore {
                (region.ptr, *buffer.device_ptr())
            } else {
                (*buffer.device_ptr(), region.ptr)
            };
            sys::lib()
                .cuMemcpyDtoDAsync_v2(to, from, region.storage_bytes, *device.cu_stream())
                .result()
                .map_err(|e| format!("scalar native backup {}:{e}", region.name))?;
        }
        Ok(())
    }
}
impl Storage {
    pub(crate) fn configure_speculative_backup(&self) -> Result<usize, String> {
        unsafe {
            self.execution.quiescent(||{
            let mut state=self.inner.try_lock().map_err(|_|"scalar backup setup state held/poisoned")?;
            if state.active.is_some(){return Err("scalar backup setup inside an active invocation".into());}
            if let Some(backup)=&state.backup{
                if backup.buffers.len()!=backup.regions.len(){return Err("scalar backup setup is incomplete; failed allocations remain retained through completion".into());}
                return backup.regions.iter().try_fold(0usize,|n,r|n.checked_add(r.storage_bytes).ok_or("scalar backup byte count overflow".into()));
            }
            let device=state.program.device.clone();device.bind_to_thread().map_err(|e|e.to_string())?;
            let regions=self.registry.allocations().to_vec();
            state.backup=Some(Backup{buffers:Vec::with_capacity(regions.len()),regions,cursor:None});
            let backup=state.backup.as_mut().ok_or("scalar backup setup owner absent")?;
            let mut bytes=0usize;
            for region in &backup.regions {
                region.validate()?;
                bytes=bytes.checked_add(region.storage_bytes).ok_or("scalar backup byte count overflow")?;
                backup.buffers.push(device.alloc_zeros::<u8>(region.storage_bytes).map_err(|e|e.to_string())?);
            }
            Ok(bytes)
        })
        }
    }
    /// # Safety
    /// Work copies every legacy conv/latest owner on this actual decode stream.
    /// All real graph/state/backup owners outlive the enclosing execution guard.
    pub(crate) unsafe fn save_speculative<T>(
        &self,
        cursor: usize,
        work: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        self.execution.execute(&self.decode, || {
            let mut state = self
                .inner
                .try_lock()
                .map_err(|_| "scalar backup save state held/poisoned")?;
            let device = state.program.device.clone();
            let backup = state.backup.as_mut().ok_or(
                "scalar speculative backup was not configured during measured batch setup",
            )?;
            backup.cursor = None;
            backup.enqueue(&device, false)?;
            let value = work()?;
            backup.cursor = Some(cursor);
            Ok(value)
        })
    }
    /// # Safety
    /// Same complete legacy family and stream obligations as save_speculative.
    /// Partial restore/error/unwind keeps the execution poisoned.
    pub(crate) unsafe fn restore_speculative<T>(
        &self,
        work: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        self.execution.recover(|| {
            let mut state = self
                .inner
                .try_lock()
                .map_err(|_| "scalar backup restore state held/poisoned")?;
            let backup = state
                .backup
                .as_ref()
                .ok_or("scalar speculative backup is absent")?;
            let cursor = backup
                .cursor
                .ok_or("scalar speculative backup has no successful save")?;
            backup.enqueue(&state.program.device, true)?;
            let value = work()?;
            // Metadata validation completes the real device stream before
            // reading words; the owning recover guard also drains every exit.
            for owner in state.layers.values_mut() {
                owner.validate_restored_completed(cursor)?;
            }
            Self::read_health_completed(&state, &self.decode)?;
            Ok(value)
        })
    }
}
