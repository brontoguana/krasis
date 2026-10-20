// One ownership gate and completion event for this actual device/model split.
// Each Store builds its own gate; no global GPU index or GPU-count assumption.
struct ServingExecutionDomain {
    event: sys::CUevent,
    streams: Vec<Arc<NativeStream>>,
    recorded: bool,
    poisoned: bool,
}
unsafe impl Send for ServingExecutionDomain {}
impl Drop for ServingExecutionDomain {
    fn drop(&mut self) {
        for s in &self.streams {
            if let Err(e) = s.drain() {
                crate::vram_monitor::fatal_cuda_context_error(
                    "selected execution domain completion",
                    &e,
                );
            }
        }
        let rc = unsafe { sys::lib().cuEventDestroy_v2(self.event) };
        if rc != sys::CUresult::CUDA_SUCCESS {
            crate::vram_monitor::fatal_cuda_context_error(
                "selected execution domain event release",
                &format!("{rc:?}"),
            );
        }
    }
}
impl KdaEngineRequestStorage {
    pub(crate) fn enable_serving_execution(storage: &mut Arc<Self>) -> Result<(), String> {
        let this = Arc::get_mut(storage).ok_or("selected execution must precede publication")?;
        let state = this
            .inner
            .get_mut()
            .map_err(|_| "selected execution storage poisoned")?;
        let mut streams: Vec<Arc<NativeStream>> = Vec::new();
        for (p, d) in current_sequence_owners(&state.lifetime)? {
            for s in [&p.stream, &d.stream] {
                s.validate()?;
                if !streams.iter().any(|prior| prior.handle == s.handle) {
                    streams.push(s.clone());
                }
            }
        }
        let mut event = std::ptr::null_mut();
        check(
            unsafe {
                sys::lib().cuEventCreate(
                    &mut event,
                    sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32,
                )
            },
            "selected execution event create",
        )?;
        *this
            .serving_execution
            .get_mut()
            .map_err(|_| "selected execution gate poisoned")? = Some(ServingExecutionDomain {
            event,
            streams,
            recorded: false,
            poisoned: false,
        });
        Ok(())
    }
    pub(crate) fn with_serving_execution<T>(
        &self,
        stream: sys::CUstream,
        work: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let mut slot = self
            .serving_execution
            .try_lock()
            .map_err(|_| "selected execution concurrently held/poisoned")?;
        let domain = slot
            .as_mut()
            .ok_or("selected execution gate not constructed")?;
        let source = domain
            .streams
            .iter()
            .find(|s| s.handle == stream)
            .ok_or("selected execution stream absent")?;
        source.validate()?;
        if domain.poisoned {
            return Err("selected execution state requires explicit recovery".into());
        }
        if domain.recorded {
            check(
                unsafe { sys::lib().cuStreamWaitEvent(stream, domain.event, 0) },
                "selected execution preceding completion",
            )?;
        }
        domain.poisoned = true;
        let result = work();
        // This runs after graph recording AND actual execution, never inside
        // stream capture. The next phase waits on real submitted work.
        let completion = check(
            unsafe { sys::lib().cuEventRecord(domain.event, stream) },
            "selected execution completion record",
        );
        if let Err(e) = completion {
            for s in &domain.streams {
                if let Err(wait) = s.drain() {
                    crate::vram_monitor::fatal_cuda_context_error(
                        "selected failed execution completion",
                        &wait,
                    );
                }
            }
            return Err(e);
        }
        domain.recorded = true;
        match result {
            Ok(value) => {
                domain.poisoned = false;
                Ok(value)
            }
            Err(error) => {
                // Explicit failed-sequence invalidation, not successful output
                // recovery. The original request still returns its actual error.
                for s in &domain.streams {
                    if let Err(wait) = s.drain() {
                        crate::vram_monitor::fatal_cuda_context_error(
                            "selected failed sequence completion",
                            &wait,
                        );
                    }
                }
                self.reset_completed_serving_state()?;
                domain.recorded = false;
                domain.poisoned = false;
                Err(error)
            }
        }
    }
    fn reset_completed_serving_state(&self) -> Result<(), String> {
        let mut state = self
            .inner
            .try_lock()
            .map_err(|_| "selected reset while phase held/poisoned")?;
        if request_storage_static_identity(&state.lifetime)? != state.static_identity {
            return Err("selected reset changed persistent storage identity".into());
        }
        let KdaEngineRequestStorageState {
            lifetime,
            resource_domain,
            poisoned,
            ..
        } = &mut *state;
        let owners = current_sequence_owners(lifetime)?;
        // Validate every live source/state/arena before any reset write.
        let mut timelines = owners
            .iter()
            .map(|(p, _)| {
                p.state
                    .timeline
                    .try_lock()
                    .map_err(|_| "selected reset family held/poisoned".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut arenas: Vec<EnqueueLease<'_>> = Vec::new();
        for (p, d) in &owners {
            for o in [*p, *d] {
                o.stream.validate()?;
                for x in &o.native_admissions {
                    native_pointer_admission(&o.stream, x.ptr, x.requested_bytes)?;
                }
                if !arenas
                    .iter()
                    .any(|a| std::ptr::eq(a.arena, o.arena.as_ref()))
                {
                    arenas.push(o.arena.lease()?);
                }
                o.validate_lease(model_arena_lease(o, &arenas)?)?;
            }
        }
        let mut resources = resource_domain
            .groups
            .iter()
            .map(|g| {
                g.timeline
                    .try_lock()
                    .map_err(|_| "selected reset resources held/poisoned".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (p, _) in &owners {
            p.reset_in_phase(&StatePhase {
                owner: p,
                stream: &p.stream,
            })?;
        }
        for (p, _) in &owners {
            if let Err(e) = p.stream.drain() {
                crate::vram_monitor::fatal_cuda_context_error(
                    "selected canonical reset completion",
                    &e,
                );
            }
        }
        for t in &mut timelines {
            let t = t.as_mut().ok_or("selected reset timeline absent")?;
            t.recorded = false;
            t.poisoned = false;
        }
        for t in &mut resources {
            t.recorded = false;
            t.poisoned = false;
        }
        *poisoned = false;
        Ok(())
    }
}
