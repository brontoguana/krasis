// Graph lifetimes are part of the Rust core, not forensic readback.
struct OwnerGraph<'a> {
    owner: &'a PreparedOwner,
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
}
// SAFETY: exclusive graph ownership may move to another bound request thread.
// Retained owner borrow keeps operands/modules/streams alive. No Sync trait is
// provided: CUDA graph calls, including inventory, require exclusive access.
unsafe impl Send for OwnerGraph<'_> {}
impl Drop for OwnerGraph<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.owner.stream.drain() {
            eprintln!("owner graph drain:{e}");
        }
        unsafe {
            if !self.exec.is_null() {
                let rc = sys::lib().cuGraphExecDestroy(self.exec);
                if rc != sys::CUresult::CUDA_SUCCESS {
                    eprintln!("owner graph exec destroy:{rc:?}");
                }
            }
            if !self.graph.is_null() {
                let rc = sys::lib().cuGraphDestroy(self.graph);
                if rc != sys::CUresult::CUDA_SUCCESS {
                    eprintln!("owner graph destroy:{rc:?}");
                }
            }
        }
    }
}

struct KdaRawOwnerGraph<'a> {
    inner: OwnerGraph<'a>,
    _producer: &'a KdaRawProducer,
}
impl<'a> std::ops::Deref for KdaRawOwnerGraph<'a> {
    type Target = OwnerGraph<'a>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl Drop for KdaRawOwnerGraph<'_> {
    fn drop(&mut self) {
        // Teardown only. An error is explicit: never free graph operands after a failed drain.
        let slot = self
            .inner
            .owner
            .state
            .timeline
            .lock()
            .expect("raw graph state-family mutex poisoned during teardown");
        if let Some(timeline) = slot.as_ref() {
            if timeline.recorded {
                check(
                    unsafe { sys::lib().cuEventSynchronize(timeline.event) },
                    "raw graph retained timeline completion",
                )
                .expect("raw graph final event completion failed");
            }
            for stream in &timeline.streams {
                stream
                    .drain()
                    .expect("raw graph admitted replay stream teardown failed");
            }
        }
    }
}
fn kda_raw_capture<'a>(
    o: &'a PreparedOwner,
    p: &'a KdaRawProducer,
) -> Result<KdaRawOwnerGraph<'a>, String> {
    let lease = o.arena.lease()?;
    o.validate_lease(&lease)?;
    let mut graph = OwnerGraph {
        owner: o,
        graph: std::ptr::null_mut(),
        exec: std::ptr::null_mut(),
    };
    check(
        unsafe {
            sys::lib().cuStreamBeginCapture_v2(
                o.stream.handle,
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        },
        "raw graph capture begin",
    )?;
    let run = (|| {
        p.enqueue(0, 1, o)?;
        o.decode_capture_in(&lease, &p.view(1)?)
    })();
    let end = check(
        unsafe { sys::lib().cuStreamEndCapture(o.stream.handle, &mut graph.graph) },
        "raw graph capture end",
    );
    run?;
    end?;
    check(
        unsafe { sys::lib().cuGraphInstantiateWithFlags(&mut graph.exec, graph.graph, 0) },
        "raw graph instantiate",
    )?;
    Ok(KdaRawOwnerGraph {
        inner: graph,
        _producer: p,
    })
}
