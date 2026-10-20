// Explicit prepared-owner failure transport. Caller must order complete model
// state phases and final logits writes before this guard and the existing DMA.
// Construction/admission are cold; graph replay reads live GPU errors only.
struct PreparedLogitGuard {
    states: Vec<Arc<PreparedPersistentState>>,
    device: Arc<CudaDevice>,
    streams: Vec<Arc<NativeStream>>,
    kernel_bank: Arc<PreparedArena>,
    table: GpuBuf<u64>,
    arithmetic_sha256: String,
    allocation_bytes: usize,
}
impl PreparedLogitGuard {
    fn new(
        owners: &[&PreparedOwner],
        streams: Vec<Arc<NativeStream>>,
        reserve: usize,
    ) -> Result<Self, String> {
        let first = *owners
            .first()
            .ok_or("prepared logits guard needs selected owners")?;
        let count = i32::try_from(owners.len()).map_err(|_| "prepared logits guard owner count")?;
        if count <= 0 || streams.is_empty() {
            return Err("prepared logits guard empty inventory".into());
        }
        let mut addresses = Vec::with_capacity(owners.len());
        for owner in owners {
            if !Arc::ptr_eq(&owner.device, &first.device)
                || owner.arithmetic_sha256 != first.arithmetic_sha256
            {
                return Err("prepared logits guard owner context/arithmetic mismatch".into());
            }
            let address = pointer(&owner.error);
            if address == 0 || addresses.contains(&address) {
                return Err("prepared logits guard duplicate/null persistent error".into());
            }
            addresses.push(address);
        }
        for (i, stream) in streams.iter().enumerate() {
            stream.validate()?;
            if !Arc::ptr_eq(&stream.device, &first.device)
                || streams[..i].iter().any(|x| x.handle == stream.handle)
            {
                return Err("prepared logits guard stream context/duplicate mismatch".into());
            }
        }
        if owners
            .iter()
            .any(|x| !streams.iter().any(|s| s.handle == x.stream.handle))
        {
            return Err("prepared logits guard owner stream unadmitted".into());
        }
        if !first
            .arena
            .functions
            .contains_key("kda_prepared_logits_guard")
        {
            return Err("prepared logits guard kernel absent".into());
        }
        let allocation_bytes = owners
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or("prepared logits guard table overflow")?;
        if memory()?[0]
            < allocation_bytes
                .checked_add(reserve)
                .ok_or("prepared logits guard reserve overflow")?
        {
            return Err("prepared logits guard measured allocation admission failed".into());
        }
        let table = native_zeroed::<u64>(&first.stream, owners.len())?;
        first.stream.drain()?;
        check(
            unsafe {
                sys::lib().cuMemcpyHtoD_v2(
                    pointer(&table),
                    addresses.as_ptr().cast(),
                    allocation_bytes,
                )
            },
            "prepared logits guard table upload",
        )?;
        if memory()?[0] < reserve {
            return Err("prepared logits guard actual allocation floor".into());
        }
        Ok(Self {
            states: owners.iter().map(|x| x.state.clone()).collect(),
            device: first.device.clone(),
            streams,
            kernel_bank: first.arena.clone(),
            table,
            arithmetic_sha256: first.arithmetic_sha256.clone(),
            allocation_bytes,
        })
    }
    fn validate_owners(&self, owners: &[&PreparedOwner]) -> Result<(), String> {
        if owners.len() != self.states.len()
            || owners.iter().zip(&self.states).any(|(owner, state)| {
                !Arc::ptr_eq(&owner.state, state)
                    || !Arc::ptr_eq(&owner.device, &self.device)
                    || owner.arithmetic_sha256 != self.arithmetic_sha256
            })
        {
            return Err("prepared logits guard selected persistent owners changed".into());
        }
        Ok(())
    }
    fn required_pointers(&self) -> Vec<u64> {
        std::iter::once(pointer(&self.table))
            .chain(self.states.iter().map(|x| pointer(&x.error)))
            .collect()
    }
    // SAFETY: caller retains captured graph/table and owns/orders ALL selected
    // state writes and final logits. This is never a substitute for model phase
    // admission. stream and logits are owned typed allocations, not bare FFI.
    unsafe fn launch(
        &self,
        stream: &Arc<NativeStream>,
        logits: &GpuBuf<f32>,
    ) -> Result<(), String> {
        if !self.streams.iter().any(|x| Arc::ptr_eq(x, stream))
            || !Arc::ptr_eq(&stream.device, &self.device)
            || logits.len() == 0
        {
            return Err("prepared logits guard unadmitted stream/empty logits".into());
        }
        stream.validate()?;
        native_pointer_admission(
            stream,
            pointer(logits),
            logits
                .len()
                .checked_mul(4)
                .ok_or("prepared logits extent overflow")?,
        )?;
        invoke(
            &self.device,
            stream.handle,
            self.kernel_bank
                .functions
                .get("kda_prepared_logits_guard")
                .ok_or("prepared logits guard kernel changed")?,
            (1, 1, 1),
            1,
            0,
            &mut [
                Parameter::Ptr(pointer(&self.table)),
                Parameter::Ptr(pointer(logits)),
                Parameter::Int(self.states.len() as i32),
            ],
        )
    }
}
impl Drop for PreparedLogitGuard {
    fn drop(&mut self) {
        // Cold ordered teardown. Streams retain their actual owners; no token
        // execution or graph replay pays a host synchronization here.
        for stream in &self.streams {
            if let Err(error) = stream.drain() {
                crate::vram_monitor::fatal_cuda_context_error("selected logits guard completion", &error);
            }
        }
    }
}
// Borrowed graph ownership prevents table/logit/state teardown before graph
// destruction. The guard itself retains actual selected persistent owners.
struct PreparedGuardGraph<'a> {
    guard: &'a PreparedLogitGuard,
    _logits: &'a GpuBuf<f32>,
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
}
impl<'a> PreparedGuardGraph<'a> {
    unsafe fn capture(
        guard: &'a PreparedLogitGuard,
        stream: &Arc<NativeStream>,
        logits: &'a GpuBuf<f32>,
    ) -> Result<Self, String> {
        let mut result = Self {
            guard,
            _logits: logits,
            graph: std::ptr::null_mut(),
            exec: std::ptr::null_mut(),
        };
        stream.drain()?;
        check(
            sys::lib().cuStreamBeginCapture_v2(
                stream.handle,
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            ),
            "prepared guard capture begin",
        )?;
        let run = guard.launch(stream, logits);
        let end = check(
            sys::lib().cuStreamEndCapture(stream.handle, &mut result.graph),
            "prepared guard capture end",
        );
        run?;
        end?;
        let mut count = 0usize;
        check(
            sys::lib().cuGraphGetNodes(result.graph, std::ptr::null_mut(), &mut count),
            "prepared guard captured node count",
        )?;
        if count != 1 {
            return Err("prepared guard capture must contain exactly one kernel".into());
        }
        let mut node = std::ptr::null_mut();
        check(
            sys::lib().cuGraphGetNodes(result.graph, &mut node, &mut count),
            "prepared guard captured node",
        )?;
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        check(
            sys::lib().cuGraphNodeGetType(node, &mut kind),
            "prepared guard captured kind",
        )?;
        if kind != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
            return Err("prepared guard captured node not kernel".into());
        }
        check(
            sys::lib().cuGraphInstantiateWithFlags(&mut result.exec, result.graph, 0),
            "prepared guard instantiate",
        )?;
        Ok(result)
    }
    unsafe fn replay(&self, stream: &Arc<NativeStream>) -> Result<(), String> {
        if !self.guard.streams.iter().any(|x| Arc::ptr_eq(x, stream)) {
            return Err("prepared guard graph unadmitted replay stream".into());
        }
        stream.validate()?;
        check(
            sys::lib().cuGraphLaunch(self.exec, stream.handle),
            "prepared guard graph replay",
        )
    }
}
impl Drop for PreparedGuardGraph<'_> {
    fn drop(&mut self) {
        for stream in &self.guard.streams {
            stream
                .drain()
                .expect("prepared guard graph teardown must drain admitted streams");
        }
        unsafe {
            if !self.exec.is_null() {
                check(
                    sys::lib().cuGraphExecDestroy(self.exec),
                    "prepared guard exec teardown",
                )
                .expect("prepared guard graph exec destruction");
            }
            if !self.graph.is_null() {
                check(
                    sys::lib().cuGraphDestroy(self.graph),
                    "prepared guard graph teardown",
                )
                .expect("prepared guard graph destruction");
            }
        }
    }
}
