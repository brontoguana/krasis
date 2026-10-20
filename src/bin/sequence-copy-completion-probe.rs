#[path = "../sequence_copy_completion.rs"]
mod completion;
use krasis::vram_monitor;
mod operator {
    include!("row-projection-compensated-probe.rs");
    use cudarc::driver::{sys, CudaSlice, CudaStream};
    use std::{
        panic::{catch_unwind, AssertUnwindSafe},
        sync::Arc,
    };
    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct CopyPlan {
        source: Identity,
        ptx: Identity,
        build_report: Identity,
        input: Identity,
        gpu_name: String,
        gpu_uuid: String,
        compute_capability: u32,
        safety_margin_bytes: usize,
        queue_delay_ms: u64,
    }
    fn checked(r: sys::CUresult, label: &str) -> Result<(), String> {
        r.result().map_err(|e| format!("{label}: {e}"))
    }
    struct Pinned {
        ptr: *mut u8,
        bytes: usize,
    }
    impl Drop for Pinned {
        fn drop(&mut self) {
            if let Err(e) = checked(
                unsafe { sys::lib().cuMemFreeHost(self.ptr.cast()) },
                "pinned release",
            ) {
                eprintln!("ERROR: {e}");
            }
        }
    }
    struct ForeignContext {
        primary: Arc<CudaDevice>,
        context: sys::CUcontext,
        stream: sys::CUstream,
    }
    impl Drop for ForeignContext {
        fn drop(&mut self) {
            if !self.context.is_null() {
                if let Err(e) = checked(
                    unsafe { sys::lib().cuCtxSetCurrent(self.context) },
                    "foreign cleanup bind",
                ) {
                    eprintln!("ERROR: {e}");
                }
                if !self.stream.is_null() {
                    if let Err(e) = checked(
                        unsafe { sys::lib().cuStreamDestroy_v2(self.stream) },
                        "foreign stream release",
                    ) {
                        eprintln!("ERROR: {e}");
                    }
                }
                if let Err(e) = checked(
                    unsafe { sys::lib().cuCtxDestroy_v2(self.context) },
                    "foreign context release",
                ) {
                    eprintln!("ERROR: {e}");
                }
                if let Err(e) = self.primary.bind_to_thread() {
                    eprintln!("ERROR: primary context restore: {e}");
                }
            }
        }
    }
    fn foreign_context_control(dev: &Arc<CudaDevice>, owners: &Owners) -> Result<(), String> {
        let mut foreign = ForeignContext {
            primary: dev.clone(),
            context: std::ptr::null_mut(),
            stream: std::ptr::null_mut(),
        };
        checked(
            unsafe { sys::lib().cuCtxCreate_v2(&mut foreign.context, 0, *dev.cu_device()) },
            "foreign context create",
        )?;
        checked(
            unsafe {
                sys::lib().cuStreamCreate(
                    &mut foreign.stream,
                    sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                )
            },
            "foreign stream create",
        )?;
        let mut called = false;
        let r = unsafe {
            crate::completion::with_copy_completion(dev, foreign.stream, owners, |_| {
                called = true;
                Ok(())
            })
        };
        if r.as_ref().map(|_| ()).map_err(String::as_str)
            != Err("copy phase stream/context mismatch")
            || called
        {
            return Err(format!(
                "foreign admission did not reject before callback: {r:?}/{called}"
            ));
        }
        drop(foreign);
        dev.bind_to_thread().map_err(|e| e.to_string())?;
        Ok(())
    }
    struct Owners {
        device: Arc<CudaDevice>,
        stream: Option<CudaStream>,
        pinned: [Option<Pinned>; 2],
        device_slots: [Option<CudaSlice<u8>>; 2],
    }
    impl Drop for Owners {
        fn drop(&mut self) {
            if let Err(e) = self.device.bind_to_thread() {
                eprintln!("ERROR: owner bind {e}");
            }
            if let Some(stream) = &self.stream {
                if let Err(e) = checked(
                    unsafe { sys::lib().cuStreamSynchronize(stream.stream) },
                    "owner drain",
                ) {
                    eprintln!("ERROR: {e}");
                }
            }
        }
    }
    impl Owners {
        fn allocate(dev: Arc<CudaDevice>, bytes: usize) -> Result<Self, String> {
            let mut o = Self {
                device: dev,
                stream: None,
                pinned: [None, None],
                device_slots: [None, None],
            };
            o.stream = Some(o.device.fork_default_stream().map_err(|e| e.to_string())?);
            for i in 0..2 {
                let mut ptr = std::ptr::null_mut();
                checked(
                    unsafe { sys::lib().cuMemHostAlloc(&mut ptr, bytes, 0) },
                    "copy pinned allocation",
                )?;
                o.pinned[i] = Some(Pinned {
                    ptr: ptr.cast(),
                    bytes,
                });
                o.device_slots[i] = Some(
                    o.device
                        .alloc_zeros::<u8>(bytes)
                        .map_err(|e| e.to_string())?,
                );
            }
            o.device.synchronize().map_err(|e| e.to_string())?;
            Ok(o)
        }
        fn stream_handle(&self) -> sys::CUstream {
            self.stream
                .as_ref()
                .expect("owned stream initialized")
                .stream
        }
        fn fill_pinned(&self, payload: &[u8], guard: usize) -> Result<(), String> {
            for pin in &self.pinned {
                let p = pin.as_ref().ok_or("missing pinned owner")?;
                if p.bytes != payload.len() + guard {
                    return Err("pinned geometry".into());
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(payload.as_ptr(), p.ptr, payload.len());
                    std::ptr::write_bytes(p.ptr.add(payload.len()), 0xa5, guard);
                }
            }
            Ok(())
        }
        fn enqueue(&self, index: usize, direction: &str, bytes: usize) -> Result<(), String> {
            let pin = self.pinned[index].as_ref().ok_or("pinned owner")?;
            let d = self.device_slots[index].as_ref().ok_or("device owner")?;
            let r = unsafe {
                match direction {
                    "h2d" => sys::lib().cuMemcpyHtoDAsync_v2(
                        *d.device_ptr(),
                        pin.ptr.cast(),
                        bytes,
                        self.stream_handle(),
                    ),
                    "d2h" => sys::lib().cuMemcpyDtoHAsync_v2(
                        pin.ptr.cast(),
                        *d.device_ptr(),
                        bytes,
                        self.stream_handle(),
                    ),
                    _ => return Err("explicit direction required".into()),
                }
            };
            checked(r, "actual retained copy enqueue")
        }
        fn query(&self) -> sys::CUresult {
            unsafe { sys::lib().cuStreamQuery(self.stream_handle()) }
        }
        fn bytes(&self, index: usize, direction: &str) -> Result<Vec<u8>, String> {
            if direction == "h2d" {
                self.device
                    .dtoh_sync_copy(self.device_slots[index].as_ref().ok_or("device owner")?)
                    .map_err(|e| e.to_string())
            } else {
                let p = self.pinned[index].as_ref().ok_or("pinned owner")?;
                Ok(unsafe { std::slice::from_raw_parts(p.ptr, p.bytes) }.to_vec())
            }
        }
    }
    fn qualify(plan: &Path, out: &Path) -> Result<Value, String> {
        let p: CopyPlan = serde_json::from_slice(&fs::read(plan).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        if p.safety_margin_bytes != 600 * 1024 * 1024 || p.queue_delay_ms == 0 {
            return Err("explicit reserve/queue control required".into());
        }
        p.source.read()?;
        let ptx = p.ptx.read()?;
        let input = p.input.read()?;
        if input.is_empty() {
            return Err("empty actual state input".into());
        }
        let built: Value =
            serde_json::from_slice(&p.build_report.read()?).map_err(|e| e.to_string())?;
        if built["ptx"] != serde_json::to_value(&p.ptx).map_err(|e| e.to_string())?
            || built["plan"]["source"]
                != serde_json::to_value(&p.source).map_err(|e| e.to_string())?
        {
            return Err("copy build provenance".into());
        }
        let dev = CudaDevice::new_with_stream(0).map_err(|e| e.to_string())?;
        let mut uuid = unsafe { std::mem::zeroed::<sys::CUuuid>() };
        checked(
            unsafe { sys::lib().cuDeviceGetUuid_v2(&mut uuid, *dev.cu_device()) },
            "GPU UUID",
        )?;
        let h = uuid
            .bytes
            .iter()
            .map(|&b| format!("{:02x}", b as u8))
            .collect::<String>();
        let uuid = format!(
            "GPU-{}-{}-{}-{}-{}",
            &h[..8],
            &h[8..12],
            &h[12..16],
            &h[16..20],
            &h[20..]
        );
        use sys::CUdevice_attribute as A;
        let cc = dev
            .attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| e.to_string())?
            * 10
            + dev
                .attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                .map_err(|e| e.to_string())?;
        if uuid != p.gpu_uuid
            || cc as u32 != p.compute_capability
            || cc < 80
            || dev.name().map_err(|e| e.to_string())? != p.gpu_name
        {
            return Err("actual GPU admission differs".into());
        }
        dev.load_ptx(
            cudarc::nvrtc::Ptx::from_src(std::str::from_utf8(&ptx).map_err(|e| e.to_string())?),
            "copy_control",
            &["retained_copy_queue_delay"],
        )
        .map_err(|e| e.to_string())?;
        let delay = dev
            .get_func("copy_control", "retained_copy_queue_delay")
            .ok_or("queue control kernel absent")?;
        let rate = u64::try_from(
            dev.attribute(A::CU_DEVICE_ATTRIBUTE_CLOCK_RATE)
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let cycles = rate.checked_mul(p.queue_delay_ms).ok_or("cycle overflow")?;
        let guard = std::mem::size_of::<u64>() * 4;
        let bytes = add(input.len(), guard)?;
        let tensor_bytes = mul(bytes, 2)?;
        let before = cudarc::driver::result::mem_get_info()
            .map_err(|e| e.to_string())?
            .0;
        if before < add(tensor_bytes, p.safety_margin_bytes)? {
            return Err("actual copy tensors plus reserve unavailable".into());
        }
        let owners = Owners::allocate(dev.clone(), bytes)?;
        let floor = cudarc::driver::result::mem_get_info()
            .map_err(|e| e.to_string())?
            .0;
        if floor < p.safety_margin_bytes {
            return Err("copy allocations breached reserve".into());
        }
        let enqueue_delay = || -> Result<(), String> {
            unsafe {
                delay.clone().launch_on_stream(
                    owners.stream.as_ref().ok_or("copy stream owner absent")?,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (cycles,),
                )
            }
            .map_err(|e| format!("queue liveness control: {e}"))
        };
        let mut expected = input.clone();
        expected.resize(bytes, 0xa5);
        owners.fill_pinned(&input, guard)?;
        // Baseline returns the injected event-record branch error with owners retained outside it.
        let baseline_error = "injected event record callback error";
        let baseline_result: Result<(), String> = (|| {
            enqueue_delay()?;
            owners.enqueue(0, "h2d", bytes)?;
            Err(baseline_error.into())
        })();
        if baseline_result.as_ref().map(|_| ()).map_err(String::as_str) != Err(baseline_error) {
            return Err("baseline error changed".into());
        }
        let baseline_query = owners.query();
        if baseline_query != sys::CUresult::CUDA_ERROR_NOT_READY {
            return Err(format!(
                "baseline queue already complete:{baseline_query:?}"
            ));
        }
        checked(
            unsafe { sys::lib().cuStreamSynchronize(owners.stream_handle()) },
            "safe baseline retention drain",
        )?;
        if owners.bytes(0, "h2d")? != expected {
            return Err("baseline payload differs".into());
        }
        let mut controls = Vec::new();
        for direction in ["h2d", "d2h"] {
            for mode in ["success", "callback-error", "panic"] {
                owners.fill_pinned(&input, guard)?;
                if direction == "d2h" {
                    for p in &owners.pinned {
                        let pin = p.as_ref().ok_or("pin")?;
                        unsafe {
                            std::ptr::write_bytes(pin.ptr, 0xa5, pin.bytes);
                        }
                    }
                }
                // Initialize both complete device slots before each measured control, retaining owner windows.
                if direction == "d2h" {
                    for i in 0..2 {
                        let d = owners.device_slots[i].as_ref().ok_or("slot")?;
                        checked(
                            unsafe {
                                sys::lib().cuMemcpyHtoD_v2(
                                    *d.device_ptr(),
                                    expected.as_ptr().cast(),
                                    bytes,
                                )
                            },
                            "D2H exact setup",
                        )?;
                    }
                }
                let mut queued_query = None;
                let invoke = || unsafe {
                    crate::completion::with_copy_completion(
                        &dev,
                        owners.stream_handle(),
                        &owners,
                        |o| {
                            enqueue_delay()?;
                            for i in 0..2 {
                                o.enqueue(i, direction, bytes)?;
                            }
                            let q = o.query();
                            queued_query = Some(q);
                            if q != sys::CUresult::CUDA_ERROR_NOT_READY {
                                return Err(format!("control queue already complete:{q:?}"));
                            }
                            match mode {
                                "callback-error" => Err(baseline_error.into()),
                                "panic" => panic!("injected retained copy callback panic"),
                                _ => Ok(()),
                            }
                        },
                    )
                };
                let outcome = catch_unwind(AssertUnwindSafe(invoke));
                match (mode, &outcome) {
                    ("success", Ok(Ok(()))) => {}
                    ("callback-error", Ok(Err(e))) if e == baseline_error => {}
                    ("panic", Err(_)) => {}
                    _ => {
                        return Err(format!(
                            "callback result suppressed/changed:{mode} {outcome:?}"
                        ))
                    }
                };
                let after = owners.query();
                if after != sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("returned before copy completion:{after:?}"));
                }
                let mut receipts = Vec::new();
                for i in 0..2 {
                    let raw = owners.bytes(i, direction)?;
                    let path = out.join(format!("{direction}-{mode}-slot{i}.bin"));
                    fs::write(&path, &raw).map_err(|e| e.to_string())?;
                    let identity = Identity {
                        path,
                        bytes: raw.len(),
                        sha256: format!("{:x}", Sha256::digest(&raw)),
                    };
                    receipts.push(identity);
                    if raw != expected {
                        return Err("complete copy payload/guard mismatch; raw retained".into());
                    }
                }
                controls.push(json!({"direction":direction,"mode":mode,"queued_query":format!("{:?}",queued_query),"query_after":format!("{after:?}"),"owners_retained_through_completion":true,"callback_result_preserved":true,"complete_payload_and_guards_exact":true,"raw":receipts}));
            }
        }
        let mut called = false;
        let null = unsafe {
            crate::completion::with_copy_completion(&dev, std::ptr::null_mut(), &owners, |_| {
                called = true;
                Ok(())
            })
        };
        if null.is_ok() || called {
            return Err("null stream admission wrote callback".into());
        }
        foreign_context_control(&dev, &owners)?;
        crate::completion::error_contract_controls()?;
        p.input.read()?;
        p.source.read()?;
        p.ptx.read()?;
        p.build_report.read()?;
        Ok(
            json!({"passed":true,"plan":p,"baseline_injected_return_queue":format!("{baseline_query:?}"),"baseline_owner_retention_drained_safely":true,"controls":controls,"null_stream_rejected_before_callback":true,"tensor_bytes":tensor_bytes,"pinned_bytes":tensor_bytes,"actual_free_after_allocation":floor,"safety_margin_bytes":p.safety_margin_bytes,"device_clock_khz":rate,"queue_control_cycles":cycles,"actual_Store_methods_integrated":false,"foreign_context_control_completed":true,"completion_error_join_control_passed":true,"injected_actual_CUDA_completion_failure":false,"model_or_performance_qualified":false,"adopted":false}),
        )
    }
    fn build_copy(plan: &Path, out: &Path) -> Result<Value, String> {
        let p: BuildPlan = serde_json::from_slice(&fs::read(plan).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        p.source.read()?;
        p.nvcc.read()?;
        if p.flags
            != [
                "-ptx",
                "-allow-unsupported-compiler",
                "-arch=sm_80",
                "-O3",
                "--use_fast_math",
            ]
        {
            return Err("copy compiler flags differ".into());
        }
        let z = Command::new(&p.nvcc.path)
            .args(&p.flags)
            .arg("-o")
            .arg(out.join("operator.ptx"))
            .arg(&p.source.path)
            .output()
            .map_err(|e| e.to_string())?;
        fs::write(out.join("compiler.stdout"), z.stdout).map_err(|e| e.to_string())?;
        fs::write(out.join("compiler.stderr"), z.stderr).map_err(|e| e.to_string())?;
        if !z.status.success() {
            return Err(format!("copy compiler failed:{}", z.status));
        }
        let path = out.join("operator.ptx");
        let raw = fs::read(&path).map_err(|e| e.to_string())?;
        Ok(
            json!({"plan":p,"ptx":Identity{path,bytes:raw.len(),sha256:format!("{:x}",Sha256::digest(&raw))},"adopted":false}),
        )
    }
    pub fn entry() -> Result<(), String> {
        if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
            return Err("Use ./dev sequence-copy-probe".into());
        }
        let a = std::env::args().skip(1).collect::<Vec<_>>();
        if a.len() != 3 {
            return Err("build|probe frozen-plan fresh-output".into());
        }
        let out = PathBuf::from(&a[2]);
        fs::create_dir(&out).map_err(|e| e.to_string())?;
        let result = match a[0].as_str() {
            "build" => build_copy(Path::new(&a[1]), &out),
            "probe" => qualify(Path::new(&a[1]), &out),
            _ => Err("invalid mode".into()),
        };
        match result {
            Ok(r) => {
                fs::write(
                    out.join("REPORT.json"),
                    serde_json::to_vec_pretty(&r).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                println!("Evidence saved {}", out.display());
                Ok(())
            }
            Err(e) => {
                fs::write(out.join("FAILURE.txt"), &e).map_err(|x| x.to_string())?;
                Err(e)
            }
        }
    }
}
fn main() {
    if let Err(e) = operator::entry() {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
