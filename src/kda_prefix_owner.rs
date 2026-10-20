//! Private all-F32 per-key KDA owner prototype. No serving dispatch is admitted.
include!("kda_typed_schema.rs");
use cudarc::driver::CudaFunction;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, ffi::c_void};
include!("kda_native_memory.rs");

const TRANSPORT: &[&str] = &[
    "kda_prepared_logits_guard",
    "kda_prefix_validate",
    "kda_prepared_validate",
    "kda_prefix_prepend",
    "kda_prepared_transport_import",
    "kda_prefix_raw_finalize",
    "kda_cursor_validate",
    "kda_cursor_stage_raw",
    "kda_cursor_commit_raw",
];
const MATH: &[&str] = &[
    "kda_dual_decay_tables",
    "kda_dual_coeff_ieee",
    "kda_dual_inverse_ieee",
    "kda_dual_carry_ieee",
    "kda_batch_emit_copy",
    "kda_cursor_emit_copy",
    "kda_cursor_stage_count",
    "kda_commit_validate",
    "kda_emit_bf16",
    "kda_cursor_commit_meta",
];
enum Parameter {
    Ptr(u64),
    Int(i32),
    Float(f32),
}
impl Parameter {
    fn address(&mut self) -> *mut c_void {
        match self {
            Self::Ptr(x) => (x as *mut u64).cast(),
            Self::Int(x) => (x as *mut i32).cast(),
            Self::Float(x) => (x as *mut f32).cast(),
        }
    }
}
fn pointer<T: DeviceAddress>(x: &T) -> u64 {
    x.address()
}
fn typed_pointers<I: PreparedInput>(x: &I) -> Vec<Parameter> {
    x.addresses().into_iter().map(Parameter::Ptr).collect()
}
fn invoke(
    device: &Arc<CudaDevice>,
    stream: sys::CUstream,
    f: &CudaFunction,
    grid: (u32, u32, u32),
    threads: u32,
    shared: u32,
    args: &mut [Parameter],
) -> Result<(), String> {
    let mut pointers = args.iter_mut().map(Parameter::address).collect::<Vec<_>>();
    check(
        unsafe {
            sys::lib().cuLaunchKernel(
                f.cu_function(),
                grid.0,
                grid.1,
                grid.2,
                threads,
                1,
                1,
                shared,
                stream,
                pointers.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        },
        "persistentGDN kernel",
    )
}
struct PreparedLayout {
    raw_schema: Schema,
    pending_schema: Schema,
    hd: usize,
    padded: usize,
    cells: usize,
    groups: usize,
    layouts: [(&'static str, usize); 19],
    power_layouts: [(&'static str, usize); 2],
    allocation_bytes: usize,
    persistent_bytes: usize,
}
impl PreparedLayout {
    fn checked(heads: usize, dim: usize, max_new_rows: usize) -> Result<Self, String> {
        let count = max_new_rows.checked_add(15).ok_or("row extent overflow")?;
        let padded = count.checked_add(15).ok_or("padded overflow")? / 16 * 16;
        let raw_schema = Schema::checked(
            RawContract::KdaPreparedF32VectorGate,
            padded,
            heads,
            dim,
            dim,
        )?;
        let pending_schema =
            Schema::checked(RawContract::KdaPreparedF32VectorGate, 16, heads, dim, dim)?;
        if max_new_rows == 0 {
            return Err("zero batch capacity".into());
        }
        let hd = heads.checked_mul(dim).ok_or("HD overflow")?;
        let st = hd.checked_mul(dim).ok_or("state overflow")?;
        let cells = padded.checked_mul(hd).ok_or("cells overflow")?;
        let groups = padded / 16;
        let mats = groups
            .checked_mul(heads)
            .and_then(|x| x.checked_mul(256))
            .ok_or("matrix overflow")?;
        let pair_cells = mats
            .checked_mul(dim)
            .ok_or("vector factor extent overflow")?;
        let layouts = [
            ("q", cells),
            ("k", cells),
            ("v", cells),
            ("g", cells),
            ("beta", padded * heads),
            ("inc_hi", cells),
            ("inc_lo", cells),
            ("pair_hi", pair_cells),
            ("pair_lo", pair_cells),
            ("a", mats),
            ("qk", mats),
            ("inverse", mats),
            ("out", cells),
            ("emitted", max_new_rows * hd),
            ("incoming", st),
            ("complete", st),
            ("latest", st),
            ("final", st),
            ("completed", st),
        ];
        for &(_, n) in &layouts {
            if n == 0 || n > i32::MAX as usize {
                return Err("F32 bounded signedindex ABI".into());
            }
        }
        let float_bytes = layouts
            .iter()
            .try_fold(0usize, |n, (_, v)| {
                v.checked_mul(4).and_then(|x| n.checked_add(x))
            })
            .ok_or("F32 allocation overflow")?;
        let power_layouts = [("inc_power", cells), ("pair_power", pair_cells)];
        let power_bytes = power_layouts
            .iter()
            .try_fold(0usize, |n, (_, v)| {
                v.checked_mul(4).and_then(|x| n.checked_add(x))
            })
            .ok_or("I32 factor allocation overflow")?;
        let allocation_bytes = raw_schema
            .requested_bytes
            .checked_add(pending_schema.requested_bytes)
            .and_then(|x| x.checked_add(float_bytes))
            .and_then(|x| x.checked_add(power_bytes))
            .and_then(|x| x.checked_add(5 * 4 + 8))
            .and_then(|x| {
                max_new_rows
                    .checked_mul(hd)
                    .and_then(|n| n.checked_mul(2))
                    .and_then(|n| x.checked_add(n))
            })
            .ok_or("owner allocation overflow")?;
        let persistent_bytes = pending_schema
            .requested_bytes
            .checked_add(3 * st * 4 + 4 * 2 + 8)
            .ok_or("owner persistent overflow")?;
        Ok(Self {
            raw_schema,
            pending_schema,
            hd,
            padded,
            cells,
            groups,
            layouts,
            power_layouts,
            allocation_bytes,
            persistent_bytes,
        })
    }
}

struct PreparedArena {
    device: Arc<CudaDevice>,
    heads: usize,
    dim: usize,
    max_new_rows: usize,
    padded: usize,
    functions: BTreeMap<&'static str, CudaFunction>,
    raw: NativeRawOwner,
    arrays: BTreeMap<&'static str, GpuBuf<f32>>,
    powers: BTreeMap<&'static str, GpuBuf<i32>>,
    bf16_emitted: GpuBuf<u16>,
    count: GpuBuf<i32>,
    raw_count: GpuBuf<i32>,
    one: GpuBuf<i32>,
    arithmetic_sha256: String,
    requested_bytes: usize,
    enqueue: std::sync::Mutex<()>,
    native_admissions: Vec<NativePointerAdmission>,
    stream: Arc<NativeStream>,
}
struct EnqueueLease<'a> {
    arena: &'a PreparedArena,
    _guard: std::sync::MutexGuard<'a, ()>,
}
impl PreparedArena {
    fn lease(&self) -> Result<EnqueueLease<'_>, String> {
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        let guard = self
            .enqueue
            .try_lock()
            .map_err(|_| "arena enqueue already leased/poisoned")?;
        Ok(EnqueueLease {
            arena: self,
            _guard: guard,
        })
    }
}
struct PreparedPersistentState {
    timeline: std::sync::Mutex<Option<StateTimeline>>,
    pending_raw: NativeRawOwner,
    arrays: BTreeMap<&'static str, GpuBuf<f32>>,
    pending: GpuBuf<i32>,
    error: GpuBuf<i32>,
    absolute: GpuBuf<u64>,
    native_admissions: Vec<NativePointerAdmission>,
}
struct PreparedOwner {
    device: Arc<CudaDevice>,
    heads: usize,
    dim: usize,
    max_new_rows: usize,
    padded: usize,
    arena: Arc<PreparedArena>,
    state: Arc<PreparedPersistentState>,
    pub arithmetic_sha256: String,
    pub persistent_bytes: usize,
    pub allocation_bytes: usize,
    stream: Arc<NativeStream>,
}
impl std::ops::Deref for PreparedOwner {
    type Target = PreparedPersistentState;
    fn deref(&self) -> &Self::Target {
        &self.state
    }
}
impl Drop for PreparedOwner {
    fn drop(&mut self) {
        if let Err(e) = self.stream.drain() {
            crate::vram_monitor::fatal_cuda_context_error("prepared state owner final completion", &e);
        }
    }
}
impl PreparedOwner {
    fn new(
        device: Arc<CudaDevice>,
        heads: usize,
        dim: usize,
        max_new_rows: usize,
        reserve: usize,
        transport: &str,
        math: &str,
    ) -> Result<Self, String> {
        let stream = NativeStream::borrowed_device(device.clone())?;
        Self::new_on_stream(
            device,
            stream,
            heads,
            dim,
            max_new_rows,
            reserve,
            transport,
            math,
        )
    }
    fn new_on_stream(
        device: Arc<CudaDevice>,
        stream: Arc<NativeStream>,
        heads: usize,
        dim: usize,
        max_new_rows: usize,
        reserve: usize,
        transport: &str,
        math: &str,
    ) -> Result<Self, String> {
        stream.validate()?;
        if !Arc::ptr_eq(&device, &stream.device) {
            return Err("native owner foreigndevice stream".into());
        }
        let PreparedLayout {
            raw_schema,
            pending_schema,
            hd,
            padded,
            cells,
            groups,
            layouts,
            power_layouts,
            allocation_bytes,
            persistent_bytes,
        } = PreparedLayout::checked(heads, dim, max_new_rows)?;
        device.bind_to_thread().map_err(|e| e.to_string())?;
        let before = memory()?;
        if before[0]
            < allocation_bytes
                .checked_add(reserve)
                .ok_or("owner reserve overflow")?
        {
            return Err("actual measured owner admission".into());
        }
        let mut functions = BTreeMap::new();
        for (s, names) in [(transport, TRANSPORT), (math, MATH)] {
            let identity = format!(
                "prepared-kda-vector-fixed16-{:x}",
                Sha256::digest(s.as_bytes())
            );
            let existing = names
                .iter()
                .filter(|&&n| device.get_func(&identity, n).is_some())
                .count();
            if existing != 0 && existing != names.len() {
                return Err("partial owner module identity".into());
            }
            if existing == 0 {
                device
                    .load_ptx(cudarc::nvrtc::Ptx::from_src(s), &identity, names)
                    .map_err(|e| e.to_string())?;
            }
            for &name in names {
                let f = device
                    .get_func(&identity, name)
                    .ok_or("required owner symbol")?;
                let mut threads = 0;
                check(
                    unsafe {
                        sys::lib().cuFuncGetAttribute(
                            &mut threads,
                            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                            f.cu_function(),
                        )
                    },
                    "actual owner kernel resource",
                )?;
                if threads < 256 {
                    return Err("actual owner thread support".into());
                }
                functions.insert(name, f);
            }
        }
        let max_z = device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
            .map_err(|e| e.to_string())? as usize;
        let max_x = device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
            .map_err(|e| e.to_string())? as usize;
        let max_y = device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
            .map_err(|e| e.to_string())? as usize;
        if cells.div_ceil(256) > max_x
            || heads > max_x
            || groups > max_y
            || dim > max_z
            || dim.div_ceil(16) > max_y
        {
            return Err("actual owner grid limit".into());
        }
        if memory()?[0]
            < allocation_bytes
                .checked_add(reserve)
                .ok_or("post-module reserve overflow")?
        {
            return Err("actual postmodule measured owner admission".into());
        }
        let pending_raw =
            NativeRawOwner::allocate(stream.clone(), pending_schema.clone(), reserve)?;
        let mut persistent_arrays = BTreeMap::new();
        for (name, n) in layouts
            .iter()
            .copied()
            .filter(|(name, _)| matches!(*name, "incoming" | "complete" | "latest"))
        {
            persistent_arrays.insert(name, native_zeroed::<f32>(&stream, n)?);
        }
        let pending = native_zeroed::<i32>(&stream, 1)?;
        let error = native_zeroed::<i32>(&stream, 1)?;
        let absolute = native_zeroed::<u64>(&stream, 1)?;
        let arithmetic_sha256=format!("{:x}",Sha256::digest(serde_json::to_vec(&serde_json::json!({"ownership":"explicit-owned-stream-zero-v1","contract":"vector-KDA/16-local-F64-perkey-gates/dual-hi-lo-i32/F32-decayed-state-pair/IEEE-FMA-triangular-v1/transactional-count","output_units":"native-prepared-scaledQ/F32-scale1/BF16-RNE-finitecommit","rawschema":pending_raw.schema,"transport_ptx":format!("{:x}",Sha256::digest(transport.as_bytes())),"math_ptx":format!("{:x}",Sha256::digest(math.as_bytes())),"math_source":format!("{:x}",Sha256::digest(include_bytes!("cuda/kda_owner_math.cu")))})).map_err(|e|e.to_string())?));
        let mut native_admissions = Vec::new();
        for x in persistent_arrays.values() {
            native_admissions.push(native_pointer_admission(&stream, pointer(x), x.len() * 4)?);
        }
        for x in [&pending, &error] {
            native_admissions.push(native_pointer_admission(&stream, pointer(x), 4)?);
        }
        native_admissions.push(native_pointer_admission(&stream, pointer(&absolute), 8)?);
        let arena = PreparedArena::allocate_storage(
            device.clone(),
            stream.clone(),
            heads,
            dim,
            max_new_rows,
            reserve,
            functions,
            arithmetic_sha256.clone(),
        )?;
        let result = Self {
            device,
            heads,
            dim,
            max_new_rows,
            padded,
            arena,
            state: Arc::new(PreparedPersistentState {
                timeline: std::sync::Mutex::new(None),
                pending_raw,
                arrays: persistent_arrays,
                pending,
                error,
                absolute,
                native_admissions,
            }),
            arithmetic_sha256,
            persistent_bytes,
            allocation_bytes,
            stream,
        };
        result.reset()?;
        result.stream.drain()?;
        if memory()?[0] < reserve {
            return Err("observed owner allocation floor".into());
        }
        Ok(result)
    }
    fn array(&self, n: &str) -> &GpuBuf<f32> {
        self.arrays
            .get(n)
            .or_else(|| self.arena.arrays.get(n))
            .expect("internal fixed arena role")
    }
    fn validate_lease(&self, lease: &EnqueueLease<'_>) -> Result<(), String> {
        if !std::ptr::eq(lease.arena, Arc::as_ptr(&self.arena))
            || !Arc::ptr_eq(&self.device, &lease.arena.device)
            || self.heads != lease.arena.heads
            || self.dim != lease.arena.dim
            || self.max_new_rows != lease.arena.max_new_rows
            || self.padded != lease.arena.padded
            || self.arithmetic_sha256 != lease.arena.arithmetic_sha256
            || self.stream.handle != lease.arena.stream.handle
        {
            return Err("foreign arena/context/stream/geometry/arithmetic lease".into());
        }
        Schema::checked(
            RawContract::KdaPreparedF32VectorGate,
            self.padded,
            self.heads,
            self.dim,
            self.dim,
        )?
        .admits(&lease.arena.raw.schema)?;
        self.device.bind_to_thread().map_err(|e| e.to_string())
    }
    fn from_shared(source: &Self, reserve: usize) -> Result<Self, String> {
        let stream = source.stream.clone();
        let lease = source.arena.lease()?;
        source.validate_lease(&lease)?;
        let device = source.device.clone();
        let st = source
            .heads
            .checked_mul(source.dim)
            .and_then(|x| x.checked_mul(source.dim))
            .ok_or("shared state extent")?;
        let schema = Schema::checked(
            RawContract::KdaPreparedF32VectorGate,
            16,
            source.heads,
            source.dim,
            source.dim,
        )?;
        let persistent_bytes = schema
            .requested_bytes
            .checked_add(st.checked_mul(12).ok_or("shared carry bytes")?)
            .and_then(|x| x.checked_add(16))
            .ok_or("shared persistent bytes")?;
        if persistent_bytes != source.persistent_bytes {
            return Err("shared persistent schema accounting".into());
        }
        if memory()?[0]
            < persistent_bytes
                .checked_add(reserve)
                .ok_or("shared reserve extent")?
        {
            return Err("actual measured shared owner admission".into());
        }
        let pending_raw = NativeRawOwner::allocate(source.stream.clone(), schema, reserve)?;
        let mut arrays = BTreeMap::new();
        for role in ["incoming", "complete", "latest"] {
            arrays.insert(role, native_zeroed::<f32>(&stream, st)?);
        }
        let pending = native_zeroed::<i32>(&stream, 1)?;
        let error = native_zeroed::<i32>(&stream, 1)?;
        let absolute = native_zeroed::<u64>(&stream, 1)?;
        let mut native_admissions = Vec::new();
        for x in arrays.values() {
            native_admissions.push(native_pointer_admission(
                &source.stream,
                pointer(x),
                x.len() * 4,
            )?);
        }
        for x in [&pending, &error] {
            native_admissions.push(native_pointer_admission(&source.stream, pointer(x), 4)?);
        }
        native_admissions.push(native_pointer_admission(
            &source.stream,
            pointer(&absolute),
            8,
        )?);
        let result = Self {
            device,
            heads: source.heads,
            dim: source.dim,
            max_new_rows: source.max_new_rows,
            padded: source.padded,
            arena: source.arena.clone(),
            state: Arc::new(PreparedPersistentState {
                timeline: std::sync::Mutex::new(None),
                pending_raw,
                arrays,
                pending,
                error,
                absolute,
                native_admissions,
            }),
            arithmetic_sha256: source.arithmetic_sha256.clone(),
            persistent_bytes,
            allocation_bytes: persistent_bytes,
            stream: source.stream.clone(),
        };
        result.reset()?;
        result.stream.drain()?;
        if memory()?[0] < reserve {
            return Err("observed shared owner allocation floor".into());
        }
        Ok(result)
    }
    fn bind_persistent(arena_owner: &Self, state_owner: &Self) -> Result<Self, String> {
        arena_owner.stream.validate()?;
        state_owner.stream.validate()?;
        if !Arc::ptr_eq(&arena_owner.device, &state_owner.device)
            || arena_owner.stream.handle != state_owner.stream.handle
            || arena_owner.heads != state_owner.heads
            || arena_owner.dim != state_owner.dim
            || arena_owner.arithmetic_sha256 != state_owner.arithmetic_sha256
            || arena_owner.persistent_bytes != state_owner.persistent_bytes
        {
            return Err("persistent binding requires exact geometry/context/orderedstream/arithmetic/layout".into());
        }
        // Views must be retained by the phase/graph owner; no reset, host copy or per-step allocation.
        Ok(Self {
            device: arena_owner.device.clone(),
            heads: arena_owner.heads,
            dim: arena_owner.dim,
            max_new_rows: arena_owner.max_new_rows,
            padded: arena_owner.padded,
            arena: arena_owner.arena.clone(),
            state: state_owner.state.clone(),
            arithmetic_sha256: arena_owner.arithmetic_sha256.clone(),
            persistent_bytes: state_owner.persistent_bytes,
            allocation_bytes: 0,
            stream: arena_owner.stream.clone(),
        })
    }
    fn launch_graph_in(
        &self,
        lease: &EnqueueLease<'_>,
        graph: &OwnerGraph<'_>,
    ) -> Result<(), String> {
        self.validate_lease(lease)?;
        if !std::ptr::eq(self, graph.owner) || graph.exec.is_null() {
            return Err("foreign owner/uncaptured graph".into());
        }
        check(
            unsafe { sys::lib().cuGraphLaunch(graph.exec, self.stream.handle) },
            "shared owner leased graphlaunch",
        )
    }
    fn p(&self, n: &str) -> u64 {
        if let Some(x) = self.arena.powers.get(n) {
            pointer(x)
        } else {
            pointer(self.array(n))
        }
    }
    fn kernel(
        &self,
        n: &str,
        grid: (u32, u32, u32),
        shared: u32,
        args: &mut [Parameter],
    ) -> Result<(), String> {
        invoke(
            &self.device,
            self.stream.handle,
            self.arena
                .functions
                .get(n)
                .ok_or_else(|| format!("missing required vector kernel {n}"))?,
            grid,
            256,
            shared,
            args,
        )
    }
    fn linear(&self, n: &str, count: usize, args: &mut [Parameter]) -> Result<(), String> {
        self.kernel(n, (count.div_ceil(256) as u32, 1, 1), 0, args)
    }
    fn reset(&self) -> Result<(), String> {
        self.reset_impl(None)
    }
    fn reset_in_phase(&self, phase: &StatePhase<'_>) -> Result<(), String> {
        if !std::ptr::eq(self, phase.owner) || phase.stream.handle != self.stream.handle {
            return Err("reset requires actual admitted owner phase".into());
        }
        self.reset_impl(Some(phase))
    }
    fn reset_impl(&self, phase: Option<&StatePhase<'_>>) -> Result<(), String> {
        self.stream.validate()?;
        if let Some(phase) = phase {
            self.pending_raw.validate_phase_invalidation(phase)?;
        } else if self.pending_raw.stream.handle != self.stream.handle {
            return Err("cross-stream shared reset requires ordered phase".into());
        }
        for name in ["incoming", "complete", "latest"] {
            let x = &self.arrays[name];
            check(
                unsafe { sys::lib().cuMemsetD32Async(pointer(x), 0, x.len(), self.stream.handle) },
                "owner canonical carryreset",
            )?;
        }
        match phase {
            Some(phase) => self.pending_raw.invalidate_in_phase(phase)?,
            None => unsafe {
                self.pending_raw.invalidate(self.stream.handle)?;
            },
        }
        for x in [&self.pending, &self.error] {
            check(
                unsafe { sys::lib().cuMemsetD32Async(pointer(x), 0, 1, self.stream.handle) },
                "owner canonical scalarreset",
            )?;
        }
        check(
            unsafe {
                sys::lib().cuMemsetD8Async(pointer(&self.absolute), 0, 8, self.stream.handle)
            },
            "owner canonical absolutereset",
        )
    }
    fn validate_view<I: PreparedInput>(&self, live: &I, rows: usize) -> Result<(), String> {
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        let want = Schema::checked(
            RawContract::KdaPreparedF32VectorGate,
            rows,
            self.heads,
            self.dim,
            self.dim,
        )?;
        want.admits(live.schema())?;
        if !Arc::ptr_eq(&self.device, live.device()) || live.stream() != self.stream.handle {
            return Err("foreign prepared owner context".into());
        }
        Ok(())
    }
    fn domain<I: PreparedInput>(&self, raw: &I, count: u64, capacity: usize) -> Result<(), String> {
        let mut a = typed_pointers(raw);
        a.extend([
            Parameter::Ptr(count),
            Parameter::Int(capacity as i32),
            Parameter::Int(self.heads as i32),
            Parameter::Int(self.dim as i32),
            Parameter::Ptr(pointer(&self.error)),
        ]);
        self.linear(
            "kda_prepared_validate",
            raw.schema().planes[0].elements,
            &mut a,
        )
    }
    fn import<I: PreparedInput>(&self, raw: &I, count: u64, padded: usize) -> Result<(), String> {
        let mut a = typed_pointers(raw);
        a.extend(
            ["q", "k", "v", "g", "beta"]
                .iter()
                .map(|n| Parameter::Ptr(self.p(n))),
        );
        a.extend([
            Parameter::Ptr(count),
            Parameter::Int(padded as i32),
            Parameter::Int(self.heads as i32),
            Parameter::Int(self.dim as i32),
            Parameter::Ptr(pointer(&self.error)),
        ]);
        self.linear(
            "kda_prepared_transport_import",
            padded * self.heads * self.dim,
            &mut a,
        )
    }
    fn math(&self, count: u64, groups: usize) -> Result<(), String> {
        use Parameter::{Int as I, Ptr as P};
        let h = self.heads as i32;
        let d = self.dim as i32;
        let error = pointer(&self.error);
        let grid = (self.heads as u32, groups as u32, 1);
        self.kernel(
            "kda_dual_decay_tables",
            (grid.0, grid.1, self.dim as u32),
            0,
            &mut [
                P(self.p("g")),
                P(self.p("inc_hi")),
                P(self.p("inc_lo")),
                P(self.p("inc_power")),
                P(self.p("pair_hi")),
                P(self.p("pair_lo")),
                P(self.p("pair_power")),
                I(h),
                I(d),
                P(count),
                P(error),
            ],
        )?;
        self.kernel(
            "kda_dual_coeff_ieee",
            grid,
            0,
            &mut [
                P(self.p("q")),
                P(self.p("k")),
                P(self.p("beta")),
                P(self.p("pair_hi")),
                P(self.p("pair_lo")),
                P(self.p("pair_power")),
                P(self.p("a")),
                P(self.p("qk")),
                I(h),
                I(d),
                P(count),
                P(error),
            ],
        )?;
        self.kernel(
            "kda_dual_inverse_ieee",
            grid,
            0,
            &mut [P(self.p("a")), P(self.p("inverse")), P(count), P(error)],
        )?;
        self.kernel(
            "kda_dual_carry_ieee",
            (self.heads as u32, self.dim.div_ceil(16) as u32, 1),
            0,
            &mut [
                P(self.p("v")),
                P(self.p("beta")),
                P(self.p("k")),
                P(self.p("q")),
                P(self.p("inc_hi")),
                P(self.p("inc_lo")),
                P(self.p("inc_power")),
                P(self.p("pair_hi")),
                P(self.p("pair_lo")),
                P(self.p("pair_power")),
                P(self.p("qk")),
                P(self.p("inverse")),
                P(self.p("incoming")),
                P(self.p("out")),
                P(self.p("final")),
                P(self.p("final")),
                P(self.p("completed")),
                I(h),
                I(d),
                P(count),
                I(0),
                P(error),
            ],
        )
    }
    fn prefill<I: PreparedInput>(&self, live: &I, start: u64, rows: usize) -> Result<(), String> {
        let lease = self.arena.lease()?;
        self.prefill_in(&lease, live, start, rows)
    }
    fn prefill_in<I: PreparedInput>(
        &self,
        lease: &EnqueueLease<'_>,
        live: &I,
        start: u64,
        rows: usize,
    ) -> Result<(), String> {
        self.validate_lease(lease)?;
        use Parameter::{Int as I, Ptr as P};
        if rows == 0 || rows > self.max_new_rows {
            return Err("owner prefill capacity".into());
        }
        start.checked_add(rows as u64).ok_or("owner endoverflow")?;
        self.validate_view(live, rows)?;
        let partial = (start % 16) as usize;
        let total = partial + rows;
        if total > self.padded {
            return Err("owner workspace extent".into());
        }
        let m = rows as i32;
        check(
            unsafe {
                sys::lib().cuMemsetD32Async(
                    pointer(&self.arena.count),
                    total as u32,
                    1,
                    self.stream.handle,
                )
            },
            "owner immutable batchcount",
        )?;
        self.set_raw_count(m)?;
        self.kernel(
            "kda_prefix_validate",
            (1, 1, 1),
            0,
            &mut [
                P(pointer(&self.pending)),
                P(pointer(&self.error)),
                P(pointer(&self.absolute)),
                P(start),
                I(m),
            ],
        )?;
        self.domain(&self.pending_raw, pointer(&self.pending), 16)?;
        self.domain(live, self.raw_count_ptr(), rows)?;
        let mut a = typed_pointers(live);
        a.extend(typed_pointers(&self.pending_raw));
        a.extend(typed_pointers(&self.arena.raw));
        a.extend([
            P(pointer(&self.pending)),
            P(pointer(&self.error)),
            P(pointer(&self.absolute)),
            P(start),
            I(m),
            I(self.heads as i32),
            I(self.dim as i32),
        ]);
        self.linear("kda_prefix_prepend", total * self.heads * self.dim, &mut a)?;
        self.import(
            &self.arena.raw,
            pointer(&self.arena.count),
            total.div_ceil(16) * 16,
        )?;
        self.math(pointer(&self.arena.count), total.div_ceil(16))?;
        self.validate_commit(pointer(&self.arena.count), partial, rows)?;
        self.emit_bf16(pointer(&self.arena.count), partial, rows)?;
        self.linear(
            "kda_batch_emit_copy",
            (rows * self.heads * self.dim).max(self.heads * self.dim * self.dim),
            &mut [
                P(self.p("out")),
                P(self.p("final")),
                P(self.p("completed")),
                P(self.p("emitted")),
                P(self.p("incoming")),
                P(self.p("complete")),
                P(self.p("latest")),
                P(pointer(&self.error)),
                I(partial as i32),
                I(m),
                I(self.heads as i32),
                I(self.dim as i32),
            ],
        )?;
        let mut a = typed_pointers(&self.arena.raw);
        a.extend(typed_pointers(&self.pending_raw));
        a.extend([
            P(pointer(&self.pending)),
            P(pointer(&self.error)),
            P(pointer(&self.absolute)),
            I(partial as i32),
            I(m),
            I(self.heads as i32),
            I(self.dim as i32),
        ]);
        self.linear(
            "kda_prefix_raw_finalize",
            16 * self.heads * self.dim,
            &mut a,
        )
    }
    fn decode_capture<I: PreparedInput>(&self, live: &I) -> Result<(), String> {
        let lease = self.arena.lease()?;
        self.decode_capture_in(&lease, live)
    }
    fn decode_capture_in<I: PreparedInput>(
        &self,
        lease: &EnqueueLease<'_>,
        live: &I,
    ) -> Result<(), String> {
        self.validate_lease(lease)?;
        use Parameter::{Int as I, Ptr as P};
        self.validate_view(live, 1)?;
        let pending = pointer(&self.pending);
        let error = pointer(&self.error);
        let absolute = pointer(&self.absolute);
        self.kernel(
            "kda_cursor_validate",
            (1, 1, 1),
            0,
            &mut [P(pending), P(error), P(absolute)],
        )?;
        self.domain(&self.pending_raw, pending, 16)?;
        self.domain(live, pointer(&self.arena.one), 1)?;
        self.kernel(
            "kda_cursor_stage_count",
            (1, 1, 1),
            0,
            &mut [
                P(pending),
                P(absolute),
                P(pointer(&self.arena.count)),
                P(error),
            ],
        )?;
        let mut a = typed_pointers(live);
        a.extend(typed_pointers(&self.pending_raw));
        a.extend(typed_pointers(&self.arena.raw));
        a.extend([
            P(pointer(&self.arena.count)),
            P(error),
            I(self.heads as i32),
            I(self.dim as i32),
        ]);
        self.linear("kda_cursor_stage_raw", 16 * self.heads * self.dim, &mut a)?;
        self.import(&self.arena.raw, pointer(&self.arena.count), 16)?;
        self.math(pointer(&self.arena.count), 1)?;
        self.validate_commit(pointer(&self.arena.count), 0, 0)?;
        self.emit_bf16(pointer(&self.arena.count), 0, 0)?;
        self.linear(
            "kda_cursor_emit_copy",
            self.heads * self.dim * self.dim,
            &mut [
                P(self.p("out")),
                P(self.p("final")),
                P(self.p("completed")),
                P(self.p("emitted")),
                P(self.p("incoming")),
                P(self.p("complete")),
                P(self.p("latest")),
                P(pointer(&self.arena.count)),
                P(error),
                I(self.heads as i32),
                I(self.dim as i32),
            ],
        )?;
        let mut a = typed_pointers(&self.arena.raw);
        a.extend(typed_pointers(&self.pending_raw));
        a.extend([
            P(pointer(&self.arena.count)),
            P(error),
            I(self.heads as i32),
            I(self.dim as i32),
        ]);
        self.linear("kda_cursor_commit_raw", 16 * self.heads * self.dim, &mut a)?;
        self.kernel(
            "kda_cursor_commit_meta",
            (1, 1, 1),
            0,
            &mut [
                P(pending),
                P(absolute),
                P(pointer(&self.arena.count)),
                P(error),
            ],
        )
    }
    fn validate_commit(&self, count: u64, partial: usize, rows: usize) -> Result<(), String> {
        use Parameter::{Int as I, Ptr as P};
        self.linear(
            "kda_commit_validate",
            (rows.max(1) * self.heads * self.dim).max(self.heads * self.dim * self.dim),
            &mut [
                P(self.p("out")),
                P(self.p("final")),
                P(self.p("completed")),
                P(count),
                I(partial as i32),
                I(rows as i32),
                I(self.heads as i32),
                I(self.dim as i32),
                P(pointer(&self.error)),
            ],
        )
    }
    fn emit_bf16(&self, count: u64, partial: usize, rows: usize) -> Result<(), String> {
        use Parameter::{Int as I, Ptr as P};
        self.linear(
            "kda_emit_bf16",
            rows.max(1) * self.heads * self.dim,
            &mut [
                P(self.p("out")),
                P(pointer(&self.arena.bf16_emitted)),
                P(count),
                I(partial as i32),
                I(rows as i32),
                I(self.heads as i32),
                I(self.dim as i32),
                P(pointer(&self.error)),
            ],
        )
    }
    fn raw_count_ptr(&self) -> u64 {
        pointer(&self.arena.raw_count)
    }
    fn set_raw_count(&self, value: i32) -> Result<(), String> {
        check(
            unsafe {
                sys::lib().cuMemsetD32Async(
                    pointer(&self.arena.raw_count),
                    value as u32,
                    1,
                    self.stream.handle,
                )
            },
            "owner stable rawcount",
        )
    }
    fn register(
        &self,
        registry: &mut crate::session_cache::SequenceStateRegistry,
        layer: usize,
    ) -> Result<(), String> {
        use crate::session_cache::{SequenceStateAllocation, SequenceStateGrowth};
        let mut add = |role: &str, ptr: u64, dtype: &str, bytes: usize, shape: Vec<usize>| {
            let mut stride = bytes;
            let mut strides = vec![0; shape.len()];
            for i in (0..shape.len()).rev() {
                strides[i] = stride;
                stride = stride
                    .checked_mul(shape[i])
                    .ok_or("registry owner extentoverflow")?;
            }
            registry.register(SequenceStateAllocation {
                name: format!("layer{layer}.kda_fixed16.{role}"),
                kind: format!("kda_fixed16_{role}"),
                layer_idx: Some(layer),
                device_ordinal: self.device.ordinal(),
                ptr,
                storage_bytes: stride,
                dtype: dtype.into(),
                element_size: bytes,
                shape,
                strides_bytes: strides,
                growth: SequenceStateGrowth::Fixed,
            })
        };
        for name in ["incoming", "complete", "latest"] {
            add(
                name,
                self.p(name),
                "float32",
                4,
                vec![self.heads, self.dim, self.dim],
            )?;
        }
        for (slot, (desc, ptr)) in self.pending_raw.inventory().into_iter().enumerate() {
            add(
                &format!("raw{slot}"),
                ptr,
                if desc.dtype == Dtype::Float32 {
                    "float32"
                } else {
                    "bfloat16"
                },
                desc.dtype.bytes(),
                desc.shape,
            )?;
        }
        add("pending", pointer(&self.pending), "int32", 4, vec![1])?;
        add("error", pointer(&self.error), "int32", 4, vec![1])?;
        add("absolute", pointer(&self.absolute), "uint64", 8, vec![1])
    }
}

#[cfg(test)]
include!("kda_owner_tests.rs");

include!("kda_raw_producer.rs");
include!("kda_owned_graph.rs");
include!("kda_model_transaction.rs");
include!("kda_model_binding_proof.rs");
include!("kda_capture_binding.rs");
#[cfg(test)]
include!("kda_capture_binding_tests.rs");
#[cfg(test)]
include!("kda_capture_binding_frozen307.rs");
include!("kda_model_arena_admission.rs");
include!("kda_producer_resource_topology.rs");
include!("kda_producer_resource_domain.rs");
#[cfg(test)]
include!("kda_model_arena_tests.rs");
#[cfg(test)]
include!("kda_transaction_frozen296.rs");
include!("kda_model_source.rs");
include!("kda_prefill_source.rs");
include!("kda_model_family_setup.rs");
include!("kda_shared_model_family_setup.rs");
include!("kda_arena_storage.rs");
include!("kda_workspace_epoch.rs");
include!("kda_source_epoch_rebind.rs");
include!("kda_model_source_transaction.rs");
include!("kda_prefill_source_lifetime.rs");
include!("kda_engine_request_storage.rs");
include!("kda_prefill_execution_phase.rs");
include!("kda_actual_prefill_shadow.rs");
#[cfg(test)]
include!("kda_raw_producer_tests.rs");

#[cfg(test)]
include!("kda_long_chronology_tests.rs");

include!("kda_state_timeline.rs");
#[cfg(test)]
include!("kda_multistream_tests.rs");
#[cfg(test)]
include!("kda_live_tiling_calibration.rs");

#[cfg(test)]
include!("kda_model_family_setup_tests.rs");
#[cfg(test)]
include!("kda_constructor_chronology_tests.rs");
#[cfg(test)]
include!("kda_model_graph_tests.rs");
#[cfg(test)]
include!("kda_quiet_ownership_tests.rs");

#[cfg(test)]
include!("kda_producer_resource_tests.rs");
#[cfg(test)]
include!("kda_resource_graph_tests.rs");
#[cfg(test)]
include!("kda_shared_factory_tests.rs");
#[cfg(test)]
include!("kda_source_epoch_tests.rs");

#[cfg(test)]
include!("kda_workspace_epoch_tests.rs");
#[cfg(test)]
include!("kda_group_workspace_tests.rs");

include!("kda_session_lifecycle.rs");
include!("kda_sequence_epoch_lifecycle.rs");

#[cfg(test)]
include!("kda_session_lifecycle_tests.rs");

#[cfg(test)]
include!("kda_model_source_transaction_tests.rs");
#[cfg(test)]
include!("kda_prefill_source_lifetime_tests.rs");

include!("kda_prepared_logit_guard.rs");
#[cfg(test)]
include!("kda_prepared_logit_guard_tests.rs");
include!("kda_diagnostic_state_admission.rs");

include!("kda_actual_shadow_completion.rs");

include!("kda_owned_producer_resource.rs");

include!("kda_resource_epoch_handoff.rs");

include!("kda_retired_resource_capability.rs");

include!("kda_current_resource_domain.rs");

#[cfg(test)]
include!("kda_current_resource_domain_tests.rs");
#[cfg(test)]
include!("kda_current_sequence_tests.rs");

include!("kda_current_sequence_phase.rs");
include!("kda_snapshot_semantics.rs");
include!("kda_installed_current_sequence.rs");
include!("kda_installed_coupled_reset.rs");
include!("kda_installed_logit_guard.rs");

#[cfg(test)]
include!("kda_original_long_tests.rs");

include!("kda_external_dma_owners.rs");
include!("kda_cold_model_storage.rs");
include!("kda_serving_dispatch.rs");
include!("kda_serving_setup.rs");
include!("kda_serving_guard.rs");
include!("kda_serving_execution.rs");
