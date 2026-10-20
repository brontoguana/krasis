//! One immutable program/native-context bank and externally leased phase scratch.
use crate::{
    scalar_gdn_layout::Layout,
    scalar_gdn_native::Native,
    scalar_gdn_owner::{drain_or_fatal, FUNCTIONS},
    scalar_gdn_phase::{Event, Stream},
};
use cudarc::driver::{sys, CudaDevice, CudaFunction, CudaSlice, DevicePtr};
use sha2::{Digest, Sha256};
use std::{
    cell::Cell,
    collections::BTreeMap,
    marker::PhantomData,
    path::Path,
    sync::{Arc, Mutex, MutexGuard, TryLockError},
};

pub struct Program {
    pub(crate) identity: String,
    pub(crate) functions: BTreeMap<&'static str, CudaFunction>,
    bindings: BTreeMap<[usize; 3], Native>,
    pub(crate) limits: [usize; 3],
    // Last: the actual context outlives every native module/library owner.
    pub(crate) device: Arc<CudaDevice>,
}
impl Program {
    pub fn new(
        device: Arc<CudaDevice>,
        layouts: &[Layout],
        sidecar: &Path,
        ptx: &str,
    ) -> Result<Arc<Self>, String> {
        let first = layouts
            .first()
            .ok_or("scalar program requires actual loaded layer geometry")?;
        device.bind_to_thread().map_err(|e| e.to_string())?;
        if device.cu_stream().is_null() {
            return Err("shared scalar program requires a real retained nondefault stream".into());
        }
        use sys::CUdevice_attribute as A;
        let attribute = |a| -> Result<usize, String> {
            usize::try_from(device.attribute(a).map_err(|e| e.to_string())?)
                .map_err(|_| "negative scalar CUDA limit".into())
        };
        let limits = [
            attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
            attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
            attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
        ];
        if attribute(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)? < 256 {
            return Err("scalar transport threads exceed actual device limit".into());
        }
        // Native context creation is cold and happens ONCE for this bank.
        let base = unsafe { Native::load(sidecar, first.heads, first.key_dim, first.value_dim)? };
        if base.context() != *device.cu_primary_ctx() {
            return Err("scalar program native context differs from real retained device".into());
        }
        let identity = format!(
            "native64-owned-abi2:{}:transport-{:x}",
            base.artifact_identity(),
            Sha256::digest(ptx.as_bytes())
        );
        let mut bindings = BTreeMap::new();
        for layout in layouts {
            let key = [layout.heads, layout.key_dim, layout.value_dim];
            if !bindings.contains_key(&key) {
                bindings.insert(key, base.for_geometry(key[0], key[1], key[2])?);
            }
        }
        let module = format!("scalar_gdn_shared_{:x}", Sha256::digest(ptx.as_bytes()));
        device
            .load_ptx(ptx.into(), &module, FUNCTIONS)
            .map_err(|e| e.to_string())?;
        let mut functions = BTreeMap::new();
        for &name in FUNCTIONS {
            functions.insert(
                name,
                device
                    .get_func(&module, name)
                    .ok_or_else(|| format!("required shared scalar transport absent:{name}"))?,
            );
        }
        Ok(Arc::new(Self {
            identity,
            functions,
            bindings,
            limits,
            device,
        }))
    }
    pub(crate) fn require_geometry(&self, layout: &Layout) -> Result<(), String> {
        if !self
            .bindings
            .contains_key(&[layout.heads, layout.key_dim, layout.value_dim])
        {
            return Err("scalar geometry was not admitted to program before calibration".into());
        }
        Ok(())
    }
    pub(crate) fn binding(&self, layout: &Layout) -> Result<Native, String> {
        self.bindings
            .get(&[layout.heads, layout.key_dim, layout.value_dim])
            .cloned()
            .ok_or_else(|| "scalar geometry was not admitted to program before calibration".into())
    }
    pub fn context(&self) -> sys::CUcontext {
        *self.device.cu_primary_ctx()
    }
}
impl Drop for Program {
    fn drop(&mut self) {
        drain_or_fatal(&self.device, "shared scalar program");
    }
}

fn workspace_bytes(layouts: &[Layout]) -> Result<BTreeMap<&'static str, usize>, String> {
    if layouts.is_empty() {
        return Err("workspace requires actual phase layouts".into());
    }
    let mut bytes: BTreeMap<&'static str, usize> = BTreeMap::new();
    for layout in layouts {
        for (&name, &size) in &layout.workspace {
            bytes
                .entry(name)
                .and_modify(|old| *old = (*old).max(size))
                .or_insert(size);
        }
    }
    Ok(bytes)
}
pub struct Workspace {
    buffers: BTreeMap<&'static str, CudaSlice<u8>>,
    bytes: BTreeMap<&'static str, usize>,
    enqueue: Mutex<()>,
    // Allocation readiness belongs to the actual allocation/use streams.
    _ready: Option<Event>,
    pub(crate) phase: Stream,
    pub(crate) program: Arc<Program>,
}
pub struct Lease<'a> {
    pub(crate) workspace: &'a Workspace,
    _guard: MutexGuard<'a, ()>,
    _exclusive: PhantomData<Cell<()>>,
}
impl Workspace {
    pub fn new(program: Arc<Program>, layouts: &[Layout]) -> Result<Arc<Self>, String> {
        let phase = Stream::Decode(program.device.clone());
        Self::on_stream(program, layouts, phase)
    }
    pub(crate) fn on_prefill(
        program: Arc<Program>,
        layouts: &[Layout],
        owner: Arc<crate::gpu_prefill::PrefillComputeStream>,
    ) -> Result<Arc<Self>, String> {
        Self::on_stream(program, layouts, Stream::Prefill(owner))
    }
    fn on_stream(
        program: Arc<Program>,
        layouts: &[Layout],
        phase: Stream,
    ) -> Result<Arc<Self>, String> {
        phase.admit(&program.device)?;
        if layouts.is_empty() {
            return Err("workspace requires actual phase layouts".into());
        }
        for layout in layouts {
            program.require_geometry(layout)?;
            if layout.heads > program.limits[0] || layout.padded_capacity / 64 > program.limits[1] {
                return Err("phase layout exceeds actual CUDA grid limits".into());
            }
        }
        let bytes = workspace_bytes(layouts)?;
        // Event creation can fail; do it before allocating async-owned buffers.
        let allocating = Stream::Decode(program.device.clone());
        let ready_event = if phase.same(&allocating) {
            None
        } else {
            Some(Event::new(program.device.clone())?)
        };
        let mut buffers = BTreeMap::new();
        let result = (|| -> Result<(), String> {
            for (&name, &size) in &bytes {
                buffers.insert(
                    name,
                    program
                        .device
                        .alloc_zeros::<u8>(size)
                        .map_err(|e| e.to_string())?,
                );
            }
            Ok(())
        })();
        if let Err(error) = result {
            drop(buffers);
            drain_or_fatal(&program.device, "shared workspace construction cleanup");
            return Err(error);
        }
        let ready = if let Some(event) = ready_event {
            if let Err(e) = event.record(&allocating).and_then(|_| event.wait(&phase)) {
                phase.drain_or_fatal("scalar workspace readiness use-stream completion");
                allocating.drain_or_fatal("scalar workspace readiness allocation completion");
                drop(buffers);
                allocating.drain_or_fatal("scalar failed workspace allocation release");
                return Err(e);
            }
            Some(event)
        } else {
            None
        };
        Ok(Arc::new(Self {
            program,
            buffers,
            bytes,
            enqueue: Mutex::new(()),
            _ready: ready,
            phase,
        }))
    }
    pub(crate) fn stream(&self) -> sys::CUstream {
        self.phase.handle()
    }
    pub fn bytes(&self) -> Result<usize, String> {
        self.bytes.values().try_fold(0usize, |n, &b| {
            n.checked_add(b)
                .ok_or_else(|| "shared workspace byte overflow".into())
        })
    }
    pub fn lease(&self) -> Result<Lease<'_>, String> {
        self.program
            .device
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        let guard = match self.enqueue.try_lock() {
            Ok(x) => x,
            Err(TryLockError::WouldBlock) => {
                return Err("scalar workspace already has an exclusive enqueue phase".into())
            }
            Err(TryLockError::Poisoned(_)) => {
                return Err("scalar workspace enqueue phase is poisoned".into())
            }
        };
        Ok(Lease {
            workspace: self,
            _guard: guard,
            _exclusive: PhantomData,
        })
    }
    pub(crate) fn supports(&self, layout: &Layout) -> Result<(), String> {
        self.program.require_geometry(layout)?;
        for (name, &needed) in &layout.workspace {
            let actual = self
                .bytes
                .get(name)
                .ok_or("required scalar scratch role missing")?;
            if *actual < needed {
                return Err(format!(
                    "scalar phase workspace {name} requires {needed}bytes but owns {actual}"
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn ptr(&self, name: &str) -> u64 {
        *self.buffers[name].device_ptr()
    }
    pub fn buffer(&self, name: &str) -> Result<&CudaSlice<u8>, String> {
        self.buffers
            .get(name)
            .ok_or_else(|| format!("scalar workspace role absent:{name}"))
    }
}
// Lease serializes host enqueue sequences on this actual phase stream. Whole
// invocation Execution orders state handoffs between phases by GPU event. No
// per-layer host wait or bare-pointer stream rewrite is introduced.
impl Drop for Workspace {
    fn drop(&mut self) {
        self.phase
            .drain_or_fatal("shared scalar workspace actual-use completion");
        drop(std::mem::take(&mut self.buffers));
        drain_or_fatal(
            &self.program.device,
            "shared scalar workspace allocation release",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn each_role_uses_its_actual_largest_layer_extent() {
        let a = Layout::checked(3, 2, 5, 1).unwrap();
        let b = Layout::checked(2, 7, 3, 65).unwrap();
        let shared = workspace_bytes(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(shared["q.bf16"], 3584);
        assert_eq!(shared["v.bf16"], 1920);
        assert_eq!(shared["beta.bf16"], 512);
        assert_eq!(shared["gate.f32"], 1024);
        assert_eq!(shared["completed-state.f32"], 168);
        assert_eq!(shared["emitted.bf16"], 780);
        for (name, n) in &shared {
            assert!(*n >= a.workspace[name] && *n >= b.workspace[name]);
        }
        assert!(workspace_bytes(&[]).is_err());
    }
}
