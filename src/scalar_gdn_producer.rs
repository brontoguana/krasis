//! Shared BF16 scalar-GDN producer and consumer under operational qualification.
//! No activation or fallback: actual configured layers and phase owners are
//! required. Preparation kernels are the unchanged installed204 replay family.
use crate::{
    cuda_region::CudaRegion,
    model_tensor_owner::ModelTensorOwner,
    scalar_gdn_owner::{Inputs, Plane},
    scalar_gdn_projection::{Layout as Projection, Program as ProjectionProgram},
};
use cudarc::driver::{sys, CudaDevice, CudaFunction, CudaSlice};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::c_void,
    sync::{Arc, Mutex, MutexGuard},
};

pub(crate) struct Geometry {
    pub nk: usize,
    pub nv: usize,
    pub dk: usize,
    pub dv: usize,
    pub hr: usize,
    pub kernel: usize,
    pub scale: f32,
    pub eps: f32,
}
impl Geometry {
    fn key(&self) -> usize {
        self.nk * self.dk
    }
    fn value(&self) -> usize {
        self.nv * self.dv
    }
    fn normalized_key(&self) -> usize {
        self.nv * self.dk
    }
    fn conv(&self) -> usize {
        2 * self.key() + self.value()
    }
    fn validate(&self) -> Result<(), String> {
        for n in [self.nk, self.nv, self.dk, self.dv, self.hr, self.kernel] {
            if n == 0 || i32::try_from(n).is_err() {
                return Err("invalid loaded scalar producer dimension".into());
            }
        }
        if self.nk.checked_mul(self.hr) != Some(self.nv)
            || !self.scale.is_finite()
            || self.scale <= 0.
            || !self.eps.is_finite()
            || self.eps < 0.
        {
            return Err("invalid loaded scalar producer head/normalization contract".into());
        }
        for n in [self.key(), self.value(), self.normalized_key(), self.conv()] {
            i32::try_from(n).map_err(|_| "scalar producer row exceeds CUDA ABI")?;
            n.checked_mul(std::mem::size_of::<f32>())
                .ok_or("scalar producer plane overflow")?;
        }
        self.conv()
            .checked_mul(self.kernel)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or("scalar producer convolution extent overflow")?;
        Ok(())
    }
}
pub(crate) struct Layer {
    pub geometry: Geometry,
    pub qkvz: Projection,
    pub ba: Projection,
    pub out: Projection,
    pub conv: Arc<ModelTensorOwner>,
    pub conv_weight: u64,
    pub a_log: u64,
    pub dt_bias: u64,
    pub norm_weight: u64,
}
impl Layer {
    fn validate(&self, context: sys::CUcontext) -> Result<(), String> {
        let g = &self.geometry;
        g.validate()?;
        for w in [&self.qkvz, &self.ba, &self.out] {
            w.validate()?;
        }
        if self.qkvz.rows != 2 * g.key() + 2 * g.value()
            || self.ba.rows != 2 * g.nv
            || self.qkvz.cols != self.ba.cols
            || self.out.cols != g.value()
            || self.out.rows != self.qkvz.cols
        {
            return Err("actual scalar producer projection dimensions disagree".into());
        }
        let (start, end) = self.conv.extent()?;
        if self.conv.context() != context
            || end - start
                != u64::try_from(g.conv() * g.kernel * std::mem::size_of::<f32>())
                    .map_err(|e| e.to_string())?
        {
            return Err("actual scalar convolution tensor context/extent differs".into());
        }
        for p in [
            start,
            self.conv_weight,
            self.a_log,
            self.dt_bias,
            self.norm_weight,
        ] {
            if p == 0 || p % std::mem::align_of::<f32>() as u64 != 0 {
                return Err("actual scalar constant pointer/alignment differs".into());
            }
        }
        Ok(())
    }
}
#[derive(Clone, Copy)]
pub(crate) enum Role {
    Qkvz,
    Ba,
    Out,
}
#[derive(Default)]
struct Capacity {
    qkvz: usize,
    ba: usize,
    key: usize,
    value: usize,
    normalized_key: usize,
    heads: usize,
}
struct Buffers {
    qkvz: CudaSlice<u16>,
    ba: CudaSlice<u16>,
    q: CudaSlice<u16>,
    k: CudaSlice<u16>,
    v: CudaSlice<u16>,
    z: CudaSlice<u16>,
    a: CudaSlice<u16>,
    b: CudaSlice<u16>,
    qconv: CudaSlice<u16>,
    kconv: CudaSlice<u16>,
    vconv: CudaSlice<u16>,
    qnorm: CudaSlice<u16>,
    knorm: CudaSlice<u16>,
    beta: CudaSlice<u16>,
    gatebf: CudaSlice<u16>,
    norm: CudaSlice<u16>,
    gate: CudaSlice<f32>,
    betaf: CudaSlice<f32>,
}
impl Buffers {
    fn new(device: &Arc<CudaDevice>, c: &Capacity) -> Result<Self, String> {
        let bf = |n| device.alloc_zeros::<u16>(n).map_err(|e| e.to_string());
        let fp = |n| device.alloc_zeros::<f32>(n).map_err(|e| e.to_string());
        Ok(Self {
            qkvz: bf(c.qkvz)?,
            ba: bf(c.ba)?,
            q: bf(c.key)?,
            k: bf(c.key)?,
            v: bf(c.value)?,
            z: bf(c.value)?,
            a: bf(c.heads)?,
            b: bf(c.heads)?,
            qconv: bf(c.key)?,
            kconv: bf(c.key)?,
            vconv: bf(c.value)?,
            qnorm: bf(c.normalized_key)?,
            knorm: bf(c.normalized_key)?,
            beta: bf(c.heads)?,
            gatebf: bf(c.heads)?,
            norm: bf(c.value)?,
            gate: fp(c.heads)?,
            betaf: fp(c.heads)?,
        })
    }
}
const FUNCTIONS: &[&str] = &[
    "la_uninterleave_qkvz_kernel",
    "la_uninterleave_ba_kernel",
    "la_fused_conv1d_silu_bf16_kernel",
    "la_update_conv_state_kernel",
    "la_fused_repeat_l2norm_bf16_kernel",
    "la_compute_gate_beta_bf16_kernel",
    "la_compute_gate_beta_kernel",
    "la_gated_rmsnorm_bf16in_kernel",
];
pub(crate) struct Producer {
    projection: Arc<ProjectionProgram>,
    layers: BTreeMap<usize, Layer>,
    functions: BTreeMap<&'static str, CudaFunction>,
    buffers: Mutex<Buffers>,
    grid: [usize; 3],
    device: Arc<CudaDevice>,
    pub(crate) identity: String,
}
enum Arg {
    P(u64),
    I(i32),
    F(f32),
}
impl Arg {
    fn address(&mut self) -> *mut c_void {
        match self {
            Self::P(x) => (x as *mut u64).cast(),
            Self::I(x) => (x as *mut i32).cast(),
            Self::F(x) => (x as *mut f32).cast(),
        }
    }
}
fn threads(n: usize, cap: usize) -> u32 {
    n.min(cap).div_ceil(32).max(1) as u32 * 32
}
impl Producer {
    pub(crate) fn context(&self) -> sys::CUcontext {
        *self.device.cu_primary_ctx()
    }
    pub(crate) fn layer_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.layers.keys().copied()
    }
    /// Real weights/constant tensor owners are retained by the configured model
    /// through the entire storage execution domain. Conv is additionally typed.
    pub(crate) fn new(
        device: Arc<CudaDevice>,
        layers: BTreeMap<usize, Layer>,
    ) -> Result<Arc<Self>, String> {
        if layers.is_empty() {
            return Err("scalar producer requires actual locally executed layers".into());
        }
        device.bind_to_thread().map_err(|e| e.to_string())?;
        let context = *device.cu_primary_ctx();
        let mut capacity = Capacity::default();
        for layer in layers.values() {
            layer.validate(context)?;
            let g = &layer.geometry;
            capacity.qkvz = capacity.qkvz.max(layer.qkvz.rows);
            capacity.ba = capacity.ba.max(layer.ba.rows);
            capacity.key = capacity.key.max(g.key());
            capacity.value = capacity.value.max(g.value());
            capacity.normalized_key = capacity.normalized_key.max(g.normalized_key());
            capacity.heads = capacity.heads.max(g.nv);
        }
        let read = |a| {
            device
                .attribute(a)
                .map_err(|e| e.to_string())
                .and_then(|x| usize::try_from(x).map_err(|e| e.to_string()))
        };
        use sys::CUdevice_attribute as A;
        if read(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)? < 1024 {
            return Err(
                "actual device cannot execute existing scalar prefill launch geometry".into(),
            );
        }
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/prefill_kernels.ptx"));
        device
            .load_ptx(ptx.into(), "scalar_gdn_common_producer_v1", FUNCTIONS)
            .map_err(|e| e.to_string())?;
        let functions = FUNCTIONS
            .iter()
            .map(|&name| {
                device
                    .get_func("scalar_gdn_common_producer_v1", name)
                    .ok_or_else(|| format!("actual scalar producer entry missing:{name}"))
                    .map(|f| (name, f))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let projection = ProjectionProgram::load(device.clone())?;
        let identity = format!(
            "scalar-bf16-producer-v1:{}:prefill-ptx-{:x}",
            projection.identity,
            Sha256::digest(ptx.as_bytes())
        );
        let buffers = Mutex::new(Buffers::new(&device, &capacity)?);
        let grid = [
            read(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
            read(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
            read(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
        ];
        for layer in layers.values() {
            let g = &layer.geometry;
            if g.conv() > grid[0] || g.nv > grid[1] {
                return Err("actual scalar producer layer exceeds device grid".into());
            }
        }
        Ok(Arc::new(Self {
            projection,
            layers,
            functions,
            buffers,
            grid,
            device,
            identity,
        }))
    }
    pub(crate) fn projection_layout(
        &self,
        layer: usize,
        role: Role,
    ) -> Result<&Projection, String> {
        let w = self
            .layers
            .get(&layer)
            .ok_or("scalar projection layer is not cold-admitted")?;
        Ok(match role {
            Role::Qkvz => &w.qkvz,
            Role::Ba => &w.ba,
            Role::Out => &w.out,
        })
    }
    /// # Safety
    /// Same real plane, immutable weight and whole-phase ownership contract as
    /// ProjectionProgram::enqueue; no temporary allocation or Python wrapper.
    pub(crate) unsafe fn project(
        &self,
        layer: usize,
        role: Role,
        input: &Plane<'_, u16>,
        output: &Plane<'_, u16>,
        rows: usize,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        self.projection.enqueue(
            self.projection_layout(layer, role)?,
            input,
            output,
            None,
            rows,
            stream,
        )
    }
    unsafe fn launch<const N: usize>(
        &self,
        name: &str,
        grid: (u32, u32, u32),
        block: u32,
        shared: u32,
        stream: sys::CUstream,
        args: &mut [Arg; N],
    ) -> Result<(), String> {
        if stream.is_null()
            || grid.0 == 0
            || grid.1 == 0
            || grid.2 == 0
            || [grid.0, grid.1, grid.2]
                .iter()
                .zip(self.grid)
                .any(|(&n, max)| n as usize > max)
        {
            return Err("scalar producer invocation exceeds actual stream/grid contract".into());
        }
        let mut addresses = [std::ptr::null_mut::<c_void>(); N];
        for (slot, arg) in addresses.iter_mut().zip(args) {
            *slot = arg.address();
        }
        sys::lib()
            .cuLaunchKernel(
                self.functions[name].cu_function(),
                grid.0,
                grid.1,
                grid.2,
                block,
                1,
                1,
                shared,
                stream,
                addresses.as_mut_ptr(),
                std::ptr::null_mut(),
            )
            .result()
            .map_err(|e| format!("scalar producer {name}:{e}"))
    }
    /// # Safety
    /// Caller retains actual model/stream/position owners inside whole decode
    /// execution. Consume native output and enqueue finish() before dropping the
    /// returned lease or allowing another producer to reuse these real buffers.
    pub(crate) unsafe fn prepare_decode<'a>(
        &'a self,
        layer: usize,
        input: &Plane<'_, u16>,
        stream: sys::CUstream,
    ) -> Result<Prepared<'a>, String> {
        use Arg::{F, I, P};
        let w = self
            .layers
            .get(&layer)
            .ok_or("scalar producer layer is not cold-admitted")?;
        let g = &w.geometry;
        let b = self
            .buffers
            .try_lock()
            .map_err(|_| "scalar producer decode arena is held/poisoned")?;
        self.projection.enqueue(
            &w.qkvz,
            input,
            &Plane::checked(&b.qkvz, 0, w.qkvz.rows)?,
            None,
            1,
            stream,
        )?;
        self.projection.enqueue(
            &w.ba,
            input,
            &Plane::checked(&b.ba, 0, w.ba.rows)?,
            None,
            1,
            stream,
        )?;
        let p = |x: &CudaSlice<u16>, n| Plane::checked(x, 0, n).map(|p| p.pointer());
        self.launch(
            "la_uninterleave_qkvz_kernel",
            (1, 1, 1),
            threads(w.qkvz.rows, 1024),
            0,
            stream,
            &mut [
                P(p(&b.q, g.key())?),
                P(p(&b.k, g.key())?),
                P(p(&b.v, g.value())?),
                P(p(&b.z, g.value())?),
                P(p(&b.qkvz, w.qkvz.rows)?),
                I(g.nk as i32),
                I(g.dk as i32),
                I(g.hr as i32),
                I(g.dv as i32),
            ],
        )?;
        self.launch(
            "la_uninterleave_ba_kernel",
            (1, 1, 1),
            threads(w.ba.rows, 1024),
            0,
            stream,
            &mut [
                P(p(&b.b, g.nv)?),
                P(p(&b.a, g.nv)?),
                P(p(&b.ba, w.ba.rows)?),
                I(g.nk as i32),
                I(g.hr as i32),
            ],
        )?;
        let conv = w.conv.extent()?.0;
        self.launch(
            "la_fused_conv1d_silu_bf16_kernel",
            (1, 1, 1),
            threads(g.conv(), 1024),
            0,
            stream,
            &mut [
                P(p(&b.qconv, g.key())?),
                P(p(&b.kconv, g.key())?),
                P(p(&b.vconv, g.value())?),
                P(conv),
                P(p(&b.q, g.key())?),
                P(p(&b.k, g.key())?),
                P(p(&b.v, g.value())?),
                P(w.conv_weight),
                I(1),
                I(g.key() as i32),
                I(g.value() as i32),
                I(g.kernel as i32),
            ],
        )?;
        self.launch(
            "la_update_conv_state_kernel",
            (g.conv() as u32, 1, 1),
            threads(g.kernel, 32),
            0,
            stream,
            &mut [
                P(conv),
                P(p(&b.q, g.key())?),
                P(p(&b.k, g.key())?),
                P(p(&b.v, g.value())?),
                I(1),
                I(g.key() as i32),
                I(g.value() as i32),
                I(g.kernel as i32),
            ],
        )?;
        let nt = threads(g.dk, 256);
        for (output, source, scale) in [(&b.qnorm, &b.qconv, g.scale), (&b.knorm, &b.kconv, 1.)] {
            self.launch(
                "la_fused_repeat_l2norm_bf16_kernel",
                (1, g.nv as u32, 1),
                nt,
                nt * 4,
                stream,
                &mut [
                    P(p(output, g.normalized_key())?),
                    P(p(source, g.key())?),
                    I(g.nk as i32),
                    I(g.dk as i32),
                    I(g.hr as i32),
                    F(scale),
                ],
            )?;
        }
        self.launch(
            "la_compute_gate_beta_bf16_kernel",
            (1, 1, 1),
            threads(g.nv, 1024),
            0,
            stream,
            &mut [
                P(p(&b.beta, g.nv)?),
                P(p(&b.gatebf, g.nv)?),
                P(p(&b.b, g.nv)?),
                P(p(&b.a, g.nv)?),
                P(w.a_log),
                P(w.dt_bias),
                I(g.nv as i32),
            ],
        )?;
        self.launch(
            "la_compute_gate_beta_kernel",
            (1, 1, 1),
            threads(g.nv, 1024),
            0,
            stream,
            &mut [
                P(Plane::checked(&b.betaf, 0, g.nv)?.pointer()),
                P(Plane::checked(&b.gate, 0, g.nv)?.pointer()),
                P(p(&b.b, g.nv)?),
                P(p(&b.a, g.nv)?),
                P(w.a_log),
                P(w.dt_bias),
                I(g.nv as i32),
            ],
        )?;
        Ok(Prepared {
            producer: self,
            layer: w,
            buffers: b,
        })
    }
}
pub(crate) struct Prepared<'a> {
    producer: &'a Producer,
    layer: &'a Layer,
    buffers: MutexGuard<'a, Buffers>,
}
impl Prepared<'_> {
    pub(crate) fn inputs(&self) -> Result<Inputs<'_>, String> {
        let g = &self.layer.geometry;
        let b = &self.buffers;
        Ok(Inputs {
            q: Plane::checked(&b.qnorm, 0, g.normalized_key())?,
            k: Plane::checked(&b.knorm, 0, g.normalized_key())?,
            v: Plane::checked(&b.vconv, 0, g.value())?,
            beta: Plane::checked(&b.beta, 0, g.nv)?,
            gate: Plane::checked(&b.gate, 0, g.nv)?,
        })
    }
    /// # Safety
    /// Native emitted and output owners survive completion of the enclosing
    /// invocation; this producer lease stays held until both consumers enqueue.
    pub(crate) unsafe fn finish(
        &self,
        emitted: &Plane<'_, u16>,
        output: &Plane<'_, u16>,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        use Arg::{F, I, P};
        let g = &self.layer.geometry;
        let b = &self.buffers;
        emitted.require_context_extent(*self.producer.device.cu_primary_ctx(), g.value())?;
        let norm = Plane::checked(&b.norm, 0, g.value())?;
        let z = Plane::checked(&b.z, 0, g.value())?;
        let nt = threads(g.dv, 256);
        self.producer.launch(
            "la_gated_rmsnorm_bf16in_kernel",
            (1, g.nv as u32, 1),
            nt,
            nt * 4,
            stream,
            &mut [
                P(norm.pointer()),
                P(emitted.pointer()),
                P(z.pointer()),
                P(self.layer.norm_weight),
                I(g.nv as i32),
                I(g.dv as i32),
                F(g.eps),
            ],
        )?;
        self.producer
            .projection
            .enqueue(&self.layer.out, &norm, output, None, 1, stream)
    }
}
