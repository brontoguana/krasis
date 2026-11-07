//! Bounded Qwen3.5 image encoder. Startup loads immutable host weights;
//! every block executes and stages weights entirely in Rust/CUDA.
use crate::{
    dense_stream::DenseStream,
    weights::safetensors_io::{Dtype, MmapSafetensors},
};
use cudarc::{
    cublas::{sys as bs, CudaBlas},
    driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig},
};
use pyo3::prelude::*;
use std::{collections::HashMap, path::Path, sync::Arc};
#[derive(Clone)]
struct Weight {
    ptr: u64,
    shape: Vec<usize>,
}
#[pyclass]
pub struct Qwen35Vision {
    device: Arc<CudaDevice>,
    blas: CudaBlas,
    bank: Arc<DenseStream>,
    weights: HashMap<String, Weight>,
    hidden: usize,
    inter: usize,
    out: usize,
    heads: usize,
    depth: usize,
    patch: usize,
    side: usize,
    max_patches: usize,
    rope_inv_freq: CudaSlice<f32>,
    lt_workspace: CudaSlice<u8>,
    flash: Option<crate::gpu_prefill::FlashAttnFwdFn>,
}
fn pyerr(e: impl ToString) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}
#[pymethods]
impl Qwen35Vision {
    #[new]
    #[pyo3(signature=(path,ordinal=0,max_patches=1024))]
    fn new(path: &str, ordinal: usize, max_patches: usize) -> PyResult<Self> {
        Self::load(Path::new(path), ordinal, max_patches).map_err(pyerr)
    }
    fn encode(
        &self,
        pixels_ptr: usize,
        grid_h: usize,
        grid_w: usize,
        output_ptr: usize,
    ) -> PyResult<()> {
        self.forward(pixels_ptr as u64, grid_h, grid_w, output_ptr as u64)
            .map_err(pyerr)
    }
    fn encode_debug(
        &self,
        pixels_ptr: usize,
        grid_h: usize,
        grid_w: usize,
        output_ptr: usize,
    ) -> PyResult<HashMap<String, Vec<u16>>> {
        let mut capture = HashMap::new();
        self.forward_inner(
            pixels_ptr as u64,
            grid_h,
            grid_w,
            output_ptr as u64,
            Some(&mut capture),
            None,
        )
        .map_err(pyerr)?;
        Ok(capture)
    }
    fn block_debug(&self, input_ptr:usize, grid_h:usize, grid_w:usize, block:usize, output_ptr:usize) -> PyResult<HashMap<String,Vec<u16>>> {
        if block > self.depth { return Err(pyerr("invalid vision block")); }
        let mut capture=HashMap::new();
        self.forward_inner(input_ptr as u64,grid_h,grid_w,output_ptr as u64,Some(&mut capture),Some(block)).map_err(pyerr)?;
        Ok(capture)
    }
    fn memory_info(&self) -> PyResult<String> {
        self.bank.stats().map_err(pyerr)
    }
}
impl Qwen35Vision {
    fn load(path: &Path, ordinal: usize, max_patches: usize) -> Result<Self, String> {
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path.join("config.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if config["model_type"] != "qwen3_5" {
            return Err("native image encoder requires qwen3_5 checkpoint".into());
        }
        let c = &config["vision_config"];
        let get = |key: &str| {
            c[key]
                .as_u64()
                .filter(|v| *v > 0)
                .map(|v| v as usize)
                .ok_or_else(|| format!("missing vision {key}"))
        };
        let (hidden, inter, out, heads, depth, patch, side) = (
            get("hidden_size")?,
            get("intermediate_size")?,
            get("out_hidden_size")?,
            get("num_heads")?,
            get("depth")?,
            get("patch_size")?,
            (get("num_position_embeddings")? as f64).sqrt() as usize,
        );
        if hidden % heads != 0
            || hidden / heads > 128
            || hidden / heads % 4 != 0
            || get("spatial_merge_size")? != 2
            || get("temporal_patch_size")? != 2
            || get("in_channels")? != 3
            || c["hidden_act"] != "gelu_pytorch_tanh"
            || max_patches > 1024
            || max_patches < 4
        {
            return Err("unsupported Qwen3.5 vision geometry or patch limit".into());
        }
        if c.get("deepstack_visual_indexes")
            .and_then(|v| v.as_array())
            .is_some_and(|a| !a.is_empty())
        {
            return Err("Qwen3.5 image path does not support deepstack variants".into());
        }
        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path.join("model.safetensors.index.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let map = index["weight_map"]
            .as_object()
            .ok_or("missing vision weight map")?;
        let group = |name: &str| -> Result<usize, String> {
            if let Some(rest) = name.strip_prefix("blocks.") {
                rest.split('.')
                    .next()
                    .unwrap()
                    .parse::<usize>()
                    .map(|i| i + 1)
                    .map_err(|e| e.to_string())
            } else if name.starts_with("merger.") {
                Ok(depth + 1)
            } else {
                Ok(0)
            }
        };
        let mut shards = HashMap::new();
        let mut entries = Vec::new();
        let mut sizes = HashMap::<usize, usize>::new();
        for (full, file) in map {
            if let Some(name) = full.strip_prefix("model.visual.") {
                let file = file.as_str().ok_or("invalid shard name")?;
                if Path::new(file).components().count() != 1 {
                    return Err("vision shard must be local filename".into());
                }
                if !shards.contains_key(file) {
                    shards.insert(
                        file.to_owned(),
                        MmapSafetensors::open(&path.join(file)).map_err(|e| e.to_string())?,
                    );
                }
                let shard = &shards[file];
                let info = shard.tensor_info(full).ok_or("missing vision tensor")?;
                if info.dtype != Dtype::Bf16 {
                    return Err(format!("native vision requires BF16 source: {full}"));
                }
                let bytes = shard.tensor_data(full).map_err(|e| e.to_string())?.len();
                let g = group(name)?;
                *sizes.entry(g).or_default() += crate::dense_stream::align(bytes)?;
                entries.push((
                    name.to_owned(),
                    full.to_owned(),
                    file.to_owned(),
                    g,
                    info.shape.clone(),
                ));
            }
        }
        let device = CudaDevice::new_with_stream(ordinal).map_err(|e| e.to_string())?;
        let bank = DenseStream::new(
            device.clone(),
            *sizes.values().max().ok_or("missing vision weights")?,
        )?;
        let mut weights = HashMap::new();
        for (name, full, file, g, shape) in entries {
            let bytes = shards[&file]
                .tensor_data(&full)
                .map_err(|e| e.to_string())?;
            let ptr = unsafe { bank.register(g, bytes.as_ptr() as usize, bytes.len()) }?;
            weights.insert(name, Weight { ptr, shape });
        }
        bank.seal()?;
        let required = |name: &str, shape: &[usize]| -> Result<(), String> {
            let actual = weights
                .get(name)
                .ok_or_else(|| format!("missing vision tensor {name}"))?;
            if actual.shape != shape {
                return Err(format!(
                    "vision shape {name}: {:?} != {shape:?}",
                    actual.shape
                ));
            }
            Ok(())
        };
        required("patch_embed.proj.weight", &[hidden, 3, 2, patch, patch])?;
        required("patch_embed.proj.bias", &[hidden])?;
        required("pos_embed.weight", &[side * side, hidden])?;
        for i in 0..depth {
            let prefix = format!("blocks.{i}");
            for norm in ["norm1", "norm2"] {
                for suffix in ["weight", "bias"] {
                    required(&format!("{prefix}.{norm}.{suffix}"), &[hidden])?;
                }
            }
            for (name, rows, cols) in [
                ("attn.qkv", 3 * hidden, hidden),
                ("attn.proj", hidden, hidden),
                ("mlp.linear_fc1", inter, hidden),
                ("mlp.linear_fc2", hidden, inter),
            ] {
                required(&format!("{prefix}.{name}.weight"), &[rows, cols])?;
                required(&format!("{prefix}.{name}.bias"), &[rows])?;
            }
        }
        for suffix in ["weight", "bias"] {
            required(&format!("merger.norm.{suffix}"), &[hidden])?;
        }
        for (name, rows, cols) in [
            ("merger.linear_fc1", hidden * 4, hidden * 4),
            ("merger.linear_fc2", out, hidden * 4),
        ] {
            required(&format!("{name}.weight"), &[rows, cols])?;
            required(&format!("{name}.bias"), &[rows])?;
        }
        let names = [
            "qv_finish",
            "qv_activate",
            "qv_norm",
            "qv_add",
            "qv_position",
            "qv_rope",
            "qv_attention",
            "qv_split_pad",
            "qv_unpad",
        ];
        device
            .load_ptx(
                cudarc::nvrtc::Ptx::from_src(include_str!(concat!(
                    env!("OUT_DIR"),
                    "/qwen35_vision.ptx"
                ))),
                "qwen35_vision",
                &names,
            )
            .map_err(|e| e.to_string())?;
        let blas = CudaBlas::new(device.clone()).map_err(|e| e.to_string())?;
        let quarter = hidden/heads/4;
        // Reference constructs inv_freq on CPU in FP32 before moving to CUDA.
        let frequencies: Vec<f32> = (0..quarter).map(|i|1f32/10000f32.powf(i as f32/quarter as f32)).collect();
        let rope_inv_freq = device.htod_sync_copy(&frequencies).map_err(|e|e.to_string())?;
        let lt_workspace = device.alloc_zeros::<u8>(1024*1024).map_err(|e|e.to_string())?;
        Ok(Self {
            rope_inv_freq,
            lt_workspace,
            device,
            blas,
            bank,
            weights,
            hidden,
            inter,
            out,
            heads,
            depth,
            patch,
            side,
            max_patches,
            flash: if std::env::var("KRASIS_QWEN_VISION_ATTENTION").as_deref() == Ok("portable") {
                None
            } else {
                crate::gpu_prefill::load_flash_attn()
            },
        })
    }
    fn w(&self, name: &str) -> Result<&Weight, String> {
        self.weights
            .get(name)
            .ok_or_else(|| format!("missing vision weight {name}"))
    }
    fn alloc(&self, n: usize) -> Result<CudaSlice<u16>, String> {
        self.device.alloc_zeros::<u16>(n).map_err(|e| e.to_string())
    }
    fn linear(
        &self,
        x: u64,
        out: u64,
        m: usize,
        name: &str,
        act: i32,
        tmp: &CudaSlice<f32>,
    ) -> Result<(), String> {
        let w = self.w(&format!("{name}.weight"))?;
        let n = w.shape[0];
        let k = w.shape[1..].iter().product::<usize>();
        let bias = self
            .weights
            .get(&format!("{name}.bias"))
            .map_or(0, |w| w.ptr);
        if act != 3 && std::env::var("KRASIS_QWEN_VISION_LINEAR").as_deref() != Ok("fp32") {
            self.linear_lt(x,w.ptr,bias,out,m,n,k)?;
            if act != 0 {
                unsafe { self.device.get_func("qwen35_vision","qv_activate").ok_or("activation kernel")?
                    .launch(LaunchConfig::for_num_elems((m*n) as u32),(out,(m*n) as i32,act)).map_err(|e|e.to_string())?; }
            }
            return Ok(());
        }
        let alpha = 1f32;
        let beta = 0f32;
        unsafe {
            let status = bs::lib().cublasGemmEx(
                *self.blas.handle(),
                bs::cublasOperation_t::CUBLAS_OP_T,
                bs::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const _ as _,
                w.ptr as _,
                bs::cudaDataType::CUDA_R_16BF,
                k as i32,
                x as _,
                bs::cudaDataType::CUDA_R_16BF,
                k as i32,
                &beta as *const _ as _,
                *tmp.device_ptr() as _,
                bs::cudaDataType::CUDA_R_32F,
                n as i32,
                bs::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                bs::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            );
            if status != bs::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(format!("vision {name} GEMM {status:?}"));
            }
        }
        unsafe {
            self.device
                .get_func("qwen35_vision", "qv_finish")
                .ok_or("missing qv_finish")?
                .launch(
                    LaunchConfig::for_num_elems((m * n) as u32),
                    (*tmp.device_ptr(), bias, out, (m * n) as i32, n as i32, act),
                )
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    fn norm(&self, x: u64, out: u64, m: usize, name: &str) -> Result<(), String> {
        unsafe {
            self.device
                .get_func("qwen35_vision", "qv_norm")
                .ok_or("norm kernel")?
                .launch(
                    LaunchConfig {
                        grid_dim: (m as u32, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        x,
                        self.w(&format!("{name}.weight"))?.ptr,
                        self.w(&format!("{name}.bias"))?.ptr,
                        out,
                        self.hidden as i32,
                    ),
                )
                .map_err(|e| e.to_string())
        }
    }
    fn record(&self, capture: &mut Option<&mut HashMap<String, Vec<u16>>>, name: String, value: &CudaSlice<u16>) -> Result<(), String> {
        // Explicit offline diagnostic intervention; no automatic serving input.
        if let Ok(directory)=std::env::var("KRASIS_QWEN_VISION_SUBSTITUTION_DIR") {
            let path=std::path::Path::new(&directory).join(format!("{name}.bf16"));
            match std::fs::read(&path) {
                Ok(packet)=>{
                    if packet.len()!=value.len()*2 || packet.len()>32*1024*1024 {return Err("invalid vision substitution extent".into());}
                    self.device.synchronize().map_err(|e|e.to_string())?;
                    unsafe {cudarc::driver::sys::lib().cuMemcpyHtoD_v2(*value.device_ptr(),packet.as_ptr() as _,packet.len()).result().map_err(|e|e.to_string())?;}
                }
                Err(e) if e.kind()==std::io::ErrorKind::NotFound=>{},
                Err(e)=>return Err(e.to_string()),
            }
        }
        if let Some(c) = capture.as_deref_mut() {
            c.insert(name, self.device.dtoh_sync_copy(value).map_err(|e| e.to_string())?);
        }
        Ok(())
    }
    fn add(&self, x: u64, y: u64, n: usize) -> Result<(), String> {
        unsafe {
            self.device
                .get_func("qwen35_vision", "qv_add")
                .ok_or("add kernel")?
                .launch(LaunchConfig::for_num_elems(n as u32), (x, y, n as i32))
                .map_err(|e| e.to_string())
        }
    }
    fn forward(&self, pixels: u64, gh: usize, gw: usize, out: u64) -> Result<(), String> {
        self.forward_inner(pixels, gh, gw, out, None, None)
    }
    fn forward_inner(
        &self,
        pixels: u64,
        gh: usize,
        gw: usize,
        out: u64,
        mut capture: Option<&mut HashMap<String, Vec<u16>>>,
        replay: Option<usize>,
    ) -> Result<(), String> {
        let n = gh.checked_mul(gw).ok_or("image extent overflow")?;
        if pixels == 0
            || out == 0
            || gh == 0
            || gw == 0
            || gh % 2 != 0
            || gw % 2 != 0
            || n > self.max_patches
        {
            return Err(
                "one image requires even patch grid within configured 1024-patch maximum".into(),
            );
        }
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        let (free, _) = cudarc::driver::result::mem_get_info().map_err(|e| e.to_string())?;
        let peak = n * (7*self.hidden+self.inter.max(self.hidden))*2
            + n*self.inter.max(3*self.hidden).max(self.out)*4
            + n*self.heads*128*8 + n*self.heads*4
            + 8*n*self.heads*(96+1)*4;
        if free < peak + 600 * 1024 * 1024 {
            return Err("vision would breach 600 MiB reserve".into());
        }
        let x = self.alloc(n * self.hidden)?;
        let a = self.alloc(n * self.hidden)?;
        let b = self.alloc(n * self.hidden)?;
        let qkv = self.alloc(n * self.hidden * 3)?;
        let mid = self.alloc(n * self.inter.max(self.hidden))?;
        let merged = self.alloc(n * self.hidden)?;
        let tmp = self
            .device
            .alloc_zeros::<f32>(n * self.inter.max(self.hidden * 3).max(self.out))
            .map_err(|e| e.to_string())?;
        let (xp, ap, bp, qp, mp, mergep) = (
            *x.device_ptr(),
            *a.device_ptr(),
            *b.device_ptr(),
            *qkv.device_ptr(),
            *mid.device_ptr(),
            *merged.device_ptr(),
        );
        let compatible_sdpa = self.hidden/self.heads == 72
            && !matches!(std::env::var("KRASIS_QWEN_VISION_ATTENTION").as_deref(),Ok("fast")|Ok("portable"));
        let padded = if compatible_sdpa {72} else {((self.hidden / self.heads + 31) / 32 * 32).max(64)};
        let split_buffers = if compatible_sdpa {
            Some((self.device.alloc_zeros::<f32>(8*self.heads*n*96).map_err(|e|e.to_string())?,
                  self.device.alloc_zeros::<f32>(8*self.heads*n).map_err(|e|e.to_string())?))
        } else {None};
        let flash_buffers = if self.flash.is_some() || compatible_sdpa {
            Some((
                self.alloc(n * self.heads * padded)?,
                self.alloc(n * self.heads * padded)?,
                self.alloc(n * self.heads * padded)?,
                self.alloc(n * self.heads * padded)?,
                self.device
                    .alloc_zeros::<f32>(n * self.heads)
                    .map_err(|e| e.to_string())?,
            ))
        } else {
            None
        };
        let stream = *self.device.cu_stream();
        if replay.is_some() {
            unsafe { cudarc::driver::sys::lib().cuMemcpyDtoDAsync_v2(xp,pixels,n*self.hidden*2,stream).result().map_err(|e|e.to_string())?; }
        } else {
        self.bank.stage(0, stream)?;
        self.linear(pixels, xp, n, "patch_embed.proj", 3, &tmp)?;
        if let Some(c) = capture.as_deref_mut() {
            c.insert(
                "patch".into(),
                self.device.dtoh_sync_copy(&x).map_err(|e| e.to_string())?,
            );
        }
        unsafe {
            self.device
                .get_func("qwen35_vision", "qv_position")
                .ok_or("position kernel")?
                .launch(
                    LaunchConfig::for_num_elems((n * self.hidden) as u32),
                    (
                        xp,
                        self.w("pos_embed.weight")?.ptr,
                        n as i32,
                        self.hidden as i32,
                        gh as i32,
                        gw as i32,
                        self.side as i32,
                    ),
                )
                .map_err(|e| e.to_string())?;
        }
        if let Some(c) = capture.as_deref_mut() {
            c.insert(
                "patch_position".into(),
                self.device.dtoh_sync_copy(&x).map_err(|e| e.to_string())?,
            );
        }
        }
        for i in 0..self.depth {
            if replay.is_some_and(|j| j != i) { continue; }
            self.bank.stage(i + 1, stream)?;
            let p = format!("blocks.{i}");
            self.norm(xp, ap, n, &format!("{p}.norm1"))?;
            self.record(&mut capture, format!("{i}.norm1"), &a)?;
            if i == 0 {
                if let Some(c) = capture.as_deref_mut() {
                    c.insert(
                        "norm1".into(),
                        self.device.dtoh_sync_copy(&a).map_err(|e| e.to_string())?,
                    );
                }
            }
            self.linear(ap, qp, n, &format!("{p}.attn.qkv"), 0, &tmp)?;
            self.record(&mut capture, format!("{i}.qkv"), &qkv)?;
            if i == 0 {
                if let Some(c) = capture.as_deref_mut() {
                    c.insert(
                        "qkv".into(),
                        self.device
                            .dtoh_sync_copy(&qkv)
                            .map_err(|e| e.to_string())?,
                    );
                }
            }
            unsafe {
                self.device
                    .get_func("qwen35_vision", "qv_rope")
                    .ok_or("rope kernel")?
                    .launch(
                        LaunchConfig::for_num_elems(
                            (n * self.heads * (self.hidden / self.heads)) as u32,
                        ),
                        (
                            qp,
                            n as i32,
                            self.heads as i32,
                            (self.hidden / self.heads) as i32,
                            gw as i32,
                            *self.rope_inv_freq.device_ptr(),
                        ),
                    )
                    .map_err(|e| e.to_string())?;
                if let Some((q,k,v,output,lse)) = flash_buffers.as_ref() {
                    self.device
                        .get_func("qwen35_vision", "qv_split_pad")
                        .ok_or("split kernel")?
                        .launch(
                            LaunchConfig::for_num_elems((n * self.heads * padded) as u32),
                            (
                                qp,
                                *q.device_ptr(),
                                *k.device_ptr(),
                                *v.device_ptr(),
                                n as i32,
                                self.heads as i32,
                                (self.hidden / self.heads) as i32,
                                padded as i32,
                            ),
                        )
                        .map_err(|e| e.to_string())?;
                    let status = if compatible_sdpa {
                        let (acc,la)=split_buffers.as_ref().ok_or("missing split buffers")?;
                        let sm=self.device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).map_err(|e|e.to_string())? as usize;
                        let splits=qwen_vision_splits(self.heads*((n+63)/64),2*sm,(n+127)/128);
                        krasis_qwen_vision_sdpa(*q.device_ptr() as _,*k.device_ptr() as _,*v.device_ptr() as _,*output.device_ptr() as _,*lse.device_ptr() as _,*acc.device_ptr() as _,*la.device_ptr() as _,n as i32,self.heads as i32,72,splits as i32,stream as _)
                    } else { self.flash.ok_or("missing fast vision attention")?(
                        *q.device_ptr() as _,
                        *k.device_ptr() as _,
                        *v.device_ptr() as _,
                        *output.device_ptr() as _,
                        *lse.device_ptr() as _,
                        std::ptr::null(),
                        std::ptr::null(),
                        1,
                        n as i32,
                        n as i32,
                        self.heads as i32,
                        self.heads as i32,
                        padded as i32,
                        n as i32,
                        n as i32,
                        1.0 / ((self.hidden / self.heads) as f32).sqrt(),
                        0,
                        0,
                        stream as _,
                    ) };
                    if status != 0 {
                        return Err(format!("Qwen3.5 vision FlashAttention failed: {status}"));
                    }
                    self.device
                        .get_func("qwen35_vision", "qv_unpad")
                        .ok_or("unpad kernel")?
                        .launch(
                            LaunchConfig::for_num_elems((n * self.hidden) as u32),
                            (
                                *output.device_ptr(),
                                bp,
                                (n * self.hidden) as i32,
                                (self.hidden / self.heads) as i32,
                                padded as i32,
                            ),
                        )
                        .map_err(|e| e.to_string())?;
                } else {
                    self.device
                        .get_func("qwen35_vision", "qv_attention")
                        .ok_or("attention kernel")?
                        .launch(
                            LaunchConfig {
                                grid_dim: ((n * self.heads) as u32, 1, 1),
                                block_dim: (32, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                qp,
                                bp,
                                n as i32,
                                self.heads as i32,
                                (self.hidden / self.heads) as i32,
                            ),
                        )
                        .map_err(|e| e.to_string())?;
                }
            }
            if i == 0 {
                if let Some(c) = capture.as_deref_mut() {
                    c.insert(
                        "qkv_rope".into(),
                        self.device
                            .dtoh_sync_copy(&qkv)
                            .map_err(|e| e.to_string())?,
                    );
                    c.insert(
                        "attn_core".into(),
                        self.device.dtoh_sync_copy(&b).map_err(|e| e.to_string())?,
                    );
                }
            }
            self.record(&mut capture, format!("{i}.attn_core"), &b)?;
            self.record(&mut capture, format!("{i}.qkv_rope"), &qkv)?;
            self.linear(bp, ap, n, &format!("{p}.attn.proj"), 0, &tmp)?;
            self.record(&mut capture, format!("{i}.attn_proj"), &a)?;
            self.add(xp, ap, n * self.hidden)?;
            self.record(&mut capture, format!("{i}.post_attn"), &x)?;
            self.norm(xp, ap, n, &format!("{p}.norm2"))?;
            self.record(&mut capture, format!("{i}.norm2"), &a)?;
            self.linear(ap, mp, n, &format!("{p}.mlp.linear_fc1"), 1, &tmp)?;
            self.record(&mut capture, format!("{i}.fc1_act"), &mid)?;
            self.linear(mp, ap, n, &format!("{p}.mlp.linear_fc2"), 0, &tmp)?;
            self.record(&mut capture, format!("{i}.fc2"), &a)?;
            self.add(xp, ap, n * self.hidden)?;
            if let Some(c) = capture.as_deref_mut() {
                c.insert(
                    format!("block_{i}"),
                    self.device.dtoh_sync_copy(&x).map_err(|e| e.to_string())?,
                );
            }
        }
        if replay.is_some_and(|i|i < self.depth) {
            unsafe { cudarc::driver::sys::lib().cuMemcpyDtoDAsync_v2(out,xp,n*self.hidden*2,stream).result().map_err(|e|e.to_string())?; }
            return self.device.synchronize().map_err(|e|e.to_string());
        }
        self.bank.stage(self.depth + 1, stream)?;
        self.norm(xp, ap, n, "merger.norm")?;
        self.record(&mut capture, "merger_norm".into(), &a)?;
        self.linear(ap, mergep, n / 4, "merger.linear_fc1", 2, &tmp)?;
        self.linear(mergep, out, n / 4, "merger.linear_fc2", 0, &tmp)?;
        self.device.synchronize().map_err(|e| e.to_string())?;
        Ok(())
    }
}

/// Place authentic image features at image-pad IDs and copy only requested
/// embedding rows from immutable host storage. One synchronized native handoff.
#[pyfunction]
pub(crate) fn assemble_qwen35_image(
    ordinal: usize,
    embedding_ptr: usize,
    vocab: usize,
    hidden: usize,
    ids: Vec<usize>,
    image_id: usize,
    features_ptr: usize,
    feature_rows: usize,
    output_ptr: usize,
) -> PyResult<()> {
    use cudarc::driver::sys;
    if embedding_ptr == 0
        || features_ptr == 0
        || output_ptr == 0
        || hidden == 0
        || ids.iter().any(|&i| i >= vocab)
        || ids.iter().filter(|&&i| i == image_id).count() != feature_rows
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "image embedding geometry/token mismatch",
        ));
    }
    let device = CudaDevice::new_with_stream(ordinal).map_err(pyerr)?;
    let bytes = hidden
        .checked_mul(2)
        .ok_or_else(|| pyerr("embedding extent overflow"))?;
    let mut image_row = 0;
    for (row, &id) in ids.iter().enumerate() {
        let status = unsafe {
            if id == image_id {
                let src = features_ptr
                    .checked_add(image_row * bytes)
                    .ok_or_else(|| pyerr("image extent overflow"))?;
                image_row += 1;
                sys::lib().cuMemcpyDtoDAsync_v2(
                    (output_ptr + row * bytes) as u64,
                    src as u64,
                    bytes,
                    *device.cu_stream(),
                )
            } else {
                sys::lib().cuMemcpyHtoDAsync_v2(
                    (output_ptr + row * bytes) as u64,
                    (embedding_ptr + id * bytes) as *const _,
                    bytes,
                    *device.cu_stream(),
                )
            }
        };
        if status != sys::CUresult::CUDA_SUCCESS {
            let _ = device.synchronize();
            return Err(pyerr(format!("image embedding placement: {status:?}")));
        }
    }
    device.synchronize().map_err(pyerr)
}

include!("qwen35_vision_lt.rs");

extern "C" {fn krasis_qwen_vision_sdpa(q:*mut std::ffi::c_void,k:*mut std::ffi::c_void,v:*mut std::ffi::c_void,y:*mut std::ffi::c_void,lse:*mut std::ffi::c_void,acc:*mut std::ffi::c_void,la:*mut std::ffi::c_void,n:i32,heads:i32,dim:i32,splits:i32,stream:*mut std::ffi::c_void)->i32;}
// Reference occupancy heuristic, bounded by eight KV blocks at 1024 patches.
fn qwen_vision_splits(blocks:usize,sms:usize,key_blocks:usize)->usize {
    if blocks as f32 >= 0.8*sms as f32 {return 1;}
    let max=8.min(sms).min(key_blocks);let mut efficiencies=Vec::new();let mut best=0f32;
    for s in 1..=max {
        let eligible=s==1 || (key_blocks+s-1)/s != (key_blocks+s-2)/(s-1);
        let waves=(blocks*s) as f32/sms as f32;
        let eff=if eligible {waves/waves.ceil()} else {0.};best=best.max(eff);efficiencies.push(eff);
    }
    efficiencies.iter().position(|&x|x>=0.85*best).map_or(1,|i|i+1)
}
