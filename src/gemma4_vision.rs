//! Image-only Gemma encoder: immutable packed host files, one layer slab,
//! bounded BF16 dequantisation scratch, native scheduling and CUDA execution.
use crate::dense_stream::DenseStream;
use cudarc::{
    cublas::{sys as bs, CudaBlas},
    driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig},
};
use pyo3::prelude::*;
use std::{collections::HashMap, sync::Arc};
struct Weight {
    ptr: u64,
    rows: usize,
    cols: usize,
    packed: bool,
}
#[pyclass]
pub struct Gemma4Vision {
    device: Arc<CudaDevice>,
    blas: CudaBlas,
    bank: Arc<DenseStream>,
    weights: HashMap<String, Weight>,
    depth: usize,
    max_patches: usize,
}
fn err(e: impl ToString) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}
#[pymethods]
impl Gemma4Vision {
    #[new]
    fn new(
        entries: Vec<(String, String, usize, usize, bool, usize)>,
        ordinal: usize,
        max_patches: usize,
    ) -> PyResult<Self> {
        if max_patches == 0 || max_patches > 2520 {
            return Err(err("Gemma patch limit must be 1..2520"));
        }
        let device = CudaDevice::new_with_stream(ordinal).map_err(err)?;
        device
            .load_ptx(
                cudarc::nvrtc::Ptx::from_src(include_str!(concat!(
                    env!("OUT_DIR"),
                    "/gemma4_vision.ptx"
                ))),
                "gemma4_vision",
                &[
                    "gv_unpack",
                    "gv_rms",
                    "gv_pixels",
                    "gv_position",
                    "gv_rope",
                    "gv_gate",
                    "gv_add",
                    "gv_pool",
                    "gv_pad",
                    "gv_assemble",
                ],
            )
            .map_err(err)?;
        let mut extents = HashMap::<usize, usize>::new();
        for (_, _, r, c, p, g) in &entries {
            let b = if *p {
                r * ((c + 127) / 128 * 64) + r * ((c + 127) / 128) * 2
            } else {
                r * c * 2
            };
            let v = extents.entry(*g).or_default();
            *v = (*v + 255) & !255;
            *v += b;
        }
        let bank = DenseStream::new(
            device.clone(),
            *extents
                .values()
                .max()
                .ok_or_else(|| err("empty vision weights"))?,
            0,
        )
        .map_err(err)?;
        let mut weights = HashMap::new();
        for (name, path, rows, cols, packed, group) in entries {
            let bytes = if packed {
                rows * ((cols + 127) / 128 * 64) + rows * ((cols + 127) / 128) * 2
            } else {
                rows * cols * 2
            };
            let ptr = bank
                .register_file(group, &path, bytes, false)
                .map_err(err)?;
            weights.insert(
                name,
                Weight {
                    ptr,
                    rows,
                    cols,
                    packed,
                },
            );
        }
        bank.seal().map_err(err)?;
        let blas = CudaBlas::new(device.clone()).map_err(err)?;
        Ok(Self {
            device,
            blas,
            bank,
            weights,
            depth: 27,
            max_patches,
        })
    }
    fn assemble(
        &self,
        embedding_ptr: usize,
        vocab: usize,
        hidden: usize,
        ids: Vec<usize>,
        image_id: usize,
        features_ptr: usize,
        feature_rows: usize,
        output_ptr: usize,
        scale: f32,
    ) -> PyResult<()> {
        if embedding_ptr == 0
            || features_ptr == 0
            || output_ptr == 0
            || hidden != 2816
            || ids.is_empty()
            || ids.iter().any(|&v| v >= vocab)
            || ids.iter().filter(|&&v| v == image_id).count() != feature_rows
            || !scale.is_finite()
        {
            return Err(err("Gemma embedding assembly contract"));
        }
        self.device.bind_to_thread().map_err(err)?;
        let mut host = vec![0u16; ids.len() * hidden];
        let mut map = Vec::<i32>::with_capacity(ids.len());
        let mut image = 0i32;
        for (row, &id) in ids.iter().enumerate() {
            if id == image_id {
                map.push(image);
                image += 1;
            } else {
                map.push(-1);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (embedding_ptr as *const u16).add(id * hidden),
                        host.as_mut_ptr().add(row * hidden),
                        hidden,
                    );
                }
            }
        }
        let indices = self.device.htod_sync_copy(&map).map_err(err)?;
        let result = (|| {
            unsafe {
                let status = cudarc::driver::sys::lib().cuMemcpyHtoDAsync_v2(
                    output_ptr as u64,
                    host.as_ptr() as _,
                    host.len() * 2,
                    *self.device.cu_stream(),
                );
                if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(err(format!("Gemma embedding copy {status:?}")));
                }
                self.device
                    .get_func("gemma4_vision", "gv_assemble")
                    .unwrap()
                    .launch(
                        LaunchConfig::for_num_elems(host.len() as u32),
                        (
                            output_ptr as u64,
                            features_ptr as u64,
                            *indices.device_ptr(),
                            host.len() as i32,
                            hidden as i32,
                            scale,
                        ),
                    )
                    .map_err(err)?;
            }
            Ok(())
        })();
        // Host rows and device indices must outlive every queued operation,
        // including a failed launch after the asynchronous host copy.
        let completion = self.device.synchronize().map_err(err);
        result.and(completion)
    }
    fn projection_debug(
        &self,
        name: String,
        input_ptr: usize,
        output_ptr: usize,
        rows: usize,
    ) -> PyResult<()> {
        if rows == 0 || rows > self.max_patches || input_ptr == 0 || output_ptr == 0 {
            return Err(err("invalid projection replay"));
        }
        let group = if name.starts_with("encoder.layers.") {
            name.split('.')
                .nth(2)
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&v| v < 27)
                .ok_or_else(|| err("invalid layer"))?
                + 1
        } else if name.starts_with("embed_vision.") {
            28
        } else {
            0
        };
        self.device.bind_to_thread().map_err(err)?;
        self.bank
            .stage(group, *self.device.cu_stream())
            .map_err(err)?;
        let matrix = self.alloc(4304 * 1152).map_err(err)?;
        self.linear(input_ptr as u64, output_ptr as u64, rows, &name, &matrix)
            .map_err(err)?;
        self.device.synchronize().map_err(err)
    }
    fn memory_info(&self) -> PyResult<String> {
        self.bank.stats().map_err(err)
    }
    fn encode(&self, pixels_ptr: usize, gh: usize, gw: usize, output_ptr: usize) -> PyResult<()> {
        self.forward(pixels_ptr as u64, gh, gw, output_ptr as u64, None)
            .map_err(err)
    }
    fn encode_debug(
        &self,
        pixels_ptr: usize,
        gh: usize,
        gw: usize,
        output_ptr: usize,
    ) -> PyResult<HashMap<String, Vec<u16>>> {
        let mut capture = HashMap::new();
        self.forward(
            pixels_ptr as u64,
            gh,
            gw,
            output_ptr as u64,
            Some(&mut capture),
        )
        .map_err(err)?;
        Ok(capture)
    }
}
impl Gemma4Vision {
    fn alloc(&self, n: usize) -> Result<CudaSlice<u16>, String> {
        self.device.alloc_zeros(n).map_err(|e| e.to_string())
    }
    fn w(&self, name: &str) -> Result<&Weight, String> {
        self.weights
            .get(name)
            .ok_or_else(|| format!("missing Gemma weight {name}"))
    }
    fn rms(
        &self,
        x: u64,
        y: u64,
        rows: usize,
        cols: usize,
        name: Option<&str>,
    ) -> Result<(), String> {
        let w = match name {
            Some(s) => self.w(s)?.ptr,
            None => 0,
        };
        unsafe {
            self.device
                .get_func("gemma4_vision", "gv_rms")
                .unwrap()
                .launch(
                    LaunchConfig {
                        grid_dim: (rows as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (x, w, y, cols as i32),
                )
                .map_err(|e| e.to_string())
        }
    }
    fn linear(
        &self,
        x: u64,
        y: u64,
        m: usize,
        name: &str,
        scratch: &CudaSlice<u16>,
    ) -> Result<(), String> {
        let w = self.w(name)?;
        if !w.packed || w.rows * w.cols > scratch.len() {
            return Err(format!("invalid Gemma projection {name}"));
        }
        let packed_bytes = w.rows * ((w.cols + 127) / 128 * 64);
        unsafe {
            self.device
                .get_func("gemma4_vision", "gv_unpack")
                .unwrap()
                .launch(
                    LaunchConfig::for_num_elems((w.rows * w.cols) as u32),
                    (
                        w.ptr,
                        w.ptr + packed_bytes as u64,
                        *scratch.device_ptr(),
                        w.rows as i32,
                        w.cols as i32,
                    ),
                )
                .map_err(|e| e.to_string())?;
            let alpha = 1f32;
            let beta = 0f32;
            let status = bs::lib().cublasGemmEx(
                *self.blas.handle(),
                bs::cublasOperation_t::CUBLAS_OP_T,
                bs::cublasOperation_t::CUBLAS_OP_N,
                w.rows as i32,
                m as i32,
                w.cols as i32,
                &alpha as *const _ as _,
                *scratch.device_ptr() as _,
                bs::cudaDataType::CUDA_R_16BF,
                w.cols as i32,
                x as _,
                bs::cudaDataType::CUDA_R_16BF,
                w.cols as i32,
                &beta as *const _ as _,
                y as _,
                bs::cudaDataType::CUDA_R_16BF,
                w.rows as i32,
                bs::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                bs::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            );
            if status != bs::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(format!("Gemma GEMM {name}: {status:?}"));
            }
        }
        Ok(())
    }
    fn add(&self, x: u64, y: u64, n: usize) -> Result<(), String> {
        unsafe {
            self.device
                .get_func("gemma4_vision", "gv_add")
                .unwrap()
                .launch(LaunchConfig::for_num_elems(n as u32), (x, y, n as i32))
                .map_err(|e| e.to_string())
        }
    }
    fn record(
        &self,
        capture: &mut Option<&mut HashMap<String, Vec<u16>>>,
        name: String,
        x: &CudaSlice<u16>,
    ) -> Result<(), String> {
        if let Some(c) = capture {
            c.insert(
                name,
                self.device.dtoh_sync_copy(x).map_err(|e| e.to_string())?,
            );
        }
        Ok(())
    }
    fn record_row(
        &self,
        capture: &mut Option<&mut HashMap<String, Vec<u16>>>,
        name: String,
        ptr: u64,
        width: usize,
    ) -> Result<(), String> {
        if let Some(c) = capture {
            self.device.synchronize().map_err(|e| e.to_string())?;
            let mut value = vec![0u16; width];
            unsafe {
                let status = cudarc::driver::sys::lib().cuMemcpyDtoH_v2(
                    value.as_mut_ptr() as _,
                    ptr,
                    width * 2,
                );
                if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("vision capture {status:?}"));
                }
            }
            c.insert(name, value);
        }
        Ok(())
    }
    fn forward(
        &self,
        pixels: u64,
        gh: usize,
        gw: usize,
        out: u64,
        mut capture: Option<&mut HashMap<String, Vec<u16>>>,
    ) -> Result<(), String> {
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        let n = gh.checked_mul(gw).ok_or("patch overflow")?;
        if n == 0
            || n > self.max_patches
            || gh % 3 != 0
            || gw % 3 != 0
            || gh > 10240
            || gw > 10240
            || pixels == 0
            || out == 0
        {
            return Err("unsupported Gemma image geometry".into());
        }
        let h = 1152;
        let inter = 4304;
        let heads = 16;
        let dim = 72;
        let peak = n * (h * 7 + inter * 2) + 4304 * 1152 + 16 * n * 96 + 16 * n * 4;
        let (free, _) = cudarc::driver::result::mem_get_info().map_err(|e| e.to_string())?;
        if free < peak * 2 + (600 << 20) {
            return Err("Gemma vision would breach 600 MiB reserve".into());
        }
        let x = self.alloc(n * h)?;
        let norm = self.alloc(n * h)?;
        let q = self.alloc(n * h)?;
        let k = self.alloc(n * h)?;
        let v = self.alloc(n * h)?;
        let y = self.alloc(n * h)?;
        let z = self.alloc(n * h)?;
        let gate = self.alloc(n * inter)?;
        let up = self.alloc(n * inter)?;
        let matrix = self.alloc(inter * h)?;
        let pixel = self.alloc(n * 768)?;
        let pooled = self.alloc(n / 9 * h)?;
        self.bank.stage(0, *self.device.cu_stream())?;
        unsafe {
            self.device
                .get_func("gemma4_vision", "gv_pixels")
                .unwrap()
                .launch(
                    LaunchConfig::for_num_elems((n * 768) as u32),
                    (pixels, *pixel.device_ptr(), (n * 768) as i32),
                )
                .map_err(|e| e.to_string())?;
        }
        self.linear(
            *pixel.device_ptr(),
            *x.device_ptr(),
            n,
            "patch_embedder.input_proj.weight",
            &matrix,
        )?;
        unsafe {
            self.device
                .get_func("gemma4_vision", "gv_position")
                .unwrap()
                .launch(
                    LaunchConfig::for_num_elems((n * h) as u32),
                    (
                        *x.device_ptr(),
                        self.w("patch_embedder.position_embedding_table")?.ptr,
                        n as i32,
                        h as i32,
                        gw as i32,
                        10240i32,
                    ),
                )
                .map_err(|e| e.to_string())?;
        }
        self.record(&mut capture, "patch".into(), &x)?;
        // SDPA wrapper accepts padded 96-wide query/key/value rows; all valid rows
        // are contiguous, so padding keys never enter the attention domain.
        let qp = self.alloc(n * heads * 96)?;
        let kp = self.alloc(n * heads * 96)?;
        let vp = self.alloc(n * heads * 96)?;
        let yp = self.alloc(n * heads * 96)?;
        let lse = self
            .device
            .alloc_zeros::<f32>(heads * n)
            .map_err(|e| e.to_string())?;
        for i in 0..self.depth {
            self.bank.stage(i + 1, *self.device.cu_stream())?;
            let p = format!("encoder.layers.{i}");
            self.rms(
                *x.device_ptr(),
                *norm.device_ptr(),
                n,
                h,
                Some(&format!("{p}.input_layernorm.weight")),
            )?;
            self.record_row(
                &mut capture,
                format!("layer{i}.attn_input"),
                *norm.device_ptr(),
                h,
            )?;
            for (name, target, scaled) in [("q", &q, true), ("k", &k, true), ("v", &v, false)] {
                self.linear(
                    *norm.device_ptr(),
                    *target.device_ptr(),
                    n,
                    &format!("{p}.self_attn.{name}_proj.linear.weight"),
                    &matrix,
                )?;
                let norm_name = format!("{p}.self_attn.{name}_norm.weight");
                if i == 0 {
                    self.record(&mut capture, format!("layer0.{name}_raw"), target)?;
                }
                self.rms(
                    *target.device_ptr(),
                    *target.device_ptr(),
                    n * heads,
                    dim,
                    if scaled {
                        Some(norm_name.as_str())
                    } else {
                        None
                    },
                )?;
                if i == 0 {
                    self.record(&mut capture, format!("layer0.{name}_norm"), target)?;
                }
                if scaled {
                    unsafe {
                        self.device
                            .get_func("gemma4_vision", "gv_rope")
                            .unwrap()
                            .launch(
                                LaunchConfig::for_num_elems((n * heads * (dim / 2)) as u32),
                                (
                                    *target.device_ptr(),
                                    n as i32,
                                    heads as i32,
                                    dim as i32,
                                    gw as i32,
                                ),
                            )
                            .map_err(|e| e.to_string())?;
                    }
                }
            }
            unsafe {
                gv_pad(
                    self.device.clone(),
                    *q.device_ptr(),
                    *qp.device_ptr(),
                    n,
                    heads,
                    dim,
                    false,
                )?;
                gv_pad(
                    self.device.clone(),
                    *k.device_ptr(),
                    *kp.device_ptr(),
                    n,
                    heads,
                    dim,
                    false,
                )?;
                gv_pad(
                    self.device.clone(),
                    *v.device_ptr(),
                    *vp.device_ptr(),
                    n,
                    heads,
                    dim,
                    false,
                )?;
                let code = krasis_gemma_vision_sdpa(
                    *qp.device_ptr() as _,
                    *kp.device_ptr() as _,
                    *vp.device_ptr() as _,
                    *yp.device_ptr() as _,
                    *lse.device_ptr() as _,
                    n as i32,
                    heads as i32,
                    *self.device.cu_stream() as _,
                );
                if code != 0 {
                    return Err(format!("Gemma SDPA {code}"));
                }
                gv_pad(
                    self.device.clone(),
                    *yp.device_ptr(),
                    *y.device_ptr(),
                    n,
                    heads,
                    dim,
                    true,
                )?;
            }
            if i == 0 {
                self.record(&mut capture, "layer0.attn".into(), &y)?;
            }
            self.record_row(
                &mut capture,
                format!("layer{i}.o_input"),
                *y.device_ptr(),
                h,
            )?;
            self.linear(
                *y.device_ptr(),
                *z.device_ptr(),
                n,
                &format!("{p}.self_attn.o_proj.linear.weight"),
                &matrix,
            )?;
            self.rms(
                *z.device_ptr(),
                *z.device_ptr(),
                n,
                h,
                Some(&format!("{p}.post_attention_layernorm.weight")),
            )?;
            self.add(*x.device_ptr(), *z.device_ptr(), n * h)?;
            self.rms(
                *x.device_ptr(),
                *norm.device_ptr(),
                n,
                h,
                Some(&format!("{p}.pre_feedforward_layernorm.weight")),
            )?;
            self.record_row(
                &mut capture,
                format!("layer{i}.mlp_input"),
                *norm.device_ptr(),
                h,
            )?;
            self.linear(
                *norm.device_ptr(),
                *gate.device_ptr(),
                n,
                &format!("{p}.mlp.gate_proj.linear.weight"),
                &matrix,
            )?;
            self.linear(
                *norm.device_ptr(),
                *up.device_ptr(),
                n,
                &format!("{p}.mlp.up_proj.linear.weight"),
                &matrix,
            )?;
            unsafe {
                self.device
                    .get_func("gemma4_vision", "gv_gate")
                    .unwrap()
                    .launch(
                        LaunchConfig::for_num_elems((n * inter) as u32),
                        (*gate.device_ptr(), *up.device_ptr(), (n * inter) as i32),
                    )
                    .map_err(|e| e.to_string())?;
            }
            self.record_row(
                &mut capture,
                format!("layer{i}.down_input"),
                *gate.device_ptr(),
                inter,
            )?;
            self.linear(
                *gate.device_ptr(),
                *z.device_ptr(),
                n,
                &format!("{p}.mlp.down_proj.linear.weight"),
                &matrix,
            )?;
            self.rms(
                *z.device_ptr(),
                *z.device_ptr(),
                n,
                h,
                Some(&format!("{p}.post_feedforward_layernorm.weight")),
            )?;
            self.add(*x.device_ptr(), *z.device_ptr(), n * h)?;
            self.record(&mut capture, format!("layer{i}"), &x)?;
        }
        self.bank.stage(self.depth + 1, *self.device.cu_stream())?;
        unsafe {
            self.device
                .get_func("gemma4_vision", "gv_pool")
                .unwrap()
                .launch(
                    LaunchConfig::for_num_elems((n / 9 * h) as u32),
                    (
                        *x.device_ptr(),
                        self.w("std_bias")?.ptr,
                        self.w("std_scale")?.ptr,
                        *pooled.device_ptr(),
                        (n / 9) as i32,
                        h as i32,
                        gh as i32,
                        gw as i32,
                    ),
                )
                .map_err(|e| e.to_string())?;
        }
        self.record(&mut capture, "pooled".into(), &pooled)?;
        self.rms(*pooled.device_ptr(), *pooled.device_ptr(), n / 9, h, None)?;
        self.record_row(
            &mut capture,
            "projector_input".into(),
            *pooled.device_ptr(),
            h,
        )?;
        self.linear(
            *pooled.device_ptr(),
            out,
            n / 9,
            "embed_vision.embedding_projection.weight",
            &matrix,
        )?;
        self.device.synchronize().map_err(|e| e.to_string())
    }
}
unsafe fn gv_pad(
    device: Arc<CudaDevice>,
    input: u64,
    output: u64,
    n: usize,
    heads: usize,
    dim: usize,
    reverse: bool,
) -> Result<(), String> {
    device
        .get_func("gemma4_vision", "gv_pad")
        .unwrap()
        .launch(
            LaunchConfig::for_num_elems((n * heads * 96) as u32),
            (
                input,
                output,
                (n * heads) as i32,
                dim as i32,
                if reverse { 1i32 } else { 0i32 },
            ),
        )
        .map_err(|e| e.to_string())
}
extern "C" {
    fn krasis_gemma_vision_sdpa(
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        y: *mut std::ffi::c_void,
        lse: *mut std::ffi::c_void,
        n: i32,
        heads: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}
