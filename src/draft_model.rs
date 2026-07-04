//! Draft model for speculative decoding.
//!
//! Loads a small dense model (e.g. Qwen3-1.7B) into VRAM with INT4 quantized
//! layer weights and BF16 embedding. Generates draft tokens at HBM speed,
//! which are verified by the target model alongside its normal decode step.
//!
//! Weight quantization: RTN (round-to-nearest) INT4 with group_size=128.
//! Each group of 128 BF16 values is quantized to 4-bit unsigned (0..15, stored
//! as packed u8 pairs) with one FP32 scale per group.
//!
//! The draft model reuses existing CUDA kernels from decode_kernels.cu:
//! embedding_lookup, fused_add_rmsnorm, per_head_rmsnorm, apply_rope,
//! kv_cache_write, gqa_attention, silu_mul, fp32_to_bf16,
//! simple_int4_gemv_f32, simple_int4_gemv_bf16.

use cudarc::cublas::result as cublas_result;
use cudarc::cublas::{sys as cublas_sys, CudaBlas};
use cudarc::driver::sys as cuda_sys;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use crate::weights::safetensors_io::MmapSafetensors;

const MODULE_NAME: &str = "decode_kernels";
const INT4_GROUP_SIZE: usize = 128;

/// An INT4 quantized weight matrix in VRAM.
/// Packed format: each u8 holds 2 INT4 values (low nibble = even col, high nibble = odd col).
/// Unsigned 0..15, kernel subtracts 8 for signed -8..+7.
struct DraftWeightInt4 {
    d_packed: CudaSlice<u8>,  // [rows, cols/2] packed INT4
    d_scales: CudaSlice<f32>, // [rows, cols/group_size] FP32
    rows: usize,
    cols: usize,
    group_size: usize,
}

impl DraftWeightInt4 {
    fn packed_ptr(&self) -> u64 {
        *self.d_packed.device_ptr()
    }
    fn scales_ptr(&self) -> u64 {
        *self.d_scales.device_ptr()
    }
}

/// Per-layer weights for the draft model.
struct DraftLayer {
    // Attention (INT4)
    q_proj: DraftWeightInt4,
    k_proj: DraftWeightInt4,
    v_proj: DraftWeightInt4,
    o_proj: DraftWeightInt4,
    d_q_norm: CudaSlice<f32>, // [head_dim] FP32 (converted from BF16 at load)
    d_k_norm: CudaSlice<f32>, // [head_dim] FP32 (converted from BF16 at load)

    // MLP (INT4, dense not MoE)
    gate_proj: DraftWeightInt4,
    up_proj: DraftWeightInt4,
    down_proj: DraftWeightInt4,

    // Layer norms (BF16)
    d_input_norm: CudaSlice<u16>,     // [hidden_size] BF16
    d_post_attn_norm: CudaSlice<u16>, // [hidden_size] BF16
}

/// Configuration for the draft model.
pub struct DraftConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub eps: f32,
    pub rope_theta: f64,
    pub max_seq: usize,
}

/// A small dense model for generating draft tokens in speculative decode.
pub struct DraftModel {
    cfg: DraftConfig,

    // Layers
    layers: Vec<DraftLayer>,

    // Embedding (BF16), also used as LM head for tied-weight models
    d_embedding: CudaSlice<u16>,
    d_final_norm: CudaSlice<u16>,

    // Scratch buffers (all sized for draft model dimensions)
    d_hidden: CudaSlice<u16>,   // [hidden_size] BF16
    d_residual: CudaSlice<u16>, // [hidden_size] BF16
    d_scratch: CudaSlice<u16>,  // [max(vocab_size, 2*intermediate_size)] BF16
    d_logits: CudaSlice<f32>,   // [vocab_size] FP32

    // GQA scratch (FP32)
    d_gqa_q: CudaSlice<f32>,   // [num_heads * head_dim]
    d_gqa_k: CudaSlice<f32>,   // [num_kv_heads * head_dim]
    d_gqa_v: CudaSlice<f32>,   // [num_kv_heads * head_dim]
    d_gqa_out: CudaSlice<f32>, // [num_heads * head_dim]

    // MLP scratch
    d_gate_up: CudaSlice<u16>,     // [2 * intermediate_size] BF16
    d_mlp_scratch: CudaSlice<u16>, // [intermediate_size] BF16

    // KV cache: FP8 E4M3 (reuses existing gqa_attention kernel)
    kv_k: Vec<CudaSlice<u8>>,
    kv_v: Vec<CudaSlice<u8>>,
    kv_pos: usize,

    // RoPE tables (FP32)
    d_rope_cos: CudaSlice<f32>,
    d_rope_sin: CudaSlice<f32>,

    // Host-side logits for sampling
    pub h_logits: Vec<f32>,

    // VRAM usage (bytes)
    pub vram_bytes: usize,
}

/// Quantize a BF16 weight matrix to INT4 on the host.
/// Returns (packed_u8, scales_f32).
fn quantize_bf16_to_int4(
    bf16_data: &[u16],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(bf16_data.len(), rows * cols);
    assert_eq!(
        cols % group_size,
        0,
        "cols {} must be divisible by group_size {}",
        cols,
        group_size
    );
    assert_eq!(cols % 2, 0, "cols must be even for INT4 packing");

    let n_groups_per_row = cols / group_size;
    let mut packed = vec![0u8; rows * cols / 2];
    let mut scales = vec![0.0f32; rows * n_groups_per_row];

    for r in 0..rows {
        let row_offset = r * cols;
        for g in 0..n_groups_per_row {
            let g_start = g * group_size;

            // Find max absolute value in this group
            let mut max_abs: f32 = 0.0;
            for j in 0..group_size {
                let bf16_val = bf16_data[row_offset + g_start + j];
                let f32_val = f32::from_bits((bf16_val as u32) << 16);
                let abs_val = f32_val.abs();
                if abs_val > max_abs {
                    max_abs = abs_val;
                }
            }

            // Scale: maps [-max_abs, +max_abs] to [-7, +7]
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            scales[r * n_groups_per_row + g] = scale;

            let inv_scale = 1.0 / scale;

            // Quantize and pack pairs
            for j in (0..group_size).step_by(2) {
                let idx = g_start + j;
                let v0 = f32::from_bits((bf16_data[row_offset + idx] as u32) << 16);
                let v1 = f32::from_bits((bf16_data[row_offset + idx + 1] as u32) << 16);

                let q0 = ((v0 * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;
                let q1 = ((v1 * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;

                packed[r * (cols / 2) + idx / 2] = q0 | (q1 << 4);
            }
        }
    }

    (packed, scales)
}

impl DraftModel {
    /// Load a Qwen3-style dense model from safetensors into VRAM.
    /// Layer weights quantized to INT4. Embedding stays BF16.
    pub fn load(device: &Arc<CudaDevice>, model_dir: &str, max_seq: usize) -> Result<Self, String> {
        use std::path::Path;

        let dir = Path::new(model_dir);
        let config_path = dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        let hidden_size = config["hidden_size"].as_u64().unwrap_or(0) as usize;
        let num_layers = config["num_hidden_layers"].as_u64().unwrap_or(0) as usize;
        let num_heads = config["num_attention_heads"].as_u64().unwrap_or(0) as usize;
        let num_kv_heads = config["num_key_value_heads"].as_u64().unwrap_or(0) as usize;
        let head_dim = config["head_dim"].as_u64().unwrap_or(128) as usize;
        let intermediate_size = config["intermediate_size"].as_u64().unwrap_or(0) as usize;
        let vocab_size = config["vocab_size"].as_u64().unwrap_or(0) as usize;
        let eps = config["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let rope_theta = config["rope_theta"].as_f64().unwrap_or(1e6);

        if hidden_size == 0 || num_layers == 0 || vocab_size == 0 {
            return Err("Invalid model config: zero dimensions".to_string());
        }

        log::info!("DraftModel: loading {} INT4 (hs={}, layers={}, heads={}, kv_heads={}, hd={}, inter={}, vocab={})",
            model_dir, hidden_size, num_layers, num_heads, num_kv_heads, head_dim, intermediate_size, vocab_size);

        let cfg = DraftConfig {
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            eps,
            rope_theta,
            max_seq,
        };

        // Open safetensors file(s) — supports both single and sharded models
        let single_path = dir.join("model.safetensors");
        let index_path = dir.join("model.safetensors.index.json");
        let shards: Vec<MmapSafetensors> = if single_path.exists() {
            vec![MmapSafetensors::open(&single_path)
                .map_err(|e| format!("Failed to open safetensors: {}", e))?]
        } else if index_path.exists() {
            let idx_str = std::fs::read_to_string(&index_path)
                .map_err(|e| format!("Failed to read index: {}", e))?;
            let idx: serde_json::Value = serde_json::from_str(&idx_str)
                .map_err(|e| format!("Failed to parse index: {}", e))?;
            let weight_map = idx["weight_map"]
                .as_object()
                .ok_or_else(|| "weight_map not found in index".to_string())?;
            let mut shard_files: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            shard_files.sort();
            shard_files.dedup();
            let mut shards = Vec::new();
            for sf in &shard_files {
                let sp = dir.join(sf);
                if !sp.exists() {
                    return Err(format!("Shard file {} not found", sf));
                }
                shards.push(
                    MmapSafetensors::open(&sp)
                        .map_err(|e| format!("Failed to open shard {}: {}", sf, e))?,
                );
            }
            log::info!("DraftModel: opened {} shards", shards.len());
            shards
        } else {
            return Err(format!(
                "No model.safetensors or index found in {}",
                model_dir
            ));
        };

        let mut vram_bytes: usize = 0;

        // Helper: find tensor across shards
        let find_tensor = |name: &str| -> Result<(&MmapSafetensors, usize), String> {
            for (i, shard) in shards.iter().enumerate() {
                if shard.tensor_info(name).is_some() {
                    return Ok((shard, i));
                }
            }
            Err(format!("tensor {} not found in any shard", name))
        };

        // Helper: read BF16 tensor data from safetensors
        let read_bf16 = |name: &str, expected_numel: usize| -> Result<Vec<u16>, String> {
            let (shard, _) = find_tensor(name)?;
            let data = shard
                .tensor_data(name)
                .map_err(|e| format!("tensor {}: {}", name, e))?;
            let info = shard
                .tensor_info(name)
                .ok_or_else(|| format!("tensor {} not found", name))?;
            let numel = info.numel();
            if numel != expected_numel {
                return Err(format!(
                    "tensor {}: expected {} elements, got {}",
                    name, expected_numel, numel
                ));
            }
            let slice = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, numel) };
            Ok(slice.to_vec())
        };

        // Helper: upload BF16 tensor to VRAM
        let upload_bf16 = |data: Vec<u16>| -> Result<CudaSlice<u16>, String> {
            device.htod_copy(data).map_err(|e| format!("htod: {:?}", e))
        };

        // Helper: read BF16 weight, quantize to INT4, upload packed + scales
        let upload_weight_int4 =
            |name: &str, rows: usize, cols: usize| -> Result<DraftWeightInt4, String> {
                let bf16_data = read_bf16(name, rows * cols)?;
                let (packed, scales) =
                    quantize_bf16_to_int4(&bf16_data, rows, cols, INT4_GROUP_SIZE);

                let d_packed = device
                    .htod_copy(packed)
                    .map_err(|e| format!("htod packed {}: {:?}", name, e))?;
                let d_scales = device
                    .htod_copy(scales)
                    .map_err(|e| format!("htod scales {}: {:?}", name, e))?;

                Ok(DraftWeightInt4 {
                    d_packed,
                    d_scales,
                    rows,
                    cols,
                    group_size: INT4_GROUP_SIZE,
                })
            };

        // Helper: read BF16, convert to FP32, upload
        let upload_bf16_as_f32 =
            |name: &str, expected_numel: usize| -> Result<CudaSlice<f32>, String> {
                let bf16_data = read_bf16(name, expected_numel)?;
                let f32_data: Vec<f32> = bf16_data
                    .iter()
                    .map(|&v| f32::from_bits((v as u32) << 16))
                    .collect();
                device
                    .htod_copy(f32_data)
                    .map_err(|e| format!("htod {}: {:?}", name, e))
            };

        // ── Load embedding (BF16) ──
        let emb_data = read_bf16("model.embed_tokens.weight", vocab_size * hidden_size)?;
        let d_embedding = upload_bf16(emb_data)?;
        vram_bytes += vocab_size * hidden_size * 2;
        log::info!(
            "DraftModel: embedding loaded ({:.1} MB BF16)",
            vocab_size as f64 * hidden_size as f64 * 2.0 / 1e6
        );

        // ── Load final norm (BF16) ──
        let norm_data = read_bf16("model.norm.weight", hidden_size)?;
        let d_final_norm = upload_bf16(norm_data)?;
        vram_bytes += hidden_size * 2;

        // ── Load layers (INT4 quantized) ──
        let kv_stride = num_kv_heads * head_dim;
        let q_size = num_heads * head_dim;
        let mut layers = Vec::with_capacity(num_layers);
        let n_groups_hs = hidden_size / INT4_GROUP_SIZE;
        let n_groups_inter = intermediate_size / INT4_GROUP_SIZE;

        for i in 0..num_layers {
            let pfx = format!("model.layers.{}", i);

            let q_proj = upload_weight_int4(
                &format!("{}.self_attn.q_proj.weight", pfx),
                q_size,
                hidden_size,
            )?;
            let k_proj = upload_weight_int4(
                &format!("{}.self_attn.k_proj.weight", pfx),
                kv_stride,
                hidden_size,
            )?;
            let v_proj = upload_weight_int4(
                &format!("{}.self_attn.v_proj.weight", pfx),
                kv_stride,
                hidden_size,
            )?;
            let o_proj = upload_weight_int4(
                &format!("{}.self_attn.o_proj.weight", pfx),
                hidden_size,
                q_size,
            )?;

            let d_q_norm =
                upload_bf16_as_f32(&format!("{}.self_attn.q_norm.weight", pfx), head_dim)?;
            let d_k_norm =
                upload_bf16_as_f32(&format!("{}.self_attn.k_norm.weight", pfx), head_dim)?;

            let gate_proj = upload_weight_int4(
                &format!("{}.mlp.gate_proj.weight", pfx),
                intermediate_size,
                hidden_size,
            )?;
            let up_proj = upload_weight_int4(
                &format!("{}.mlp.up_proj.weight", pfx),
                intermediate_size,
                hidden_size,
            )?;
            let down_proj = upload_weight_int4(
                &format!("{}.mlp.down_proj.weight", pfx),
                hidden_size,
                intermediate_size,
            )?;

            let in_norm_data = read_bf16(&format!("{}.input_layernorm.weight", pfx), hidden_size)?;
            let d_input_norm = upload_bf16(in_norm_data)?;
            let pa_norm_data = read_bf16(
                &format!("{}.post_attention_layernorm.weight", pfx),
                hidden_size,
            )?;
            let d_post_attn_norm = upload_bf16(pa_norm_data)?;

            // VRAM per layer: packed weights (cols/2 bytes) + scales (rows * n_groups * 4) + norms
            let attn_packed =
                (q_size + kv_stride * 2 + hidden_size) * hidden_size / 2 + hidden_size * q_size / 2;
            let attn_scales = (q_size * n_groups_hs
                + kv_stride * 2 * n_groups_hs
                + hidden_size * (q_size / INT4_GROUP_SIZE))
                * 4;
            let mlp_packed =
                (intermediate_size * 2) * hidden_size / 2 + hidden_size * intermediate_size / 2;
            let mlp_scales =
                (intermediate_size * 2 * n_groups_hs + hidden_size * n_groups_inter) * 4;
            let norms_bytes = hidden_size * 2 * 2 + head_dim * 4 * 2;
            vram_bytes += attn_packed + attn_scales + mlp_packed + mlp_scales + norms_bytes;

            layers.push(DraftLayer {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                d_q_norm,
                d_k_norm,
                gate_proj,
                up_proj,
                down_proj,
                d_input_norm,
                d_post_attn_norm,
            });

            if i == 0 || i == num_layers - 1 {
                log::info!("DraftModel: layer {} loaded (INT4)", i);
            }
        }
        log::info!(
            "DraftModel: all {} layers loaded ({:.1} MB INT4 weights)",
            num_layers,
            vram_bytes as f64 / 1e6
        );

        // ── Allocate scratch buffers ──
        let max_scratch = vocab_size.max(intermediate_size * 2).max(q_size);
        let d_hidden = device
            .alloc_zeros::<u16>(hidden_size)
            .map_err(|e| format!("alloc d_hidden: {:?}", e))?;
        let d_residual = device
            .alloc_zeros::<u16>(hidden_size)
            .map_err(|e| format!("alloc d_residual: {:?}", e))?;
        let d_scratch = device
            .alloc_zeros::<u16>(max_scratch)
            .map_err(|e| format!("alloc d_scratch: {:?}", e))?;
        let d_logits = device
            .alloc_zeros::<f32>(vocab_size)
            .map_err(|e| format!("alloc d_logits: {:?}", e))?;
        let d_gqa_q = device
            .alloc_zeros::<f32>(q_size)
            .map_err(|e| format!("alloc d_gqa_q: {:?}", e))?;
        let d_gqa_k = device
            .alloc_zeros::<f32>(kv_stride)
            .map_err(|e| format!("alloc d_gqa_k: {:?}", e))?;
        let d_gqa_v = device
            .alloc_zeros::<f32>(kv_stride)
            .map_err(|e| format!("alloc d_gqa_v: {:?}", e))?;
        let d_gqa_out = device
            .alloc_zeros::<f32>(q_size)
            .map_err(|e| format!("alloc d_gqa_out: {:?}", e))?;
        let d_gate_up = device
            .alloc_zeros::<u16>(intermediate_size * 2)
            .map_err(|e| format!("alloc d_gate_up: {:?}", e))?;
        let d_mlp_scratch = device
            .alloc_zeros::<u16>(intermediate_size)
            .map_err(|e| format!("alloc d_mlp_scratch: {:?}", e))?;

        vram_bytes += (hidden_size * 2
            + hidden_size * 2
            + max_scratch * 2
            + vocab_size * 4
            + q_size * 4
            + kv_stride * 4 * 2
            + q_size * 4
            + intermediate_size * 2 * 2
            + intermediate_size * 2) as usize;

        // ── Allocate KV cache (FP8 E4M3) ──
        let kv_size_per_layer = max_seq * kv_stride;
        let mut kv_k = Vec::with_capacity(num_layers);
        let mut kv_v = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            kv_k.push(
                device
                    .alloc_zeros::<u8>(kv_size_per_layer)
                    .map_err(|e| format!("alloc kv_k: {:?}", e))?,
            );
            kv_v.push(
                device
                    .alloc_zeros::<u8>(kv_size_per_layer)
                    .map_err(|e| format!("alloc kv_v: {:?}", e))?,
            );
        }
        vram_bytes += num_layers * kv_size_per_layer * 2;
        log::info!(
            "DraftModel: KV cache allocated ({:.1} MB, max_seq={})",
            (num_layers * kv_size_per_layer * 2) as f64 / 1e6,
            max_seq
        );

        // ── Compute RoPE tables ──
        let half_dim = head_dim / 2;
        let rope_total = max_seq * half_dim;
        let mut cos_table = vec![0.0f32; rope_total];
        let mut sin_table = vec![0.0f32; rope_total];

        for pos in 0..max_seq {
            for i in 0..half_dim {
                let freq = 1.0 / (rope_theta as f64).powf(2.0 * i as f64 / head_dim as f64);
                let angle = pos as f64 * freq;
                cos_table[pos * half_dim + i] = angle.cos() as f32;
                sin_table[pos * half_dim + i] = angle.sin() as f32;
            }
        }

        let d_rope_cos = device
            .htod_copy(cos_table)
            .map_err(|e| format!("htod rope_cos: {:?}", e))?;
        let d_rope_sin = device
            .htod_copy(sin_table)
            .map_err(|e| format!("htod rope_sin: {:?}", e))?;
        vram_bytes += rope_total * 4 * 2;

        let h_logits = vec![0.0f32; vocab_size];

        log::info!(
            "DraftModel: fully loaded — {:.1} MB VRAM total",
            vram_bytes as f64 / 1e6
        );

        Ok(DraftModel {
            cfg,
            layers,
            d_embedding,
            d_final_norm,
            d_hidden,
            d_residual,
            d_scratch,
            d_logits,
            d_gqa_q,
            d_gqa_k,
            d_gqa_v,
            d_gqa_out,
            d_gate_up,
            d_mlp_scratch,
            kv_k,
            kv_v,
            kv_pos: 0,
            d_rope_cos,
            d_rope_sin,
            h_logits,
            vram_bytes,
        })
    }

    /// Get current KV cache position.
    pub fn kv_pos(&self) -> usize {
        self.kv_pos
    }

    /// Reset KV cache position (e.g. at start of new request).
    pub fn reset_kv(&mut self) {
        self.kv_pos = 0;
    }

    /// Roll back KV cache to a given position (discard tokens after pos).
    pub fn rollback_kv(&mut self, pos: usize) {
        if pos < self.kv_pos {
            self.kv_pos = pos;
        }
    }

    /// Run one token through the draft model. Updates KV cache.
    /// After this call, h_logits contains the output logits on host.
    pub fn forward(
        &mut self,
        device: &Arc<CudaDevice>,
        blas: &CudaBlas,
        token_id: usize,
        position: usize,
    ) -> Result<(), String> {
        let hs = self.cfg.hidden_size;
        let nh = self.cfg.num_heads;
        let nkv = self.cfg.num_kv_heads;
        let hd = self.cfg.head_dim;
        let inter = self.cfg.intermediate_size;
        let eps = self.cfg.eps;
        let kv_stride = nkv * hd;
        let half_dim = hd / 2;

        if position >= self.cfg.max_seq {
            return Err(format!(
                "draft position {} >= max_seq {}",
                position, self.cfg.max_seq
            ));
        }

        // ── 1. Embedding lookup ──
        {
            let threads = 256u32;
            let blocks = ((hs as u32) + threads - 1) / threads;
            let f = device
                .get_func(MODULE_NAME, "embedding_lookup")
                .ok_or_else(|| "embedding_lookup not found".to_string())?;
            unsafe {
                f.launch(
                    LaunchConfig {
                        grid_dim: (blocks, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        *self.d_hidden.device_ptr(),
                        *self.d_embedding.device_ptr(),
                        token_id as i32,
                        hs as i32,
                    ),
                )
                .map_err(|e| format!("draft embedding: {:?}", e))?;
            }
        }

        let mut first_residual = true;

        // ── 2. Layer loop ──
        for layer_idx in 0..self.cfg.num_layers {
            let layer = &self.layers[layer_idx];

            // ── Pre-attention norm ──
            {
                let smem = (hs as u32) * 4;
                let threads = 256u32.min(hs as u32);
                let f = device
                    .get_func(MODULE_NAME, "fused_add_rmsnorm")
                    .ok_or_else(|| "fused_add_rmsnorm not found".to_string())?;
                unsafe {
                    f.launch(
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (threads, 1, 1),
                            shared_mem_bytes: smem,
                        },
                        (
                            *self.d_hidden.device_ptr(),
                            *self.d_residual.device_ptr(),
                            *layer.d_input_norm.device_ptr(),
                            eps,
                            hs as i32,
                            if first_residual { 1i32 } else { 0i32 },
                        ),
                    )
                    .map_err(|e| format!("draft norm[{}]: {:?}", layer_idx, e))?;
                }
            }
            first_residual = false;

            // ── Q/K/V projections (INT4 -> FP32) ──
            Self::int4_gemv_f32(
                device,
                &layer.q_proj,
                *self.d_hidden.device_ptr(),
                *self.d_gqa_q.device_ptr(),
            )?;
            Self::int4_gemv_f32(
                device,
                &layer.k_proj,
                *self.d_hidden.device_ptr(),
                *self.d_gqa_k.device_ptr(),
            )?;
            Self::int4_gemv_f32(
                device,
                &layer.v_proj,
                *self.d_hidden.device_ptr(),
                *self.d_gqa_v.device_ptr(),
            )?;

            // ── QK norm (per-head RMSNorm) ──
            {
                let threads = 256u32;
                let norm_fn = device
                    .get_func(MODULE_NAME, "per_head_rmsnorm")
                    .ok_or_else(|| "per_head_rmsnorm not found".to_string())?;
                unsafe {
                    norm_fn
                        .clone()
                        .launch(
                            LaunchConfig {
                                grid_dim: (nh as u32, 1, 1),
                                block_dim: (threads, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                *self.d_gqa_q.device_ptr(),
                                *layer.d_q_norm.device_ptr(),
                                eps,
                                nh as i32,
                                hd as i32,
                                0i32,
                            ),
                        )
                        .map_err(|e| format!("draft q_norm[{}]: {:?}", layer_idx, e))?;
                    norm_fn
                        .launch(
                            LaunchConfig {
                                grid_dim: (nkv as u32, 1, 1),
                                block_dim: (threads, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                *self.d_gqa_k.device_ptr(),
                                *layer.d_k_norm.device_ptr(),
                                eps,
                                nkv as i32,
                                hd as i32,
                                0i32,
                            ),
                        )
                        .map_err(|e| format!("draft k_norm[{}]: {:?}", layer_idx, e))?;
                }
            }

            // ── RoPE ──
            {
                let total_heads = nh + nkv;
                let total_work = total_heads * half_dim;
                let threads = 256u32;
                let blocks = ((total_work as u32) + threads - 1) / threads;
                let rope_fn = device
                    .get_func(MODULE_NAME, "apply_rope")
                    .ok_or_else(|| "apply_rope not found".to_string())?;
                unsafe {
                    rope_fn
                        .launch(
                            LaunchConfig {
                                grid_dim: (blocks, 1, 1),
                                block_dim: (threads, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                *self.d_gqa_q.device_ptr(),
                                *self.d_gqa_k.device_ptr(),
                                *self.d_rope_cos.device_ptr(),
                                *self.d_rope_sin.device_ptr(),
                                position as i32,
                                nh as i32,
                                nkv as i32,
                                hd as i32,
                                half_dim as i32,
                            ),
                        )
                        .map_err(|e| format!("draft rope[{}]: {:?}", layer_idx, e))?;
                }
            }

            // ── KV cache write (FP32 -> FP8) ──
            {
                let threads = 256u32;
                let blocks = ((kv_stride as u32) + threads - 1) / threads;
                let kv_fn = device
                    .get_func(MODULE_NAME, "kv_cache_write")
                    .ok_or_else(|| "kv_cache_write not found".to_string())?;
                unsafe {
                    kv_fn
                        .launch(
                            LaunchConfig {
                                grid_dim: (blocks, 1, 1),
                                block_dim: (threads, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                *self.kv_k[layer_idx].device_ptr(),
                                *self.kv_v[layer_idx].device_ptr(),
                                *self.d_gqa_k.device_ptr(),
                                *self.d_gqa_v.device_ptr(),
                                position as i32,
                                kv_stride as i32,
                            ),
                        )
                        .map_err(|e| format!("draft kv_write[{}]: {:?}", layer_idx, e))?;
                }
            }

            // ── GQA attention ──
            {
                let threads = 256u32;
                let seq_len = (position + 1) as u32;
                let sm_scale = 1.0f32 / (hd as f32).sqrt();
                let q_smem = (hd as u32) * 4;
                let shared_mem_bytes = q_smem + seq_len * 4 + 128;
                let attn_fn = device
                    .get_func(MODULE_NAME, "gqa_attention")
                    .ok_or_else(|| "gqa_attention not found".to_string())?;
                unsafe {
                    attn_fn
                        .launch(
                            LaunchConfig {
                                grid_dim: (nh as u32, 1, 1),
                                block_dim: (threads, 1, 1),
                                shared_mem_bytes,
                            },
                            (
                                *self.d_gqa_out.device_ptr(),
                                *self.d_gqa_q.device_ptr(),
                                *self.kv_k[layer_idx].device_ptr(),
                                *self.kv_v[layer_idx].device_ptr(),
                                sm_scale,
                                nh as i32,
                                nkv as i32,
                                hd as i32,
                                seq_len as i32,
                                self.cfg.max_seq as i32,
                                1i32, // use_smem=true (draft sequences are short)
                            ),
                        )
                        .map_err(|e| format!("draft attn[{}]: {:?}", layer_idx, e))?;
                }
            }

            // ── O projection: FP32 -> BF16 -> INT4 GEMV -> BF16 ──
            {
                let o_size = nh * hd;
                let f = device
                    .get_func(MODULE_NAME, "fp32_to_bf16")
                    .ok_or_else(|| "fp32_to_bf16 not found".to_string())?;
                unsafe {
                    f.launch(
                        LaunchConfig::for_num_elems(o_size as u32),
                        (
                            *self.d_scratch.device_ptr(),
                            *self.d_gqa_out.device_ptr(),
                            o_size as i32,
                        ),
                    )
                    .map_err(|e| format!("draft fp32_to_bf16[{}]: {:?}", layer_idx, e))?;
                }
                Self::int4_gemv_bf16(
                    device,
                    &layer.o_proj,
                    *self.d_scratch.device_ptr(),
                    *self.d_hidden.device_ptr(),
                )?;
            }

            // ── Post-attention norm ──
            {
                let smem = (hs as u32) * 4;
                let threads = 256u32.min(hs as u32);
                let f = device
                    .get_func(MODULE_NAME, "fused_add_rmsnorm")
                    .ok_or_else(|| "fused_add_rmsnorm not found".to_string())?;
                unsafe {
                    f.launch(
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (threads, 1, 1),
                            shared_mem_bytes: smem,
                        },
                        (
                            *self.d_hidden.device_ptr(),
                            *self.d_residual.device_ptr(),
                            *layer.d_post_attn_norm.device_ptr(),
                            eps,
                            hs as i32,
                            0i32,
                        ),
                    )
                    .map_err(|e| format!("draft post_norm[{}]: {:?}", layer_idx, e))?;
                }
            }

            // ── Dense MLP ──
            // Gate projection -> first half of d_gate_up
            Self::int4_gemv_bf16(
                device,
                &layer.gate_proj,
                *self.d_hidden.device_ptr(),
                *self.d_gate_up.device_ptr(),
            )?;
            // Up projection -> second half of d_gate_up
            {
                let up_offset = inter * 2; // bytes offset = inter * sizeof(BF16)
                let up_ptr = *self.d_gate_up.device_ptr() + up_offset as u64;
                Self::int4_gemv_bf16(device, &layer.up_proj, *self.d_hidden.device_ptr(), up_ptr)?;
            }

            // SiLU * mul
            {
                let threads = 256u32;
                let blocks = ((inter as u32) + threads - 1) / threads;
                let f = device
                    .get_func(MODULE_NAME, "silu_mul")
                    .ok_or_else(|| "silu_mul not found".to_string())?;
                unsafe {
                    f.launch(
                        LaunchConfig {
                            grid_dim: (blocks, 1, 1),
                            block_dim: (threads, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            *self.d_mlp_scratch.device_ptr(),
                            *self.d_gate_up.device_ptr(),
                            inter as i32,
                        ),
                    )
                    .map_err(|e| format!("draft silu_mul[{}]: {:?}", layer_idx, e))?;
                }
            }

            // Down projection -> hidden
            Self::int4_gemv_bf16(
                device,
                &layer.down_proj,
                *self.d_mlp_scratch.device_ptr(),
                *self.d_hidden.device_ptr(),
            )?;
        }

        // ── 3. Final residual add + final norm ──
        {
            let smem = (hs as u32) * 4;
            let threads = 256u32.min(hs as u32);
            let f = device
                .get_func(MODULE_NAME, "fused_add_rmsnorm")
                .ok_or_else(|| "fused_add_rmsnorm not found".to_string())?;
            unsafe {
                f.launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: smem,
                    },
                    (
                        *self.d_hidden.device_ptr(),
                        *self.d_residual.device_ptr(),
                        *self.d_final_norm.device_ptr(),
                        eps,
                        hs as i32,
                        0i32,
                    ),
                )
                .map_err(|e| format!("draft final_norm: {:?}", e))?;
            }
        }

        // ── 4. LM head (BF16 embedding, cuBLAS for quality) ──
        {
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            unsafe {
                cublas_result::gemm_ex(
                    *blas.handle(),
                    cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    self.cfg.vocab_size as i32,
                    1,
                    hs as i32,
                    &alpha as *const f32 as *const std::ffi::c_void,
                    *self.d_embedding.device_ptr() as *const std::ffi::c_void,
                    cublas_sys::cudaDataType::CUDA_R_16BF,
                    hs as i32,
                    *self.d_hidden.device_ptr() as *const std::ffi::c_void,
                    cublas_sys::cudaDataType::CUDA_R_16BF,
                    hs as i32,
                    &beta as *const f32 as *const std::ffi::c_void,
                    *self.d_logits.device_ptr() as *mut std::ffi::c_void,
                    cublas_sys::cudaDataType::CUDA_R_32F,
                    self.cfg.vocab_size as i32,
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
                .map_err(|e| format!("draft lm_head gemv: {:?}", e))?;
            }
        }

        // ── 5. D2H logits ──
        unsafe {
            let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                self.h_logits.as_mut_ptr() as *mut std::ffi::c_void,
                *self.d_logits.device_ptr(),
                self.cfg.vocab_size * 4,
            );
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("draft D2H logits: {:?}", err));
            }
        }

        self.kv_pos = position + 1;
        Ok(())
    }

    /// Generate K draft tokens autoregressively. Returns token_ids.
    pub fn generate_draft(
        &mut self,
        device: &Arc<CudaDevice>,
        blas: &CudaBlas,
        first_token: usize,
        start_position: usize,
        num_draft: usize,
    ) -> Result<Vec<usize>, String> {
        let vocab_size = self.cfg.vocab_size;
        let mut draft_tokens = Vec::with_capacity(num_draft);
        let mut next_token = first_token;

        for i in 0..num_draft {
            let pos = start_position + i;
            self.forward(device, blas, next_token, pos)?;

            // Greedy argmax
            let mut best_id = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for j in 0..vocab_size {
                if self.h_logits[j] > best_val {
                    best_val = self.h_logits[j];
                    best_id = j;
                }
            }

            draft_tokens.push(best_id);
            next_token = best_id;
        }

        Ok(draft_tokens)
    }

    /// Warm up the draft model by feeding it a sequence of tokens.
    pub fn warmup_context(
        &mut self,
        device: &Arc<CudaDevice>,
        blas: &CudaBlas,
        tokens: &[usize],
        start_position: usize,
    ) -> Result<(), String> {
        for (i, &tok) in tokens.iter().enumerate() {
            self.forward(device, blas, tok, start_position + i)?;
        }
        Ok(())
    }

    // ── INT4 GEMV kernel launches ──

    /// INT4 GEMV with FP32 output. Used for Q/K/V projections.
    fn int4_gemv_f32(
        device: &Arc<CudaDevice>,
        w: &DraftWeightInt4,
        input_ptr: u64,
        output_ptr: u64,
    ) -> Result<(), String> {
        let rows = w.rows;
        let cols = w.cols;
        let gs = w.group_size;
        let blocks = ((rows + 7) / 8) as u32; // 8 warps per block, 1 row per warp
        let smem = (cols as u32) * 2; // BF16 input in shared memory

        let f = device
            .get_func(MODULE_NAME, "simple_int4_gemv_f32")
            .ok_or_else(|| "simple_int4_gemv_f32 not found".to_string())?;
        unsafe {
            f.launch(
                LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: smem,
                },
                (
                    w.packed_ptr(),
                    w.scales_ptr(),
                    input_ptr,
                    output_ptr,
                    rows as i32,
                    cols as i32,
                    gs as i32,
                ),
            )
            .map_err(|e| format!("int4_gemv_f32: {:?}", e))?;
        }
        Ok(())
    }

    /// INT4 GEMV with BF16 output. Used for O/gate/up/down projections.
    fn int4_gemv_bf16(
        device: &Arc<CudaDevice>,
        w: &DraftWeightInt4,
        input_ptr: u64,
        output_ptr: u64,
    ) -> Result<(), String> {
        let rows = w.rows;
        let cols = w.cols;
        let gs = w.group_size;
        let blocks = ((rows + 7) / 8) as u32;
        let smem = (cols as u32) * 2;

        let f = device
            .get_func(MODULE_NAME, "simple_int4_gemv_bf16")
            .ok_or_else(|| "simple_int4_gemv_bf16 not found".to_string())?;
        unsafe {
            f.launch(
                LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: smem,
                },
                (
                    w.packed_ptr(),
                    w.scales_ptr(),
                    input_ptr,
                    output_ptr,
                    rows as i32,
                    cols as i32,
                    gs as i32,
                ),
            )
            .map_err(|e| format!("int4_gemv_bf16: {:?}", e))?;
        }
        Ok(())
    }
}
