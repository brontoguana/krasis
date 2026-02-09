//! Marlin INT4 format conversion and handling.
//!
//! Converts HF model weights (BF16 or compressed-tensors INT4) to Marlin-permuted
//! INT4 packed format. Caches to disk as safetensors for subsequent loads.
//!
//! Format per expert per layer:
//! - w13_packed: [K/16, N*2] int32  (gate+up fused, Marlin-permuted)
//! - w13_scale:  [K/G, N]   bf16   (G = group_size, typically 128)
//! - w2_packed:  [K/16, N*2] int32  (down proj, Marlin-permuted)
//! - w2_scale:   [K/G, N]   bf16

/// Marlin tile size constants
pub const MARLIN_TILE_N: usize = 16;
pub const MARLIN_TILE_K: usize = 16;

/// Convert BF16 weight matrix to Marlin INT4 packed format.
///
/// Steps:
/// 1. Symmetric INT4 quantization per group (group_size values)
/// 2. Pack 8 INT4 values per int32
/// 3. Apply Marlin permutation (reorder for GPU warp coalescing)
pub fn quantize_and_pack_marlin(
    _weight_bf16: &[u16], // BF16 as raw u16
    _rows: usize,
    _cols: usize,
    _group_size: usize,
) -> (Vec<u32>, Vec<u16>) {
    // TODO: implement quantization + Marlin packing
    // Returns (packed_weights, scales)
    todo!("Marlin INT4 quantize and pack")
}

/// Apply Marlin permutation to already-packed INT4 weights.
/// This reorders the packed int32 words for GPU warp-level memory coalescing.
pub fn marlin_repack(_packed: &[u32], _k: usize, _n: usize) -> Vec<u32> {
    // TODO: implement the Marlin repack permutation
    // This is the same permutation as gptq_marlin_moe_repack in vLLM
    todo!("Marlin repack permutation")
}
