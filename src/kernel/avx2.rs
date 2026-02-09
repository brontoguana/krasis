//! AVX2 INT4 expert matmul kernel for Zen 2 (EPYC 7742).
//!
//! Core strategy: fused unpack-multiply-accumulate in registers.
//! - Load 32 bytes (64 INT4 values) into YMM register
//! - Separate low/high nibbles via mask + shift
//! - Sign-extend to INT16 via arithmetic right shift
//! - Multiply against INT16-quantized activations via _mm256_madd_epi16
//! - Accumulate into INT32 running totals
//! - No intermediate memory writes (bandwidth-bound workload)
//!
//! The activation vector (14KB at BF16) is pre-permuted to match Marlin's
//! weight layout, then pinned in L1 cache. Weight matrix is scanned sequentially.

// #[cfg(target_arch = "x86_64")]
// use std::arch::x86_64::*; // TODO: uncomment when kernel is implemented

/// Compute y = x @ W^T for a single expert.
///
/// # Arguments
/// * `weights` - Marlin-permuted INT4 packed weights (K/8 × N int32s)
/// * `scales` - BF16 per-group dequantization scales
/// * `activation` - Pre-permuted activation vector (K values, BF16)
/// * `output` - Output buffer (N values, FP32)
/// * `k` - Input dimension (e.g. 2048 for V2-Lite, 7168 for Kimi K2.5)
/// * `n` - Output dimension (e.g. 1408 for V2-Lite expert intermediate)
/// * `group_size` - Quantization group size (typically 128)
///
/// # Safety
/// Requires AVX2 + FMA. Pointers must be valid and properly aligned.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn expert_matmul_int4(
    _weights: *const u8,
    _scales: *const u16, // BF16 as raw u16
    _activation: *const u16, // BF16 as raw u16
    _output: *mut f32,
    _k: usize,
    _n: usize,
    _group_size: usize,
) {
    // TODO: implement the INT4 unpack + madd kernel
    // This is the hot path — every CPU expert matmul goes through here
    todo!("INT4 AVX2 matmul kernel")
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_avx2_available() {
        assert!(is_x86_feature_detected!("avx2"), "AVX2 required");
        assert!(is_x86_feature_detected!("fma"), "FMA required");
    }
}
