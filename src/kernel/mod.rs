//! CPU matmul kernels for INT4 Marlin-format weights.

#[cfg(target_arch = "x86_64")]
pub mod avx2;
