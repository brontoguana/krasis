//! MoE forward dispatch — routes experts to GPU (prefill) or CPU (decode).

use pyo3::prelude::*;

/// Main MoE runner that SGLang calls as a FusedMoE replacement.
///
/// Handles:
/// - Receiving activations + expert assignments from PyTorch
/// - Dispatching to GPU (fused_marlin_moe) for prefill
/// - Dispatching to CPU thread pool for decode
/// - Combining results and returning to PyTorch
#[pyclass]
pub struct MoERunner {
    // TODO: weight store reference, thread pool, config
}

#[pymethods]
impl MoERunner {
    #[new]
    pub fn new() -> Self {
        MoERunner {}
    }
}
