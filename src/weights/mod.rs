//! Weight loading and format management.
//!
//! Single format: Marlin INT4 (permuted) stored as safetensors, mmap'd into RAM.
//! Both GPU and CPU read from the same mmap'd region.

pub mod marlin;
pub mod safetensors_io;

use pyo3::prelude::*;

/// Manages mmap'd weight files and provides access to expert weights.
#[pyclass]
pub struct WeightStore {
    // TODO: mmap handles, layer index, NUMA placement info
}

#[pymethods]
impl WeightStore {
    #[new]
    pub fn new() -> Self {
        WeightStore {}
    }
}
