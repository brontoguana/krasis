pub mod kernel;
pub mod moe;
pub mod weights;

use pyo3::prelude::*;

/// Krasis — hybrid LLM MoE runtime
#[pymodule]
fn krasis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    env_logger::init();
    m.add_class::<moe::MoERunner>()?;
    m.add_class::<weights::WeightStore>()?;
    Ok(())
}
