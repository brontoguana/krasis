pub mod kernel;
pub mod moe;
pub mod weights;

use pyo3::prelude::*;

/// Krasis — hybrid LLM MoE runtime
#[pymodule]
fn krasis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = env_logger::try_init();
    m.add_class::<moe::KrasisEngine>()?;
    m.add_class::<weights::WeightStore>()?;
    Ok(())
}
