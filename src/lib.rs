pub mod adaptive_cold_drop;
pub mod chat_template;
pub mod cpu_tail;
pub mod cpu_tail_calibrate;
pub mod decode;
pub mod draft_model;
pub mod expert_codec;
pub mod expert_sidecar;
pub mod gguf;
pub mod gguf_kernels;
pub mod gpu_decode;
pub mod gpu_prefill;
pub mod hqq;
pub mod kernel;
pub mod moe;
pub mod numa;
pub mod pcie_batch;
pub mod server;
pub mod session_cache;
mod synthetic_repack;
pub mod syscheck;
pub mod text_only_messages;
pub mod vram_monitor;
pub mod weights;

use pyo3::prelude::*;

/// Krasis — hybrid LLM MoE runtime
#[pymodule]
fn krasis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = env_logger::try_init();
    m.add_class::<moe::KrasisEngine>()?;
    m.add_class::<weights::WeightStore>()?;
    m.add_class::<decode::CpuDecodeStore>()?;
    m.add_class::<server::RustServer>()?;
    m.add_class::<gpu_decode::GpuDecodeStore>()?;
    m.add_class::<vram_monitor::VramMonitor>()?;
    m.add_function(wrap_pyfunction!(hqq::hqq4_init_group_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(hqq::hqq4_solve_group_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(hqq::hqq4_rmse_group_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(hqq::hqq4_quantize_tensor_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(hqq::hqq_search_cuda_tensor_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(syscheck::system_check, m)?)?;
    Ok(())
}
