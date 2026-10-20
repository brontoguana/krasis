pub mod adaptive_cold_drop;
mod capture_phase;
mod capture_window;
pub mod chat_template;
pub mod checkpoint_identity;
pub mod cpu_tail;
pub mod cpu_tail_calibrate;
pub mod decode;
pub mod draft_model;
mod dsa_score;
pub mod expert_codec;
pub mod expert_sidecar;
pub mod gguf;
pub mod gguf_kernels;
pub mod gpu_decode;
pub mod gpu_prefill;
pub mod hqq;
pub mod kernel;
pub mod manager;
// BEGIN selected prefill row module
mod dense_ffn_probe;
mod matched_prefill_capture;
mod matched_prefill_row;
mod mla_probe;
// END selected prefill row module
mod matched_hqq_rounding;
mod matched_mla_stage;
mod matched_prefill_logits;
mod matched_probe;
mod matched_router_capture;
// BEGIN bounded model KDA diagnostic module
mod matched_kda_model;
// END bounded model KDA diagnostic module
mod matched_replay;
mod matched_routes;
// BEGIN explicit teacher-forced diagnostic module
mod matched_projection_transfer;
mod matched_teacher;
// END explicit teacher-forced diagnostic module
#[cfg(feature = "copy-lifecycle-test")]
mod copy_fault;
mod cuda_failure_code;
mod cuda_region;
pub mod moe;
pub mod numa;
pub mod pcie_batch;
mod sequence_copy_completion;
pub mod sequence_external_owner;
mod sequence_restore_validity;
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
    #[cfg(feature = "shared-kda-test")]
    m.add_function(wrap_pyfunction!(
        model_tensor_owner::model_tensor_owner_fixture,
        m
    )?)?;
    #[cfg(feature = "shared-kda-test")]
    m.add_function(wrap_pyfunction!(
        model_tensor_owner::model_tensor_widen_fixture,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(syscheck::system_check, m)?)?;
    m.add_function(wrap_pyfunction!(manager::run_manager, m)?)?;
    Ok(())
}

mod common_marlin;
mod common_moe_combine;
mod common_rmsnorm;
mod kda_prefix_owner;
mod mla_f32;
mod model_tensor_owner;
mod ordered_projection;
mod stable_mla_gemm;
mod stable_mla_ops;
mod stable_mla_owned;
mod stable_mla_path;
mod stable_mla_runtime;
mod stable_mla_select;

mod prefix_recurrence;

mod prefix_lifecycle_observer;

mod reference_numbers;
