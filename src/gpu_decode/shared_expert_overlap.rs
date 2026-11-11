//! Move shared W13/reduction and W2 partial production before the copy wait.
//! Single-expert gate-up/partial buffers are disjoint from routed batched
//! scratch. W2 partials remain untouched until the original reduction/addition
//! after routed work. No new buffers or arithmetic.
use super::*;

pub(super) fn enabled(graph: &GpuDecodeGraph, layer: usize) -> bool {
    if !graph.live_switches.SHARED_EXPERT_COPY_OVERLAP() {
        return false;
    }
    let Some(moe) = graph.moe_layers.get(layer).and_then(Option::as_ref) else {
        return false;
    };
    graph.glm5_hc_mean_mult.is_some()
        && graph.split_expert_launch
        && !graph.gpu_route_sync
        && !graph.mapped_reads_active
        && graph.peer_experts.is_empty()
        && graph.apfl.is_none()
        && graph.synthetic_repack.is_none()
        && graph.expert_compression.is_none()
        && !graph.mixed_expert_precision
        && graph.ordered_expert_projection.is_none()
        && graph.expert_bits == 4
        && graph.shared_expert_bits == 8
        && moe.gated_experts
        && moe.moe_input_size == 0
        && moe.latent_down_wid.is_none()
        && moe.latent_up_wid.is_none()
        && moe.shared_gate_wid.is_none()
        && graph
            .shared_expert_vram
            .get(layer)
            .and_then(Option::as_ref)
            .is_some()
        && select_shared_w13_ksplits(graph, 2 * graph.shared_expert_intermediate_size) > 1
        && select_shared_w2_ksplits(
            graph,
            graph.shared_expert_intermediate_size,
            graph.hidden_size,
            true,
        )
        .is_some_and(|splits| splits > 0)
}

pub(super) fn launch(graph: &GpuDecodeGraph, layer: usize) -> Result<(), String> {
    if !enabled(graph, layer) {
        return Ok(());
    }
    let shared = graph.shared_expert_vram[layer]
        .as_ref()
        .ok_or("shared overlap weights absent")?;
    let kernels = graph
        .kernels
        .as_ref()
        .ok_or("shared overlap kernels absent")?;
    let width = graph.hidden_size;
    let rows = 2 * graph.shared_expert_intermediate_size;
    let splits = select_shared_w13_ksplits(graph, rows);
    let partial = *graph.d_v2_partial.device_ptr();
    // Match launch_marlin_gemv_v2 and launch_reduce_ksplits_bf16 exactly.
    // Both launch onto the device's current replay stream.
    unsafe {
        kernels
            .marlin_gemv_int8_v2
            .clone()
            .launch(
                LaunchConfig {
                    grid_dim: (((rows + 15) / 16) as u32, splits as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: (width * 2 + 1024 * 4 + 64 * 4 + 16 * 16 * 4) as u32,
                },
                (
                    shared.w13_packed_ptr(),
                    shared.w13_scales_ptr(),
                    *graph.d_hidden.device_ptr(),
                    partial,
                    *graph.d_inv_weight_perm_int8.device_ptr(),
                    *graph.d_inv_scale_perm.device_ptr(),
                    width as i32,
                    rows as i32,
                    graph.group_size as i32,
                    splits as i32,
                ),
            )
            .map_err(|error| format!("shared overlap W13 layer={layer}: {error:?}"))?;
        kernels
            .reduce_ksplits_bf16
            .clone()
            .launch(
                LaunchConfig {
                    grid_dim: (((rows + 255) / 256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    *graph.d_expert_gate_up.device_ptr(),
                    partial,
                    rows as i32,
                    splits as i32,
                ),
            )
            .map_err(|error| format!("shared overlap reduce layer={layer}: {error:?}"))?;
    }
    let moe = graph.moe_layers[layer]
        .as_ref()
        .ok_or("shared overlap MoE absent")?;
    let intermediate = graph.shared_expert_intermediate_size;
    let w2_splits = select_shared_w2_ksplits(graph, intermediate, width, true)
        .ok_or("shared overlap W2 split absent")?;
    // Match launch_fused_silu_accum_v2 exactly. Do not reduce or read the
    // routed accumulator yet: that operation retains its original graph slot.
    unsafe {
        kernels
            .fused_silu_accum_v2_int8
            .clone()
            .launch(
                LaunchConfig {
                    grid_dim: (((width + 15) / 16) as u32, w2_splits as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: (intermediate * 2 + 1024 * 4 + 64 * 4 + 16 * 16 * 4) as u32,
                },
                (
                    shared.w2_packed_ptr(),
                    shared.w2_scales_ptr(),
                    *graph.d_expert_gate_up.device_ptr(),
                    partial,
                    *graph.d_inv_weight_perm_int8.device_ptr(),
                    *graph.d_inv_scale_perm.device_ptr(),
                    intermediate as i32,
                    width as i32,
                    graph.group_size as i32,
                    w2_splits as i32,
                    moe.shared_swiglu_limit,
                    if moe.deepseek_v4_activation {
                        2i32
                    } else {
                        0i32
                    },
                ),
            )
            .map_err(|error| format!("shared overlap W2 layer={layer}: {error:?}"))?;
    }
    Ok(())
}
