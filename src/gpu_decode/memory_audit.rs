//! Observation only. Enumerated live owned extents, not an allocator total.
//! Call after the decode timer; no CUDA queries, copies or synchronization.
use super::*;
use cudarc::driver::{CudaSlice,DeviceRepr,DeviceSlice};
fn push<T:DeviceRepr>(rows:&mut Vec<serde_json::Value>,label:String,b:&CudaSlice<T>){
    rows.push(serde_json::json!({"label":label,"ptr":*b.device_ptr(),"bytes":b.len()*std::mem::size_of::<T>()}));
}
pub(super) fn snapshot(store:&GpuDecodeStore)->serde_json::Value {
    let mut rows=Vec::new();
    let mut hcs=serde_json::Value::Null;
    let active_attention=crate::matched_teacher::active_attention_execution();
    let captured_attention=store.graph.as_ref().and_then(|g|g.captured_attention_execution);

    if let Some(g)=&store.graph {
    if let Some(b)=&g.d_common_moe_accum{push(&mut rows,"graph.d_common_moe_accum".into(),b);}
    if let Some(b)=&g.d_mixed_expert_bits{push(&mut rows,"graph.d_mixed_expert_bits".into(),b);}
    for (i,b) in g.owned_weight_bf16.iter().enumerate(){push(&mut rows,format!("graph.owned_weight_bf16[{i}]"),b);}
    for (i,b) in g.owned_weight_f32.iter().enumerate(){push(&mut rows,format!("graph.owned_weight_f32[{i}]"),b);}
    for (i,b) in g.owned_weight_packed.iter().enumerate(){push(&mut rows,format!("graph.owned_weight_packed[{i}]"),b);}
    push(&mut rows,"graph.d_hidden".into(),&g.d_hidden);
    push(&mut rows,"graph.d_residual".into(),&g.d_residual);
    push(&mut rows,"graph.d_logits".into(),&g.d_logits);
    push(&mut rows,"graph.d_fp32_scratch".into(),&g.d_fp32_scratch);
    for (i,b) in g.d_graph_expert_bufs.iter().enumerate(){push(&mut rows,format!("graph.d_graph_expert_bufs[{i}]"),b);}
    push(&mut rows,"graph.d_expert_buf_a0".into(),&g.d_expert_buf_a0);
    push(&mut rows,"graph.d_expert_buf_b0".into(),&g.d_expert_buf_b0);
    push(&mut rows,"graph.d_expert_buf_a1".into(),&g.d_expert_buf_a1);
    push(&mut rows,"graph.d_expert_buf_b1".into(),&g.d_expert_buf_b1);
    push(&mut rows,"graph.d_topk_indices".into(),&g.d_topk_indices);
    push(&mut rows,"graph.d_topk_weights".into(),&g.d_topk_weights);
    push(&mut rows,"graph.d_moe_out".into(),&g.d_moe_out);
    push(&mut rows,"graph.d_expert_out".into(),&g.d_expert_out);
    push(&mut rows,"graph.d_expert_gate_up".into(),&g.d_expert_gate_up);
    push(&mut rows,"graph.d_expert_scratch".into(),&g.d_expert_scratch);
    push(&mut rows,"graph.d_inv_weight_perm".into(),&g.d_inv_weight_perm);
    push(&mut rows,"graph.d_inv_scale_perm".into(),&g.d_inv_scale_perm);
    push(&mut rows,"graph.d_inv_weight_perm_int8".into(),&g.d_inv_weight_perm_int8);
    push(&mut rows,"graph.d_v2_partial".into(),&g.d_v2_partial);
    push(&mut rows,"graph.d_batch_gate_ups".into(),&g.d_batch_gate_ups);
    push(&mut rows,"graph.d_batch_partials".into(),&g.d_batch_partials);
    push(&mut rows,"graph.d_batch_expert_outs".into(),&g.d_batch_expert_outs);
    push(&mut rows,"graph.d_batch_w13_packed_ptrs".into(),&g.d_batch_w13_packed_ptrs);
    push(&mut rows,"graph.d_batch_w13_scales_ptrs".into(),&g.d_batch_w13_scales_ptrs);
    push(&mut rows,"graph.d_batch_w2_packed_ptrs".into(),&g.d_batch_w2_packed_ptrs);
    push(&mut rows,"graph.d_batch_w2_scales_ptrs".into(),&g.d_batch_w2_scales_ptrs);
    push(&mut rows,"graph.d_batch_weights".into(),&g.d_batch_weights);
    push(&mut rows,"graph.d_mixed_weights".into(),&g.d_mixed_weights);
    push(&mut rows,"graph.d_batch_upload".into(),&g.d_batch_upload);
    push(&mut rows,"graph.d_gqa_q".into(),&g.d_gqa_q);
    push(&mut rows,"graph.d_gqa_k".into(),&g.d_gqa_k);
    push(&mut rows,"graph.d_gqa_v".into(),&g.d_gqa_v);
    push(&mut rows,"graph.d_gqa_out".into(),&g.d_gqa_out);
    push(&mut rows,"graph.d_mla_q_absorbed".into(),&g.d_mla_q_absorbed);
    push(&mut rows,"graph.d_mla_kv".into(),&g.d_mla_kv);
    push(&mut rows,"graph.d_mla_attn_out".into(),&g.d_mla_attn_out);
    if let Some(workspace)=&g.mla_split_workspace{push(&mut rows,"graph.mla_split_workspace".into(),workspace);}
    if let Some(pairs)=&g.mla_pair_capture { for p in pairs {
        let prefix=format!("graph.mla_pair.layer{}",p.layer);
        push(&mut rows,format!("{prefix}.q"),&p.q); push(&mut rows,format!("{prefix}.cache"),&p.cache);
        push(&mut rows,format!("{prefix}.indices"),&p.indices); push(&mut rows,format!("{prefix}.count"),&p.count);
        push(&mut rows,format!("{prefix}.sequence"),&p.sequence); push(&mut rows,format!("{prefix}.candidate"),&p.candidate);
        push(&mut rows,format!("{prefix}.baseline"),&p.baseline);
    } }
    if let Some((p,i,c))=&g.mla_selected_stage{push(&mut rows,"graph.mla_selected_stage.packed".into(),p);push(&mut rows,"graph.mla_selected_stage.indices".into(),i);push(&mut rows,"graph.mla_selected_stage.count".into(),c);}
    if let Some(b)=&g.d_gqa_tiled_o{push(&mut rows,"graph.d_gqa_tiled_o".into(),b);}
    if let Some(b)=&g.d_gqa_tiled_lse{push(&mut rows,"graph.d_gqa_tiled_lse".into(),b);}
    push(&mut rows,"graph.d_la_recur_out".into(),&g.d_la_recur_out);
    if let Some(b)=&g.d_mamba2_conv_out{push(&mut rows,"graph.d_mamba2_conv_out".into(),b);}
    if let Some(b)=&g.d_mamba2_bf16_graph_input{push(&mut rows,"graph.d_mamba2_bf16_graph_input".into(),b);}
    if let Some(b)=&g.d_mamba2_bf16_graph_output{push(&mut rows,"graph.d_mamba2_bf16_graph_output".into(),b);}
    if let Some(b)=&g.d_rope_cos{push(&mut rows,"graph.d_rope_cos".into(),b);}
    if let Some(b)=&g.d_rope_sin{push(&mut rows,"graph.d_rope_sin".into(),b);}
    if let Some(b)=&g.d_gqa_gate_buf{push(&mut rows,"graph.d_gqa_gate_buf".into(),b);}
    if let Some(b)=&g.d_graph_gqa_k4_split_clocks{push(&mut rows,"graph.d_graph_gqa_k4_split_clocks".into(),b);}
    if let Some(b)=&g.d_graph_mixed_segment_clocks{push(&mut rows,"graph.d_graph_mixed_segment_clocks".into(),b);}
    if let Some(b)=&g.d_graph_gqa_path_clocks{push(&mut rows,"graph.d_graph_gqa_path_clocks".into(),b);}
    if let Some(b)=&g.d_graph_mla_path_clocks{push(&mut rows,"graph.d_graph_mla_path_clocks".into(),b);}
    if let Some(b)=&g.d_graph_gqa_other_clocks{push(&mut rows,"graph.d_graph_gqa_other_clocks".into(),b);}
    if let Some(b)=&g.d_graph_gqa_hd256_attn_clocks{push(&mut rows,"graph.d_graph_gqa_hd256_attn_clocks".into(),b);}
    if let Some(b)=&g.d_graph_gqa_hd256_attn_stats{push(&mut rows,"graph.d_graph_gqa_hd256_attn_stats".into(),b);}
    if let Some(b)=&g.d_graph_moe_route_clocks{push(&mut rows,"graph.d_graph_moe_route_clocks".into(),b);}
    if let Some(b)=&g.d_graph_moe_w2_clocks{push(&mut rows,"graph.d_graph_moe_w2_clocks".into(),b);}
    if let Some(b)=&g.d_graph_route_prep_clocks{push(&mut rows,"graph.d_graph_route_prep_clocks".into(),b);}
    if let Some(b)=&g.d_graph_mamba2_clocks{push(&mut rows,"graph.d_graph_mamba2_clocks".into(),b);}
    if let Some(b)=&g.d_graph_la_clocks{push(&mut rows,"graph.d_graph_la_clocks".into(),b);}
    if let Some(b)=&g.d_graph_final_clocks{push(&mut rows,"graph.d_graph_final_clocks".into(),b);}
    if let Some(b)=&g.d_batch_hidden{push(&mut rows,"graph.d_batch_hidden".into(),b);}
    if let Some(b)=&g.d_batch_residual{push(&mut rows,"graph.d_batch_residual".into(),b);}
    if let Some(b)=&g.d_batch_moe_out{push(&mut rows,"graph.d_batch_moe_out".into(),b);}
    if let Some(b)=&g.d_batch_logits{push(&mut rows,"graph.d_batch_logits".into(),b);}
    if let Some(b)=&g.d_batch_topk_ids{push(&mut rows,"graph.d_batch_topk_ids".into(),b);}
    if let Some(b)=&g.d_batch_topk_wts{push(&mut rows,"graph.d_batch_topk_wts".into(),b);}
    if let Some(b)=&g.d_batch_gate_logits{push(&mut rows,"graph.d_batch_gate_logits".into(),b);}
    if let Some(b)=&g.d_la_hidden_stack{push(&mut rows,"graph.d_la_hidden_stack".into(),b);}
    if let Some(b)=&g.d_batch_proj_a{push(&mut rows,"graph.d_batch_proj_a".into(),b);}
    if let Some(b)=&g.d_batch_proj_b{push(&mut rows,"graph.d_batch_proj_b".into(),b);}
    if let Some(b)=&g.d_batch_attn_out{push(&mut rows,"graph.d_batch_attn_out".into(),b);}
    if let Some(b)=&g.d_graph_token_id{push(&mut rows,"graph.d_graph_token_id".into(),b);}
    if let Some(b)=&g.d_graph_pos{push(&mut rows,"graph.d_graph_pos".into(),b);}
    if let Some(b)=&g.d_graph_rope_pos{push(&mut rows,"graph.d_graph_rope_pos".into(),b);}
    if let Some(b)=&g.d_graph_seq_len{push(&mut rows,"graph.d_graph_seq_len".into(),b);}
    for (i,b) in g.d_graph_seq_len_by_layer.iter().enumerate(){push(&mut rows,format!("graph.d_graph_seq_len_by_layer[{i}]"),b);}
    if let Some(b)=&g.d_dummy_expert{push(&mut rows,"graph.d_dummy_expert".into(),b);}
    if let Some(b)=&g.d_dummy_ptrs{push(&mut rows,"graph.d_dummy_ptrs".into(),b);}
    if let Some(b)=&g.d_cublas_workspace{push(&mut rows,"graph.d_cublas_workspace".into(),b);}
    for (i,b) in g.d_prefetch_bufs.iter().enumerate(){push(&mut rows,format!("graph.d_prefetch_bufs[{i}]"),b);}
    if let Some(b)=&g.d_wts_full{push(&mut rows,"graph.d_wts_full".into(),b);}
    if let Some(b)=&g.d_wts_hot{push(&mut rows,"graph.d_wts_hot".into(),b);}
        if let Some(h)=&g.hcs {hcs=serde_json::json!({"nominal_vram_bytes":h.vram_bytes,"cached_experts":h.num_cached,"hard_pool_slots":h.pool_num_slots,"soft_pool_slots":h.soft_num_slots,"soft_chunks_loaded":h.soft_chunks_loaded,"loaded_soft_chunk_nominal_bytes":h.soft_chunk_bytes.iter().take(h.soft_chunks_loaded).sum::<usize>()});}
        for (i,entry) in g.shared_expert_vram.iter().enumerate(){if let Some(entry)=entry{if let Some(b)=&entry.d_buf{push(&mut rows,format!("shared_expert[{i}]"),b);}}}
    }
    let hqq:Vec<serde_json::Value>=store.hqq_runtime_slots.iter().filter_map(|slot|slot.device_buffers.as_ref().map(|b|serde_json::json!({"layer":slot.layer_idx,"tensor":slot.tensor_name,"stage":slot.current_stage,"logical_bytes":slot.device_bytes(),"mapped_bytes":b.mapped_bytes(),"virtual_reserved_bytes":b.reserved_bytes()}))).collect();
    serde_json::json!({"active_attention_execution":active_attention,"captured_attention_execution":captured_attention,"scope":"direct graph-owned CudaSlice extents, shared expert owned buffers and HQQ VMM mappings. HCS nominal residency is separate; external Torch tensors, nested attention/indexer/state owners, graph/BLAS/module/driver allocations remain separately accounted. Do not sum aliases as physical allocations.","graph_owned_extents":rows,"hqq":hqq,"hcs":hcs})
}
