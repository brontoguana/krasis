//! Explicitly selected binary prefill diagnostics. No diagnostic GPU allocation.
use crate::matched_probe::Target;
use cudarc::driver::sys;
use serde_json::json;

fn copy_row(
    target: &Target,
    name: &str,
    ptr: u64,
    width: usize,
    element_bytes: usize,
    stream: sys::CUstream,
) -> Result<(), String> {
    let bytes = width
        .checked_mul(element_bytes)
        .ok_or("selected row size overflow")?;
    let offset = target
        .row
        .checked_mul(bytes)
        .ok_or("selected row offset overflow")?;
    target.device(
        name,
        ptr.checked_add(offset as u64)
            .ok_or("selected row pointer overflow")?,
        bytes,
        stream,
    )
}

fn row_operation(label: &str) -> Option<&'static str> {
    match label {
        "layer_glm5_hc_input_state_last" => Some("glm5_row_hc_entry"),
        "layer_glm5_attn_prepared_hidden_last" => Some("glm5_row_attn_prepared"),
        "layer_glm5_attn_norm_hidden_last" => Some("glm5_row_attn_norm"),
        "layer_glm5_attn_sublayer_last" => Some("glm5_row_attn_output"),
        "layer_glm5_post_attn_state_last" => Some("glm5_row_post_attn_state"),
        "layer_glm5_ffn_prepared_hidden_last" => Some("glm5_row_ffn_prepared"),
        "layer_glm5_ffn_norm_hidden_last" => Some("glm5_row_ffn_norm"),
        "layer_glm5_ffn_sublayer_last" => Some("glm5_row_ffn_output"),
        "layer_glm5_output_state_last" => Some("glm5_row_output_state"),
        "layer_glm5_kda_q_proj_last" => Some("glm5_row_kda_q_proj"),
        "layer_glm5_kda_k_proj_last" => Some("glm5_row_kda_k_proj"),
        "layer_glm5_kda_v_proj_last" => Some("glm5_row_kda_v_proj"),
        "layer_glm5_kda_q_conv_last" => Some("glm5_row_kda_q_conv"),
        "layer_glm5_kda_k_conv_last" => Some("glm5_row_kda_k_conv"),
        "layer_glm5_kda_v_conv_last" => Some("glm5_row_kda_v_conv"),
        "layer_glm5_kda_forget_gate_last" => Some("glm5_row_kda_forget_gate"),
        "layer_glm5_kda_beta_last" => Some("glm5_row_kda_beta"),
        "layer_glm5_kda_recurrent_last" => Some("glm5_row_kda_recurrent"),
        "layer_glm5_kda_output_gate_last" => Some("glm5_row_kda_output_gate"),
        "layer_glm5_kda_rms_gate_last" => Some("glm5_row_kda_rms_gate"),
        "layer_glm5_kda_o_proj_last" => Some("glm5_row_kda_o_proj"),
        _ => None,
    }
}

// Intersect an explicitly bounded global row window with the actual resident chunk.
fn window_fragment(start:usize,count:usize,max_bytes:usize,chunk_start:usize,chunk_end:usize,width:usize)->Result<Option<(usize,usize,usize,usize)>,String>{
    let end=start.checked_add(count).ok_or("window extent overflow")?;
    let stride=width.checked_mul(2).filter(|&n|n>0).ok_or("window width invalid")?;
    let total=count.checked_mul(stride).filter(|&n|n>0 && n<=max_bytes).ok_or("window exceeds explicit byte bound")?;
    let _=total;
    if chunk_start>=chunk_end{return Err("window chunk extent invalid".into());}
    let first=start.max(chunk_start);let last=end.min(chunk_end);
    if first>=last{return Ok(None);}
    let offset=(first-chunk_start).checked_mul(stride).ok_or("window local offset overflow")?;
    let bytes=(last-first).checked_mul(stride).ok_or("window fragment overflow")?;
    offset.checked_add(bytes).ok_or("window pointer extent overflow")?;
    Ok(Some((first,last,offset,bytes)))
}

#[cfg(test)]
#[test]
fn bounded_window_fragment_contract(){
    assert_eq!(window_fragment(10,13,104,0,12,4).unwrap(),Some((10,12,80,16)));
    assert_eq!(window_fragment(10,13,104,12,20,4).unwrap(),Some((12,20,0,64)));
    assert_eq!(window_fragment(10,13,104,20,30,4).unwrap(),Some((20,23,0,24)));
    assert_eq!(window_fragment(10,13,104,0,10,4).unwrap(),None);
    assert_eq!(window_fragment(10,13,104,23,30,4).unwrap(),None);
    for args in [(10,13,103,0,30,4),(usize::MAX,1,100,0,10,1),(0,1,100,1,1,1),(0,0,100,0,1,1),(0,1,100,0,1,0),(0,1,100,0,1,usize::MAX)]{
        assert!(window_fragment(args.0,args.1,args.2,args.3,args.4,args.5).is_err());
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn capture_stage(invocation: Option<&crate::matched_prefill_capture::Invocation>,
    label: &str,
    layer: usize,
    height: usize,
    chunk: usize,
    last_position: usize,
    ptr: u64,
    width: usize,
    stream: sys::CUstream,
    chunk_start:usize,
) -> Result<(), String> {
    let Some(operation) = row_operation(label) else {
        return Ok(());
    };
    let Some(invocation)=invocation else{return Ok(())};
    let window_operation=operation.replacen("glm5_row_","glm5_window_",1);
    if let Some(target)=crate::matched_prefill_capture::target(Some(invocation),&window_operation,layer,invocation.logical_tokens(),None)? {
        if last_position.checked_add(1)!=Some(height){return Err("window stage height mismatch".into());}
        let (count,max_bytes)=crate::matched_probe::prefill_window_parameters(&window_operation,layer,invocation.logical_tokens())?;
        if let Some((first,last,offset,bytes))=window_fragment(target.row,count,max_bytes,chunk_start,height,width)?{
            let pointer=ptr.checked_add(u64::try_from(offset).map_err(|_|"window offset exceeds u64")?).ok_or("window pointer overflow")?;
            pointer.checked_add(u64::try_from(bytes).map_err(|_|"window bytes exceed u64")?).ok_or("window pointer end overflow")?;
            let window_start=target.row;
            let fragment=target.child(&format!("chunk-{chunk}-rows-{first}-{last}"))?;
            fragment.device("values.bf16",pointer,bytes,stream)?;
            fragment.write("descriptor.json",&serde_json::to_vec_pretty(&json!({"operation":window_operation,"stage":label,"layer":layer,"height":invocation.logical_tokens(),"window_start":window_start,"window_rows":count,"window_max_bytes":max_bytes,"fragment_start":first,"fragment_end":last,"width":width,"dtype":"bf16","chunk_index":chunk,"chunk_start":chunk_start,"chunk_end":height,"local_row":first-chunk_start,"device_diagnostic_allocation_bytes":0})).map_err(|e|e.to_string())?)?;
        }
    }
    let digest_operation=format!("{window_operation}_sha256");
    if let Some(target)=crate::matched_prefill_capture::target(Some(invocation),&digest_operation,layer,invocation.logical_tokens(),None)? {
        if last_position.checked_add(1)!=Some(height){return Err("digest window stage height mismatch".into());}
        let (count,max_bytes)=crate::matched_probe::prefill_window_parameters(&digest_operation,layer,invocation.logical_tokens())?;
        if let Some((first,last,offset,bytes))=window_fragment(target.row,count,max_bytes,chunk_start,height,width)? {
            let pointer=ptr.checked_add(u64::try_from(offset).map_err(|_|"digest offset exceeds u64")?).ok_or("digest pointer overflow")?;
            let row_bytes=width.checked_mul(2).ok_or("digest row bytes overflow")?;
            let fragment=target.digest_child(&format!("chunk-{chunk}-rows-{first}-{last}"))?;
            let retained=fragment.device_row_sha256("rows.sha256",pointer,bytes,row_bytes,max_bytes,stream)?;
            fragment.write("descriptor.json",&serde_json::to_vec_pretty(&json!({"operation":digest_operation,"stage":label,"layer":layer,"height":invocation.logical_tokens(),"window_start":target.row,"window_rows":count,"window_max_bytes":max_bytes,"fragment_start":first,"fragment_end":last,"width":width,"dtype":"bf16","encoding":"independent SHA256 of each actual contiguous row","source_bytes_read":bytes,"retained_digest_bytes":retained,"chunk_index":chunk,"chunk_start":chunk_start,"chunk_end":height,"local_row":first-chunk_start,"device_diagnostic_allocation_bytes":0})).map_err(|e|e.to_string())?)?;
        }
    }
    let Some(target) = crate::matched_prefill_capture::target(Some(invocation),operation, layer, invocation.logical_tokens(),None)? else {
        return Ok(());
    };
    if last_position.checked_add(1) != Some(height) {return Err("selected stage last position differs from actual global height".into());}
    let rows=height.checked_sub(chunk_start).ok_or("selected stage chunk extent invalid")?;
    if target.row<chunk_start || target.row>=height{return Ok(())}
    let local_row=crate::mla_probe::chunk_row(target.row,chunk_start,rows)?;
    crate::mla_probe::row(&target,"values.bf16",ptr,local_row,width,2,stream)?;
    target.write("descriptor.json",serde_json::to_vec_pretty(&json!({"operation":operation,"stage":label,"layer":layer,"height":invocation.logical_tokens(),"chunk_end":height,"row":target.row,"width":width,"dtype":"bf16","chunk_index":chunk,"chunk_start":chunk_start,"chunk_rows":rows,"local_row":local_row,"device_diagnostic_allocation_bytes":0})).map_err(|e|e.to_string())?.as_slice())
}

fn routes(ids: &[u8], weights: &[u8], experts: usize, topk: usize) -> Result<Vec<usize>, String> {
    if ids.len() != topk.checked_mul(4).ok_or("route size overflow")? || weights.len() != ids.len()
    {
        return Err("selected routing size mismatch".into());
    }
    let mut out = Vec::with_capacity(topk);
    for (i, w) in ids.chunks_exact(4).zip(weights.chunks_exact(4)) {
        let i = i32::from_le_bytes(i.try_into().unwrap());
        let w = f32::from_le_bytes(w.try_into().unwrap());
        if i < 0 || i as usize >= experts || out.contains(&(i as usize)) || !w.is_finite() || w < 0.
        {
            return Err("invalid selected routed expert/weight; attribution prohibited".into());
        }
        out.push(i as usize);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn capture_routes(invocation: Option<&crate::matched_prefill_capture::Invocation>,
    layer: usize,
    height: usize,
    router_input: u64,
    expert_input: u64,
    router_width: usize,
    expert_width: usize,
    experts: usize,
    topk: usize,
    logits: u64,
    ids: u64,
    weights: u64,
    stream: sys::CUstream,
) -> Result<(), String> {
    let Some(target) = crate::matched_prefill_capture::target(invocation,"prefill_moe_row", layer, height,None)? else {
        return Ok(());
    };
    target.row("router-input.bf16", router_input, router_width, stream)?;
    target.row("expert-input.bf16", expert_input, expert_width, stream)?;
    copy_row(
        &target,
        "router-logits.f32",
        logits,
        experts,
        std::mem::size_of::<f32>(),
        stream,
    )?;
    copy_row(
        &target,
        "route-ids.i32",
        ids,
        topk,
        std::mem::size_of::<i32>(),
        stream,
    )?;
    copy_row(
        &target,
        "route-weights.f32",
        weights,
        topk,
        std::mem::size_of::<f32>(),
        stream,
    )?;
    let selected = routes(
        &std::fs::read(target.directory.join("route-ids.i32")).map_err(|e| e.to_string())?,
        &std::fs::read(target.directory.join("route-weights.f32")).map_err(|e| e.to_string())?,
        experts,
        topk,
    )?;
    target.write("routing.json",serde_json::to_vec_pretty(&json!({"layer":layer,"height":height,"row":target.row,"router_width":router_width,"expert_width":expert_width,"experts":experts,"topk":topk,"selected_experts":selected,"route_contract":"Actual scaled/normalized production router outputs before replication and expert dispatch","device_diagnostic_allocation_bytes":0})).map_err(|e|e.to_string())?.as_slice())
}

pub(crate) fn capture_slots(invocation: Option<&crate::matched_prefill_capture::Invocation>,
    layer: usize,
    height: usize,
    name: &str,
    ptr: u64,
    width: usize,
    topk: usize,
    stream: sys::CUstream,
) -> Result<(), String> {
    let Some(target) = crate::matched_prefill_capture::target(invocation,"prefill_moe_row", layer, height,None)? else {
        return Ok(());
    };
    if !["w1-slots.bf16", "activation-slots.bf16", "w2-slots.bf16"].contains(&name) {
        return Err("unsupported selected expert stage".into());
    }
    target.row(
        name,
        ptr,
        width
            .checked_mul(topk)
            .ok_or("selected expert slots overflow")?,
        stream,
    )
}

pub(crate) fn capture_accumulator(invocation: Option<&crate::matched_prefill_capture::Invocation>,
    layer: usize,
    height: usize,
    ptr: u64,
    width: usize,
    stream: sys::CUstream,
) -> Result<(), String> {
    let Some(target) = crate::matched_prefill_capture::target(invocation,"prefill_moe_row", layer, height,None)? else {
        return Ok(());
    };
    copy_row(
        &target,
        "routed-accumulator.f32",
        ptr,
        width,
        std::mem::size_of::<f32>(),
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn capture_weights(invocation: Option<&crate::matched_prefill_capture::Invocation>,
    layer: usize,
    height: usize,
    operation: &str,
    k: usize,
    n: usize,
    group: usize,
    expert_bits: &[u8],
    topk: usize,
    use_table: bool,
    tables: [u64; 2],
    bases: [u64; 2],
    strides: [usize; 2],
    activation: usize,
    limit: f32,
    raw_clipped: bool,
    scale_factor: f32,
    stream: sys::CUstream,
) -> Result<(), String> {
    let Some(target) = crate::matched_prefill_capture::target(invocation,"prefill_moe_row", layer, height,None)? else {
        return Ok(());
    };
    if !["w1", "w2"].contains(&operation) || k == 0 || n == 0 || group == 0 {
        return Err("invalid selected weight geometry".into());
    }
    let ids = std::fs::read(target.directory.join("route-ids.i32")).map_err(|e| e.to_string())?;
    let weights =
        std::fs::read(target.directory.join("route-weights.f32")).map_err(|e| e.to_string())?;
    let selected = routes(&ids, &weights, expert_bits.len(), topk)?;
    for expert in selected {
        let bits = expert_bits[expert];
        if ![4, 8].contains(&bits) {
            return Err("selected routed reference requires Marlin INT4/INT8".into());
        }
        let packed_bytes = k
            .checked_mul(n)
            .and_then(|v| v.checked_mul(bits as usize))
            .map(|v| v / 8)
            .ok_or("selected packed size overflow")?;
        let groups = k.div_ceil(group);
        let scale_bytes = groups
            .checked_mul(n)
            .and_then(|v| v.checked_mul(2))
            .ok_or("selected scale size overflow")?;
        let mut ptrs = [0u64; 2];
        for (index, ptr) in ptrs.iter_mut().enumerate() {
            if use_table {
                let offset = expert
                    .checked_mul(std::mem::size_of::<u64>())
                    .ok_or("selected table offset overflow")?;
                let file = format!("{operation}-expert{expert}-table{index}.u64");
                target.device(
                    &file,
                    tables[index]
                        .checked_add(offset as u64)
                        .ok_or("selected table pointer overflow")?,
                    std::mem::size_of::<u64>(),
                    stream,
                )?;
                let raw = std::fs::read(target.directory.join(file)).map_err(|e| e.to_string())?;
                *ptr = u64::from_le_bytes(
                    raw.try_into()
                        .map_err(|_| "selected pointer size mismatch")?,
                );
            } else {
                let bytes = if index == 0 {
                    packed_bytes
                } else {
                    scale_bytes
                };
                if strides[index] < bytes {
                    return Err("selected contiguous expert stride too small".into());
                }
                let offset = expert
                    .checked_mul(strides[index])
                    .ok_or("selected contiguous offset overflow")?;
                *ptr = bases[index]
                    .checked_add(offset as u64)
                    .ok_or("selected contiguous pointer overflow")?;
            }
            if *ptr == 0 {
                return Err("null selected expert pointer".into());
            }
        }
        target.device(
            &format!("{operation}-expert{expert}-packed.bin"),
            ptrs[0],
            packed_bytes,
            stream,
        )?;
        target.device(
            &format!("{operation}-expert{expert}-scales.bf16"),
            ptrs[1],
            scale_bytes,
            stream,
        )?;
        target.write(&format!("{operation}-expert{expert}-descriptor.json"),serde_json::to_vec_pretty(&json!({"layer":layer,"height":height,"row":target.row,"expert":expert,"bits":bits,"n":n,"k":k,"group_size":group,"groups":groups,"packed_bytes":packed_bytes,"scale_bytes":scale_bytes,"actual_gpu_pointers":ptrs,"pointer_table":use_table,"activation":activation,"swiglu_limit":limit,"deepseek_v4_activation":raw_clipped,"routed_scaling_factor":scale_factor,"output_slot_contract":"Fused Marlin writes C at original sorted_id = token_row*topk+slot; output is not indexed by sorted list position","device_diagnostic_allocation_bytes":0})).map_err(|e|e.to_string())?.as_slice())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selected_routes_reject_duplicate_out_of_range_and_nonfinite_values() {
        let ids = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        let weights = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        assert_eq!(
            routes(&ids(&[1, 3]), &weights(&[0.4, 0.6]), 4, 2).unwrap(),
            vec![1, 3]
        );
        for v in [&[1, 1][..], &[1, 4][..], &[-1, 3][..]] {
            assert!(routes(&ids(v), &weights(&[0.4, 0.6]), 4, 2).is_err())
        }
        assert!(routes(&ids(&[1, 3]), &weights(&[0.4, f32::NAN]), 4, 2).is_err());
        assert!(routes(&ids(&[1]), &weights(&[0.4, 0.6]), 4, 2).is_err());
    }
    #[test]
    fn selected_stage_map_is_scoped_to_observed_glm_ffn_boundaries() {
        assert_eq!(
            row_operation("layer_glm5_ffn_sublayer_last"),
            Some("glm5_row_ffn_output")
        );
        assert_eq!(row_operation("layer_other_model_output"), None);
        assert_eq!(row_operation("layer_glm5_kda_recurrent_last"), None);
    }
}

// BEGIN bounded routed W1 lane selection
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LaneSpec {
    schema_version: u32,
    layer: usize,
    row: usize,
    heights: Vec<usize>,
    expert: usize,
    coordinates: Vec<usize>,
}
fn resolve_lane_spec(spec:&LaneSpec,layer:usize,height:usize,row:usize,width:usize,ids:&[i32])->Result<(i32,Vec<i32>),String>{
    if spec.schema_version!=1||spec.layer!=layer||spec.row!=row||!spec.heights.contains(&height)||row>=height||spec.coordinates.is_empty()||spec.coordinates.len()>4||spec.coordinates.iter().any(|&x|x>=width)||spec.coordinates.iter().collect::<std::collections::HashSet<_>>().len()!=spec.coordinates.len(){return Err("invalid explicit selected W1 lane geometry/coordinates".into())}
    let expert=i32::try_from(spec.expert).map_err(|_|"lane expert exceeds ABI")?;
    let matches=ids.iter().enumerate().filter(|(_,x)|**x==expert).map(|(i,_)|i).collect::<Vec<_>>();
    if matches.len()!=1{return Err("selected lane expert absent or duplicated in actual routing".into())}
    let sorted=row.checked_mul(ids.len()).and_then(|x|x.checked_add(matches[0])).and_then(|x|i32::try_from(x).ok()).ok_or("selected lane row exceeds ABI")?;
    let coordinates=spec.coordinates.iter().map(|&x|i32::try_from(x).map_err(|_|"lane coordinate exceeds ABI".to_string())).collect::<Result<Vec<_>,_>>()?;
    Ok((sorted,coordinates))
}
pub(crate) struct LaneSelection {
    pub target: Target,
    pub sorted_row: i32,
    pub expert: i32,
    pub coordinates: Vec<i32>,
}
pub(crate) fn lane_selection(invocation: Option<&crate::matched_prefill_capture::Invocation>,layer:usize,height:usize,width:usize,topk:usize)->Result<Option<LaneSelection>,String>{
    let Some(target)=crate::matched_prefill_capture::target(invocation,"prefill_moe_lane",layer,height,None)? else{return Ok(None)};
    let source=crate::matched_prefill_capture::target(invocation,"prefill_moe_row",layer,height,None)?.ok_or("lane diagnostic requires actual selected-row routing capture")?;
    if source.row!=target.row{return Err("lane and routing selected row differ".into())}
    let file=std::env::var("KRASIS_MATCHED_MOE_LANE_CONFIG").map_err(|_|"explicit selected W1 lane config required")?;
    let raw=std::fs::read(&file).map_err(|e|e.to_string())?;
    let spec:LaneSpec=serde_json::from_slice(&raw).map_err(|e|e.to_string())?;
    let raw_ids=std::fs::read(source.directory.join("route-ids.i32")).map_err(|e|e.to_string())?;
    if raw_ids.len()!=topk.checked_mul(4).ok_or("lane routing bytes overflow")?{return Err("invalid actual lane routing length".into())}
    let ids=raw_ids.chunks_exact(4).map(|x|i32::from_le_bytes(x.try_into().unwrap())).collect::<Vec<_>>();
    let (sorted_row,coordinates)=resolve_lane_spec(&spec,layer,height,target.row,width,&ids)?;
    target.write("selection-config.json",&raw)?;
    Ok(Some(LaneSelection{target,sorted_row,expert:i32::try_from(spec.expert).map_err(|_|"lane expert exceeds ABI")?,coordinates}))
}
#[cfg(test)]
mod lane_tests{
 use super::*;
 #[test]fn selected_lane_requires_actual_unique_expert_and_valid_coordinates(){
  let mut spec=LaneSpec{schema_version:1,layer:2,row:3,heights:vec![10,20],expert:7,coordinates:vec![4]};
  assert_eq!(resolve_lane_spec(&spec,2,10,3,8,&[1,7]).unwrap(),(7,vec![4]));
  assert!(resolve_lane_spec(&spec,2,10,3,8,&[1,2]).is_err());assert!(resolve_lane_spec(&spec,2,10,3,8,&[7,7]).is_err());
  spec.coordinates=vec![8];assert!(resolve_lane_spec(&spec,2,10,3,8,&[1,7]).is_err());
  spec.coordinates=vec![4,4];assert!(resolve_lane_spec(&spec,2,10,3,8,&[1,7]).is_err());
  spec.coordinates=vec![4];assert!(resolve_lane_spec(&spec,2,11,3,8,&[1,7]).is_err());assert!(resolve_lane_spec(&spec,3,10,3,8,&[1,7]).is_err());
 }
}
// END bounded routed W1 lane selection

// BEGIN bounded same-operator partition replay
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PartitionSpec {
    schema_version: u32,
    layer: usize,
    row: usize,
    heights: Vec<usize>,
    policy: PartitionPolicy,
    samples: Option<usize>,
}
#[derive(Clone, Copy, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PartitionPolicy {
    OneBlockFullK,
    OneCtaPerMoeBlock,
}
impl PartitionPolicy {
    pub(crate) fn label(self) -> &'static str {
        match self { Self::OneBlockFullK => "single-block", Self::OneCtaPerMoeBlock => "concurrent-full-k" }
    }
    pub(crate) fn grid(self, padded: usize, block_size: usize, group_size: usize, k: usize) -> Result<i32, String> {
        // This diagnostic admits only homogeneous large-M launches. Each CTA
        // receives all N columns of one actual padded MoE block. Consequently
        // ceil(k_tiles*n_tiles*blocks/grid) == k_tiles*n_tiles, independent of
        // the selected thread N/K geometry and unrelated future routing.
        if block_size <= 16 || padded == 0 || padded % block_size != 0 || group_size == 0 || k % group_size != 0 {
            return Err("full-K replay requires complete homogeneous padded/group spans".into());
        }
        match self {
            Self::OneBlockFullK => Ok(1),
            Self::OneCtaPerMoeBlock => i32::try_from(padded / block_size).map_err(|_| "concurrent grid exceeds launch ABI".into()),
        }
    }
}
pub(crate) struct PartitionReplay {
    pub(crate) target: Target,
    pub(crate) policy: PartitionPolicy,
    pub(crate) samples: Option<usize>,
}
fn validate_partition(spec: &PartitionSpec, layer: usize, height: usize, row: usize) -> Result<(), String> {
    if spec.schema_version != 1 || spec.layer != layer || spec.row != row || row >= height || !spec.heights.contains(&height) {
        return Err("partition replay does not match its explicit selected operator".into());
    }
    if matches!(spec.policy, PartitionPolicy::OneCtaPerMoeBlock) && spec.samples.is_none() {
        return Err("concurrent replay requires explicit operator measurement samples".into());
    }
    if spec.samples.is_some_and(|n| n == 0 || n > 1024) {
        return Err("invalid bounded diagnostic sample count".into());
    }
    Ok(())
}
pub(crate) fn partition_target(invocation: Option<&crate::matched_prefill_capture::Invocation>,layer: usize, height: usize) -> Result<Option<PartitionReplay>, String> {
    let Some(target) = crate::matched_prefill_capture::target(invocation,"prefill_moe_partition", layer, height,None)? else { return Ok(None); };
    let lane = crate::matched_prefill_capture::target(invocation,"prefill_moe_lane", layer, height,None)?.ok_or("partition replay requires actual W1 lane diagnostics")?;
    if lane.row != target.row { return Err("partition and lane rows differ".into()); }
    let file = std::env::var("KRASIS_MATCHED_MOE_PARTITION_CONFIG").map_err(|_| "explicit partition replay config required")?;
    let raw = std::fs::read(file).map_err(|e| e.to_string())?;
    let spec: PartitionSpec = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    validate_partition(&spec, layer, height, target.row)?;
    target.write("partition-config.json", &raw)?;
    Ok(Some(PartitionReplay { target, policy: spec.policy, samples: spec.samples }))
}
pub(crate) struct PartitionEvents { start: sys::CUevent, stop: sys::CUevent }
impl PartitionEvents {
    pub(crate) fn new() -> Result<Self, String> {
        let mut result = Self { start: std::ptr::null_mut(), stop: std::ptr::null_mut() };
        for event in [&mut result.start, &mut result.stop] {
            let rc = unsafe { sys::lib().cuEventCreate(event, 0) };
            if rc != sys::CUresult::CUDA_SUCCESS { return Err(format!("partition event creation: {rc:?}")); }
        }
        Ok(result)
    }
    pub(crate) fn begin(&self, stream: sys::CUstream) -> Result<(), String> { self.record(self.start, stream) }
    pub(crate) fn end(&self, stream: sys::CUstream) -> Result<(), String> { self.record(self.stop, stream) }
    fn record(&self, event: sys::CUevent, stream: sys::CUstream) -> Result<(), String> {
        let rc = unsafe { sys::lib().cuEventRecord(event, stream) };
        if rc != sys::CUresult::CUDA_SUCCESS { return Err(format!("partition event record: {rc:?}")); }
        Ok(())
    }
    pub(crate) fn elapsed(&self) -> Result<f32, String> {
        let mut ms = 0.0;
        let rc = unsafe { sys::lib().cuEventElapsedTime(&mut ms, self.start, self.stop) };
        if rc != sys::CUresult::CUDA_SUCCESS || !ms.is_finite() || ms <= 0.0 { return Err(format!("invalid partition event measurement: {rc:?}, ms={ms}")); }
        Ok(ms)
    }
}
impl Drop for PartitionEvents {
    fn drop(&mut self) {
        for event in [self.start, self.stop] { if !event.is_null() { unsafe { let _ = sys::lib().cuEventDestroy_v2(event); } } }
    }
}
pub(crate) fn free_device_bytes() -> Result<usize, String> {
    let (mut free, mut total) = (0usize, 0usize);
    let rc = unsafe { sys::lib().cuMemGetInfo_v2(&mut free, &mut total) };
    if rc != sys::CUresult::CUDA_SUCCESS { return Err(format!("partition memory measurement: {rc:?}")); }
    Ok(free)
}
pub(crate) fn lane_variant(invocation: Option<&crate::matched_prefill_capture::Invocation>,layer: usize, height: usize, width: usize, topk: usize, variant: &str) -> Result<LaneSelection, String> {
    let mut selection = lane_selection(invocation,layer, height, width, topk)?.ok_or("partition replay lane selection missing")?;
    let raw = std::fs::read(selection.target.directory.join("selection-config.json")).map_err(|e| e.to_string())?;
    selection.target = selection.target.child(variant)?;
    selection.target.write("selection-config.json", &raw)?;
    Ok(selection)
}
pub(crate) fn device_digest(ptr: u64, bytes: usize, stream: sys::CUstream) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    if ptr == 0 || bytes == 0 || ptr.checked_add(u64::try_from(bytes).map_err(|_| "operand byte count exceeds pointer ABI")?).is_none() {
        return Err("invalid partition replay operand range".into());
    }
    let rc = unsafe { sys::lib().cuStreamSynchronize(stream) };
    if rc != sys::CUresult::CUDA_SUCCESS { return Err(format!("partition operand fence: {rc:?}")); }
    // Bounded host I/O buffer; this is neither a GPU allocation nor a model budget.
    let mut host = vec![0u8; bytes.min(1024 * 1024)];
    let mut digest = Sha256::new();
    let mut offset = 0usize;
    while offset < bytes {
        let count = host.len().min(bytes - offset);
        let rc = unsafe { sys::lib().cuMemcpyDtoH_v2(host.as_mut_ptr().cast(), ptr + offset as u64, count) };
        if rc != sys::CUresult::CUDA_SUCCESS { return Err(format!("partition operand download: {rc:?}")); }
        digest.update(&host[..count]);
        offset += count;
    }
    Ok(format!("{:x}", digest.finalize()))
}
#[cfg(test)]
mod partition_tests {
    use super::*;
    #[test]
    fn explicit_partition_refuses_unknown_policy_or_mismatched_operator() {
        let raw = r#"{"schema_version":1,"layer":2,"row":3,"heights":[10,20],"policy":"one_block_full_k"}"#;
        let spec: PartitionSpec = serde_json::from_str(raw).unwrap();
        assert!(validate_partition(&spec, 2, 10, 3).is_ok());
        for (layer, height, row) in [(3, 10, 3), (2, 11, 3), (2, 10, 2), (2, 3, 3)] {
            assert!(validate_partition(&spec, layer, height, row).is_err());
        }
        assert!(serde_json::from_str::<PartitionSpec>(&raw.replace("one_block_full_k", "automatic")).is_err());
        assert!(serde_json::from_str::<PartitionSpec>(&raw.replace("\"policy\"", "\"unknown\"")).is_err());
    }
    #[test]
    fn concurrent_partition_uses_actual_padded_work_without_template_constants() {
        let p = PartitionPolicy::OneCtaPerMoeBlock;
        for (padded, block, group, k) in [(960, 32, 64, 1536), (12352, 64, 128, 4096), (1440, 48, 32, 3072)] {
            assert_eq!(p.grid(padded, block, group, k).unwrap(), (padded / block) as i32);
            // For any valid template geometry the stripe covers complete K.
            for (tk, tn, n) in [(64, 256, 4096), (128, 64, 1536)] {
                let k_tiles = k / tk; let n_tiles = n / tn;
                let grid = p.grid(padded, block, group, k).unwrap() as usize;
                assert_eq!((k_tiles * n_tiles * (padded / block)).div_ceil(grid), k_tiles * n_tiles);
            }
        }
        for (padded, block, group, k) in [(0,64,128,4096),(65,64,128,4096),(64,16,128,4096),(64,64,0,4096),(64,64,128,4100)] {
            assert!(p.grid(padded, block, group, k).is_err());
        }
        assert!(p.grid(((i32::MAX as usize)+1)*64,64,128,4096).is_err());
        let raw=r#"{"schema_version":1,"layer":2,"row":3,"heights":[10],"policy":"one_cta_per_moe_block","samples":7}"#;
        assert!(validate_partition(&serde_json::from_str(raw).unwrap(),2,10,3).is_ok());
        for value in ["0", "1025", "null"] { assert!(validate_partition(&serde_json::from_str(&raw.replace("7}",&format!("{value}}}"))).unwrap(),2,10,3).is_err()); }
    }
}
// END bounded same-operator partition replay
