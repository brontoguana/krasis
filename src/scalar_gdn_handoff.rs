// Included only in the explicit native64 build. The complete family travels
// together; raw descriptors are backed by both retained actual Store owners.
fn native_scalar_region(region: &crate::session_cache::SequenceStateAllocation) -> bool {
    region.kind.starts_with("scalar_gdn_native64_")
        || matches!(
            region.kind.as_str(),
            "linear_attention_conv_state" | "linear_attention_recurrent_state"
        )
}
fn native_scalar_handoff_plan(
    source: &[crate::session_cache::SequenceStateAllocation],
    destination: &[crate::session_cache::SequenceStateAllocation],
    start: usize,
    end: usize,
) -> Result<
    Vec<(
        crate::session_cache::SequenceStateAllocation,
        crate::session_cache::SequenceStateAllocation,
    )>,
    String,
> {
    use std::collections::BTreeMap;
    if start >= end {
        return Err("scalar handoff requires a nonempty actual layer range".into());
    }
    let key = |r: &crate::session_cache::SequenceStateAllocation| (r.layer_idx, r.kind.clone());
    let select=|regions:&[crate::session_cache::SequenceStateAllocation]|->Result<BTreeMap<_,_>,String>{
        let mut map=BTreeMap::new();
        for region in regions.iter().filter(|r|native_scalar_region(r)){
            if let Some(layer)=region.layer_idx {if layer<start||layer>=end{continue;}}
            region.validate()?;
            if map.insert(key(region),region.clone()).is_some(){return Err("scalar handoff has duplicate family regions".into());}
        }
        Ok(map)
    };
    let source = select(source)?;
    let destination = select(destination)?;
    if source.keys().ne(destination.keys()) {
        return Err("scalar handoff complete source/destination inventories differ".into());
    }
    let mut plan = Vec::with_capacity(destination.len());
    for (key, to) in destination {
        let from = &source[&key];
        if from.storage_bytes != to.storage_bytes
            || from.dtype != to.dtype
            || from.element_size != to.element_size
            || from.shape != to.shape
            || from.strides_bytes != to.strides_bytes
            || from.growth != to.growth
        {
            return Err(format!(
                "scalar handoff actual region layout differs:{key:?}"
            ));
        }
        plan.push((from.clone(), to));
    }
    Ok(plan)
}
impl GpuDecodeStore {
    fn copy_native_scalar_states_to_aux(
        &self,
        aux: &mut Self,
        start: usize,
        end: usize,
    ) -> Result<usize, String> {
        if !self.scalar_gdn_requested || !aux.scalar_gdn_requested {
            return Err("scalar handoff requires matching explicit execution modes".into());
        }
        let Some(destination) = aux.selected_scalar_gdn.clone() else {
            let has_layer = self.sequence_state_registry.allocations().iter().any(|r| {
                r.kind == "linear_attention_recurrent_state"
                    && r.layer_idx.is_some_and(|l| l >= start && l < end)
            });
            return if has_layer {
                Err("scalar handoff destination is missing actual native state".into())
            } else {
                Ok(0)
            };
        };
        let source = self
            .selected_scalar_gdn
            .clone()
            .ok_or("scalar handoff source native state absent")?;
        if destination
            .registry()
            .allocations()
            .iter()
            .any(|r| r.layer_idx.is_some_and(|l| l < start || l >= end))
        {
            return Err("scalar handoff must cover every actual destination scalar layer".into());
        }
        let plan = native_scalar_handoff_plan(
            self.sequence_state_registry.allocations(),
            aux.sequence_state_registry.allocations(),
            start,
            end,
        )?;
        let bytes = plan.iter().try_fold(0usize, |sum, (r, _)| {
            sum.checked_add(r.storage_bytes)
                .ok_or("scalar handoff byte count overflow")
        })?;
        let largest = plan
            .iter()
            .map(|(r, _)| r.storage_bytes)
            .max()
            .ok_or("scalar handoff family is empty")?;
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        let staging = SequenceStatePinnedBuffer::new(largest)?;
        let cursor = self
            .graph
            .as_ref()
            .ok_or("scalar source graph absent")?
            .kv_current_pos;
        // Both execution domains drain before copy and remain poisoned on any
        // failure/unwind. Each real async copy retains this pinned allocation,
        // both device owners and their streams until completion even on error.
        unsafe {
            source.quiescent(cursor, || {
                destination.restore(cursor, || {
                    for (from, to) in &plan {
                        let retained = (&*self, &*aux, &staging, &source, &destination);
                        crate::sequence_copy_completion::with_copy_completion(
                            &self.device,
                            self.copy_stream.0,
                            &retained,
                            |_| {
                                cuda_sys::lib()
                                    .cuMemcpyDtoHAsync_v2(
                                        staging.ptr.cast(),
                                        from.ptr,
                                        from.storage_bytes,
                                        self.copy_stream.0,
                                    )
                                    .result()
                                    .map_err(|e| format!("scalar handoff D2H {}:{e}", from.name))
                            },
                        )?;
                        crate::sequence_copy_completion::with_copy_completion(
                            &aux.device,
                            aux.copy_stream.0,
                            &retained,
                            |_| {
                                cuda_sys::lib()
                                    .cuMemcpyHtoDAsync_v2(
                                        to.ptr,
                                        staging.ptr.cast(),
                                        to.storage_bytes,
                                        aux.copy_stream.0,
                                    )
                                    .result()
                                    .map_err(|e| format!("scalar handoff H2D {}:{e}", to.name))?;
                                if to.layer_idx.is_some() {
                                    crate::publication_failure_control::after_auxiliary_copy(
                                        "native-la",
                                    )?;
                                }
                                Ok(())
                            },
                        )?;
                    }
                    Ok(())
                })
            })?;
        }
        aux.graph
            .as_mut()
            .ok_or("scalar destination graph absent")?
            .kv_current_pos = cursor;
        log::info!("native scalar handoff: actual layers [{start},{end}), {bytes} bytes including complete pending/boundary/latest/convolution/health state");
        Ok(bytes)
    }
}
#[cfg(test)]
mod scalar_gdn_handoff_tests {
    use super::*;
    use crate::session_cache::{SequenceStateAllocation as Region, SequenceStateGrowth};
    fn region(device: usize, layer: Option<usize>, kind: &str) -> Region {
        Region {
            name: format!("device{device}.{layer:?}.{kind}"),
            kind: kind.into(),
            layer_idx: layer,
            device_ordinal: device,
            ptr: 0x1000,
            storage_bytes: 16,
            dtype: "float32".into(),
            element_size: 4,
            shape: vec![4],
            strides_bytes: vec![4],
            growth: SequenceStateGrowth::Fixed,
        }
    }
    #[test]
    fn actual_layout_and_inventory_are_checked_before_copy() {
        let source = vec![
            region(3, Some(7), "linear_attention_recurrent_state"),
            region(3, Some(7), "scalar_gdn_native64_gate.f32"),
            region(3, None, "scalar_gdn_native64_health"),
        ];
        let destination = source
            .iter()
            .map(|r| region(5, r.layer_idx, &r.kind))
            .collect::<Vec<_>>();
        let plan = native_scalar_handoff_plan(&source, &destination, 7, 8).unwrap();
        assert_eq!(plan.len(), source.len());
        assert_eq!(plan[0].0.device_ordinal, 3);
        assert_eq!(plan[0].1.device_ordinal, 5);
        assert!(native_scalar_handoff_plan(&source, &destination[..2], 7, 8).is_err());
        let mut changed = destination.clone();
        changed[1].shape = vec![2, 2];
        changed[1].strides_bytes = vec![8, 4];
        assert!(native_scalar_handoff_plan(&source, &changed, 7, 8).is_err());
        let mut duplicate = destination.clone();
        duplicate.push(destination[1].clone());
        assert!(native_scalar_handoff_plan(&source, &duplicate, 7, 8).is_err());
        assert!(native_scalar_handoff_plan(&source, &destination, 8, 7).is_err());
    }
    #[test]
    fn unrelated_layers_and_other_architecture_regions_are_excluded() {
        let source = vec![
            region(2, Some(4), "scalar_gdn_native64_gate.f32"),
            region(2, Some(9), "scalar_gdn_native64_gate.f32"),
            region(2, Some(4), "attention_kv_cache"),
        ];
        let destination = vec![region(6, Some(4), "scalar_gdn_native64_gate.f32")];
        let plan = native_scalar_handoff_plan(&source, &destination, 4, 5).unwrap();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].0.layer_idx, Some(4));
    }
}
