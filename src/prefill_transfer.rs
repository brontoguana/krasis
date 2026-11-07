//! Pure scheduling and ownership bounds used by the prefill transfer planner.
//! Compile with `rustc --edition 2021 --test src/prefill_transfer.rs` for CPU checks.

pub(crate) fn logical_chunks(tokens: usize, capacity: usize) -> Vec<usize> {
    assert!(capacity > 0);
    let mut chunks = vec![capacity; tokens / capacity];
    if tokens % capacity != 0 { chunks.push(tokens % capacity); }
    chunks
}

pub(crate) fn fixed_expert_slot(expert: usize, capacity: usize) -> Result<usize, String> {
    if expert < capacity { Ok(expert) }
    else { Err(format!("band expert slot {expert} exceeds slab capacity {capacity}")) }
}

/// Traffic model, not future route knowledge. It charges restoration and limits
/// tail demand to at most topk * tail rows. Residency savings cannot exceed demand.
pub(crate) fn traffic_score(tokens: usize, rows: usize, experts: usize, topk: usize,
    expert_bytes: usize, layers: usize, resident_bytes: usize, kept_bytes: usize,
    band_rows: usize) -> (usize, usize)
{
    assert!(rows > 0 && band_rows > 0 && kept_bytes <= resident_bytes);
    let full = tokens / rows;
    let tail = tokens % rows;
    let demand = full.saturating_mul(experts).saturating_add(
        experts.min(tail.saturating_mul(topk)))
        .saturating_mul(expert_bytes).saturating_mul(layers);
    // For a cheap tail, charge only its predicted share of the retained pool;
    // counting the complete pool again would overstate expected reuse.
    let tail_experts = experts.min(tail.saturating_mul(topk));
    let saved = kept_bytes.saturating_mul(full).saturating_add(
        kept_bytes.saturating_mul(tail_experts) / experts.max(1)).min(demand);
    let traffic = demand.saturating_sub(saved).saturating_add(resident_bytes-kept_bytes);
    let bands = full.saturating_mul(rows.div_ceil(band_rows))
        .saturating_add(tail.div_ceil(band_rows));
    (traffic, bands)
}

/// Account only for runtime memory that has not already been allocated.
/// In particular a cold slab retained from a previous chunk is owned memory,
/// not a second future allocation to fund by evicting the resident pool.
pub(crate) fn live_runtime_floor(safety: usize, runtime_reserve: usize, owned_cold: usize) -> usize {
    safety.saturating_add(runtime_reserve.saturating_sub(owned_cold))
}

/// Fixed scratch/workspace requirement leaves every recovered byte available
/// for the current resident pool, capped by genuinely owned representation bytes.
pub(crate) fn protected_resident_bytes(available: usize, required: usize, owned: usize) -> usize {
    owned.min(available.saturating_sub(required))
}

/// Fixed offsets allow phase swaps without replacing registered pointers.
/// Accept only layouts that reduce physical mappings in every phase.
pub(crate) fn fixed_component_arena(prefill: [usize; 3], decode: [usize; 3],
    granularity: usize) -> Result<Option<([usize; 3], usize)>, String> {
    if granularity == 0 { return Err("zero mapping granularity".into()); }
    let align = |n: usize, unit: usize| n.max(1).checked_add(unit - 1)
        .map(|n| n / unit * unit).ok_or_else(|| "arena extent overflow".to_string());
    let reserve: [usize; 3] = std::array::from_fn(|i| prefill[i].max(decode[i]));
    let offsets = [0, align(reserve[0],512)?, align(reserve[0],512)?
        .checked_add(align(reserve[1],512)?).ok_or("arena extent overflow")?];
    let total = offsets[2].checked_add(align(reserve[2],512)?).ok_or("arena extent overflow")?;
    let independent = |sizes: [usize; 3]| -> Result<usize, String> {
        sizes.into_iter().try_fold(0usize, |sum,n| sum.checked_add(align(n,granularity)?)
            .ok_or_else(|| "arena extent overflow".into()))
    };
    Ok((align(total,granularity)? < independent(prefill)?.min(independent(decode)?))
        .then_some((offsets,total)))
}

/// Each physical allocation is max(attention, FFN); distinct pairs remain
/// distinct allocations. This returns the saved element count, not bytes.
pub(crate) fn phase_pair_savings(attention:[usize;4],ffn:[usize;4]) -> usize {
    attention.into_iter().zip(ffn).fold(0usize,|sum,(a,b)|sum.saturating_add(a.min(b)))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn phase_pairs_save_only_the_nonoverlapping_phase_minimum() {
        let a=[20,30,40,50];let b=[50,40,30,20];
        let independent:usize=a.into_iter().chain(b).sum();
        let physical:usize=a.into_iter().zip(b).map(|(x,y)|x.max(y)).sum();
        assert_eq!(phase_pair_savings(a,b),100);
        assert_eq!(independent-physical,100);
        assert_eq!(phase_pair_savings([0;4],b),0);
    }

    #[test]
    fn fixed_arena_offsets_fit_both_phases_and_save_mappings() {
        let phases = ([4096,100,100],[4000,120,100]);
        let (offsets,total) = fixed_component_arena(phases.0,phases.1,65536).unwrap().unwrap();
        assert_eq!(offsets,[0,4096,4608]);
        assert_eq!(total,5120);
        for shape in [phases.0,phases.1] {
            assert!(shape[0]<=offsets[1]);
            assert!(offsets[1]+shape[1]<=offsets[2]);
            assert!(offsets[2]+shape[2]<=total);
        }
        assert_eq!(fixed_component_arena([65536;3],[65536;3],65536).unwrap(),None);
        assert_eq!(fixed_component_arena([1000000,100,100],[1,1,1],65536).unwrap(),None);
        assert!(fixed_component_arena([usize::MAX;3],[1;3],65536).is_err());
        assert!(fixed_component_arena([1;3],[1;3],0).is_err());
    }

    #[test]
    fn recovered_capacity_funds_residents_without_expanding_workspace() {
        let required = 20_000;
        assert_eq!(protected_resident_bytes(21_000, required, 30_000), 1_000);
        assert_eq!(protected_resident_bytes(23_000, required, 32_000), 3_000);
        assert_eq!(protected_resident_bytes(10_000, required, 32_000), 0);
        assert_eq!(protected_resident_bytes(60_000, required, 32_000), 32_000);
    }

    #[test]
    fn retained_slab_is_not_budgeted_twice_between_chunks() {
        let mib=1024*1024;
        let before=live_runtime_floor(600*mib,3728*mib,0);
        let after=live_runtime_floor(600*mib,3728*mib,3564*mib);
        assert_eq!(before,4328*mib);
        assert_eq!(after,764*mib);
        assert_eq!(after+3564*mib,before);
        assert_eq!(live_runtime_floor(600*mib,3728*mib,4000*mib),600*mib);
    }

    #[test]
    fn logical_chunk_tail_preserves_real_rows_and_capacity() {
        assert_eq!(logical_chunks(65549,32768),vec![32768,32768,13]);
        for capacity in [128,8192,24576,32768] {
            for tokens in [0,1,127,128,8191,8192,32768,65549,131072] {
                let plan=logical_chunks(tokens,capacity);
                assert_eq!(plan.iter().sum::<usize>(),tokens);
                assert!(plan.iter().all(|&rows| rows>0 && rows<=capacity));
            }
        }
    }
    #[test]
    fn fixed_slots_do_not_overwrite_previous_band_experts() {
        let mut slots=[None;8];
        for band in [&[7,1,4][..],&[4,0,7,3][..],&[6,2,5][..]] {
            for &expert in band {
                let slot=fixed_expert_slot(expert,slots.len()).unwrap();
                assert!(slots[slot].is_none()||slots[slot]==Some(expert));
                slots[slot]=Some(expert);
            }
        }
        assert_eq!(slots,[Some(0),Some(1),Some(2),Some(3),Some(4),Some(5),Some(6),Some(7)]);
        assert!(fixed_expert_slot(8,8).is_err());
    }
    #[test]
    fn cheaper_plan_can_keep_residency_instead_of_largest_chunk() {
        let smaller=traffic_score(64,32,8,2,10,2,100,100,8);
        let larger=traffic_score(64,64,8,2,10,2,100,0,8);
        assert!(smaller<larger,"{smaller:?} vs {larger:?}");
    }
    #[test]
    fn tiny_tail_is_charged_for_actual_upper_bound_and_restoration() {
        assert_eq!(traffic_score(65,32,8,2,10,2,0,0,8),(360,9));
        assert_eq!(traffic_score(65,32,8,2,10,2,100,0,8),(460,9));
        assert_eq!(traffic_score(0,32,8,2,10,2,0,0,8),(0,0));
    }
}
