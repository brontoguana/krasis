#[cfg(test)]
mod live_tiling_calibration {
    use super::*;
    pub(super) fn tiled_capacity(
        heads: usize,
        dim: usize,
        rows: usize,
        free: usize,
        reserve: usize,
        device: &CudaDevice,
    ) -> Result<usize, String> {
        let pair_per_group = heads
            .checked_mul(dim)
            .and_then(|v| v.checked_mul(16 * 16))
            .ok_or("tiling index denominator")?;
        let index_groups = (i32::MAX as usize)
            .checked_div(pair_per_group)
            .ok_or("zero geometry")?;
        let grid_groups = usize::try_from(
            device
                .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
                .map_err(|e| e.to_string())?,
        )
        .map_err(|_| "grid bound")?;
        let max_rows = index_groups
            .min(grid_groups)
            .checked_mul(16)
            .and_then(|v| v.checked_sub(15))
            .ok_or("no signed ABI capacity")?;
        let available = free
            .checked_sub(reserve)
            .ok_or("configured reserve unavailable")?;
        let need = |n| -> Result<usize, String> {
            let a = PreparedLayout::checked(heads, dim, n)?.allocation_bytes;
            let b = KdaRawProducer::allocation_blueprint(heads, dim, n)?
                .2
                .checked_sub(
                    Schema::checked(RawContract::KdaBf16VectorGate, n, heads, dim, dim)?
                        .requested_bytes,
                )
                .ok_or("live producer new payload")?;
            a.checked_add(b)
                .ok_or("combined allocation blueprint".into())
        };
        if need(1)? > available {
            return Err("no safe single-row footprint; no fallback".into());
        }
        let (mut low, mut high) = (1usize, rows.min(max_rows));
        if high == 0 {
            return Err("zero request/capacity".into());
        }
        while low < high {
            let mid = low + (high - low + 1) / 2;
            if need(mid)? <= available {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        Ok(low)
    }
    pub(super) fn calibrated_allocation_probe(
        stream: Arc<NativeStream>,
        device: Arc<CudaDevice>,
        kernels: Arc<KdaRawKernels>,
        p: &LongChronologyPlan,
        capacity: usize,
        reserve: usize,
        transport: &str,
        math: &str,
        a_log: &[u8],
        bias: &[u8],
        raw: Arc<NativeRawOwner>,
        expected: &[u8],
        expected_bf: &[u8],
    ) -> Result<serde_json::Value, String> {
        let before = memory()?;
        let precision = crate::vram_monitor::begin_precision_vram_window(device.ordinal() as i32)?
            .ok_or("required native calibration precision window disabled")?;
        let owner = PreparedOwner::new_on_stream(
            device,
            stream.clone(),
            p.heads,
            p.dim,
            capacity,
            reserve,
            transport,
            math,
        )?;
        let producer = KdaRawProducer::from_live_raw_owner(
            raw, p.heads, p.dim, capacity, reserve, p.lower, p.scale, p.epsilon, kernels, a_log,
            bias,
        )?;
        let resident = memory()?;
        {
            let lease = owner.arena.lease()?;
            producer.enqueue(0, capacity, &owner)?;
            owner.prefill_in(&lease, &producer.view(capacity)?, 0, capacity)?;
        }
        stream.drain()?;
        long_check_cursor(&owner, capacity)?;
        let after_execute = memory()?;
        let endpoints_min = resident[0].min(after_execute[0]);
        let hd = p.heads.checked_mul(p.dim).ok_or("probe HD")?;
        owner_compare(
            &owner_array(&owner, "emitted", capacity * hd)?,
            &expected[..capacity * hd * 4],
            "calibration authentic complete F32 prefix",
        )?;
        owner_compare(
            &vector_bf16_array(&owner, capacity * hd)?,
            &expected_bf[..capacity * hd * 2],
            "calibration authentic complete BF16 prefix",
        )?;
        let precision_min =
            usize::try_from(precision.finish()?).map_err(|_| "precision min extent")?;
        let lowest = endpoints_min.min(precision_min);
        if lowest < reserve {
            return Err("allocation probe precision floor below configured safety".into());
        }
        let measured_bytes = before[0]
            .checked_sub(lowest)
            .ok_or("allocation probe free increased unexpectedly")?;
        let requested_bytes = owner
            .allocation_bytes
            .checked_add(producer.allocated_bytes)
            .ok_or("probe payload overflow")?;
        drop(producer);
        drop(owner);
        stream.drain()?;
        let cleanup = memory()?;
        Ok(
            json!({"capacity":capacity,"before":before,"resident":resident,"after_execute":after_execute,"lowest":lowest,"precision_min_free_bytes":precision_min,"measured_bytes":measured_bytes,"requested_bytes":requested_bytes,"post_cleanup":cleanup,"complete_prefix_outputs_exact":true}),
        )
    }
}
