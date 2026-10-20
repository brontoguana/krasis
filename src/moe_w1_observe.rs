//! Opt-in readback of the actual native W1 consumption buffers. No GPU allocation.
use cudarc::driver::sys;
use serde_json::json;

#[derive(Clone, Copy)]
pub(crate) struct Geometry {
    pub topk: usize,
    pub hidden: usize,
    pub output: usize,
    pub experts: usize,
    pub sorted: usize,
    pub block: usize,
    pub group: usize,
    pub bits: usize,
    pub sms: usize,
    pub workspace: usize,
    pub temporary: usize,
    pub pointer_table: bool,
    pub mixed: bool,
    pub alternative: bool,
}

impl Geometry {
    fn sizes(self, height: usize, row: usize, maximum: usize) -> Result<[usize; 5], String> {
        let dimensions = [
            height,
            self.topk,
            self.hidden,
            self.output,
            self.experts,
            self.sorted,
            self.block,
            self.group,
            self.sms,
            self.workspace,
            self.temporary,
        ];
        if dimensions
            .iter()
            .any(|&x| x == 0 || i32::try_from(x).is_err())
            || row >= height
            || self.topk > self.experts
            || self.sorted % self.block != 0
            || self.hidden % self.group != 0
            || !matches!(self.bits, 4 | 8)
            || self.mixed
            || self.alternative
        {
            return Err(
                "W1 operand observation requires actual homogeneous native Marlin dimensions"
                    .into(),
            );
        }
        let slots = height
            .checked_mul(self.topk)
            .ok_or("W1 slot extent overflow")?;
        if self.sorted < slots || i32::try_from(slots).is_err() {
            return Err("invalid W1 sorted/padded slot extent".into());
        }
        let a = slots
            .checked_mul(self.hidden)
            .and_then(|x| x.checked_mul(2))
            .ok_or("W1 actual input extent overflow")?;
        let sorted = self
            .sorted
            .checked_mul(4)
            .ok_or("W1 sorted extent overflow")?;
        let blocks = (self.sorted / self.block)
            .checked_mul(4)
            .ok_or("W1 blocks extent overflow")?;
        let weights = slots.checked_mul(4).ok_or("W1 route extent overflow")?;
        let sizes = [a, sorted, blocks, 4, weights];
        let total = sizes
            .iter()
            .try_fold(0usize, |sum, &n| sum.checked_add(n))
            .ok_or("W1 total capture extent overflow")?;
        if total > maximum {
            return Err("W1 operands exceed explicit maximum_bytes".into());
        }
        Ok(sizes)
    }
}

pub(crate) fn capture(
    invocation: Option<&crate::matched_prefill_capture::Invocation>,
    layer: usize,
    height: usize,
    geometry: Geometry,
    pointers: [u64; 5],
    stream: sys::CUstream,
) -> Result<(), String> {
    const OP: &str = "prefill_moe_w1_operands";
    let Some(target) = crate::matched_prefill_capture::target(invocation, OP, layer, height, None)?
    else {
        return Ok(());
    };
    let maximum = crate::matched_probe::capture_maximum_bytes(OP, layer, Some(height))?;
    let sizes = geometry.sizes(height, target.row, maximum)?;
    // Validate every range before any readback or output. All producers are on this stream.
    for (&pointer, &bytes) in pointers.iter().zip(&sizes) {
        if pointer == 0
            || pointer
                .checked_add(u64::try_from(bytes).map_err(|_| "W1 pointer size exceeds ABI")?)
                .is_none()
        {
            return Err("invalid actual W1 operand pointer range".into());
        }
    }
    let names = [
        "actual-A.bf16",
        "sorted-ids.i32",
        "expert-block-ids.i32",
        "post-count.i32",
        "topk-weights.f32",
    ];
    for ((name, &pointer), &bytes) in names.iter().zip(&pointers).zip(&sizes) {
        target.device(name, pointer, bytes, stream)?;
    }
    target.write("dispatch.json", &serde_json::to_vec_pretty(&json!({
        "layer":layer,"height":height,"row":target.row,
        "actual_input_layout":"row-major [height * original_topk, hidden]; already replicated",
        "original_topk":geometry.topk,"top_k_argument":1,"mul_topk_weights":false,
        "is_ep":false,"size_m":height*geometry.topk,"size_n":geometry.output,"size_k":geometry.hidden,
        "experts":geometry.experts,"actual_padded_sorted_count":geometry.sorted,"moe_block_size":geometry.block,
        "group_size":geometry.group,"num_groups":geometry.hidden/geometry.group,"bits":geometry.bits,
        "sms_argument":geometry.sms,"thread_k_argument":-1,"thread_n_argument":-1,
        "workspace_u32":geometry.workspace,"C_tmp_f32":geometry.temporary,"use_ptr_table":geometry.pointer_table,
        "has_bias":false,"has_act_order":false,"is_k_full":true,"has_zp":false,
        "use_atomic":false,"fp32_reduce":true,"is_zp_float":false,
        "stream_pointer":format!("0x{:x}",stream as usize),
        "pointers":pointers.map(|p|format!("0x{p:x}")),"tensor_bytes":sizes,
        "maximum_tensor_bytes":maximum,"device_diagnostic_allocation_bytes":0,
        "scope":"Passive actual native W1 consumption readback. Geometry contains actual host ABI arguments, not inferred CUDA template selection. Numerical weights remain separately captured by the selected-row observer. No replay, forcing, timing or serving arithmetic change."
    })).map_err(|e|e.to_string())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn moe_w1_observation_bounds_actual_replicated_shape() {
        // Non-production dimensions exercise layout/bounds, not a hardware budget.
        let g = Geometry {
            topk: 3,
            hidden: 16,
            output: 8,
            experts: 5,
            sorted: 32,
            block: 8,
            group: 4,
            bits: 4,
            sms: 2,
            workspace: 4,
            temporary: 64,
            pointer_table: true,
            mixed: false,
            alternative: false,
        };
        assert_eq!(g.sizes(7, 6, 904).unwrap(), [672, 128, 16, 4, 84]);
        assert!(g.sizes(7, 6, 903).is_err());
        assert!(g.sizes(7, 7, 1000).is_err());
        assert!(g.sizes(usize::MAX, 0, usize::MAX).is_err());
        assert!(Geometry { sorted: 16, ..g }.sizes(7, 0, 1000).is_err());
        assert!(Geometry { block: 3, ..g }.sizes(7, 0, 1000).is_err());
        assert!(Geometry { group: 3, ..g }.sizes(7, 0, 1000).is_err());
        assert!(Geometry { mixed: true, ..g }.sizes(7, 0, 1000).is_err());
        assert!(Geometry {
            alternative: true,
            ..g
        }
        .sizes(7, 0, 1000)
        .is_err());
    }
}
