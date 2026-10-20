//! Bounded read-only observations of the native scalar-gate GDN/FLA path.
//! No recurrence substitution: retain actual operands before scratch reuse.
use crate::matched_probe::Target;
use cudarc::driver::sys;

#[derive(Clone, Copy, serde::Serialize)]
pub(crate) struct Geometry {
    pub real_rows: usize,
    pub padded_rows: usize,
    pub block_rows: usize,
    pub absolute_start: usize,
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

fn product(xs: &[usize]) -> Result<usize, String> {
    xs.iter().try_fold(1usize, |a, &b| {
        if b == 0 {
            return Err("GDN observation has empty dimension".into());
        }
        a.checked_mul(b)
            .ok_or("GDN observation extent overflow".into())
    })
}

impl Geometry {
    // Complete on-disk bundle bound, including inputs, cumsum, output and both
    // state endpoints. This is computed from loaded dimensions, never a budget.
    pub(crate) fn extents(self) -> Result<[usize; 6], String> {
        if self.real_rows == 0
            || self.block_rows == 0
            || self.padded_rows < self.real_rows
            || self.padded_rows % self.block_rows != 0
            || self.padded_rows - self.real_rows >= self.block_rows
            || self.absolute_start.checked_add(self.real_rows).is_none()
        {
            return Err("GDN observation real/padded row contract invalid".into());
        }
        Ok([
            product(&[self.padded_rows, self.heads, self.key_dim, 2])?,
            product(&[self.padded_rows, self.heads, self.value_dim, 2])?,
            product(&[self.padded_rows, self.heads, 2])?,
            product(&[self.padded_rows, self.heads, 4])?,
            product(&[self.heads, self.key_dim, self.value_dim, 4])?,
            product(&[self.real_rows, self.heads, self.value_dim, 2])?,
        ])
    }

    pub(crate) fn bundle_bytes(self) -> Result<usize, String> {
        let [q, v, b, g, s, o] = self.extents()?;
        let ba = product(&[self.real_rows, self.heads, 2])?;
        let parameter = product(&[self.heads, 4])?;
        [q, q, v, b, g, s, g, o, s, ba, ba, parameter, parameter]
            .into_iter()
            .try_fold(0usize, |a, b| {
                a.checked_add(b)
                    .ok_or("GDN observation bundle overflow".into())
            })
    }

    fn admit(self, maximum_bytes: usize) -> Result<[usize; 6], String> {
        if maximum_bytes == 0 || self.bundle_bytes()? > maximum_bytes {
            return Err("GDN observation exceeds explicit bundle byte bound".into());
        }
        self.extents()
    }

    pub(crate) fn inputs(
        self,
        target: &Target,
        maximum_bytes: usize,
        q: u64,
        k: u64,
        v: u64,
        beta: u64,
        gate: u64,
        h0: u64,
        a: u64,
        b: u64,
        a_log: u64,
        dt_bias: u64,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        let [qb, vb, bb, gb, sb, _] = self.admit(maximum_bytes)?;
        let ba = product(&[self.real_rows, self.heads, 2])?;
        let parameter = product(&[self.heads, 4])?;
        let operands = [
            ("q.bf16", q, qb),
            ("k.bf16", k, qb),
            ("v.bf16", v, vb),
            ("beta.bf16", beta, bb),
            ("gate.f32", gate, gb),
            ("initial-state.f32", h0, sb),
            ("a.bf16", a, ba),
            ("b.bf16", b, ba),
            ("a-log.f32", a_log, parameter),
            ("dt-bias.f32", dt_bias, parameter),
        ];
        for (_, ptr, bytes) in operands {
            if ptr == 0 || ptr.checked_add(bytes as u64).is_none() {
                return Err("GDN observation null/overflowed operand pointer".into());
            }
        }
        for (name, ptr, bytes) in operands {
            target.device(name, ptr, bytes, stream)?;
        }
        target.write("inputs.json", &serde_json::to_vec_pretty(&serde_json::json!({
            "geometry":self,"bundle_bytes":self.bundle_bytes()?,"maximum_bytes":maximum_bytes,
            "qkv_layout":"token,head,dimension; Q is already scaled",
            "gate_beta_layout":"token,head; gate is raw FP32 BEFORE transpose/cumsum",
            "state_layout":"head,key,value; native state_recurrence ABI",
            "padding":"rows real_rows..padded_rows are zero input padding, not history",
            "output_scale":1.0f32,"diagnostic_gpu_allocation_bytes":0,
            "scope":"Actual native FLA inputs; borrowed owners stay live through synchronized download"
        })).map_err(|e|e.to_string())?)
    }

    pub(crate) fn cumulative(
        self,
        target: &Target,
        maximum_bytes: usize,
        gate: u64,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        let bytes = self.admit(maximum_bytes)?[3];
        if gate == 0 || gate.checked_add(bytes as u64).is_none() {
            return Err("invalid GDN cumulative pointer".into());
        }
        target.device("cumulative-gate.f32", gate, bytes, stream)
    }

    pub(crate) fn outputs(
        self,
        target: &Target,
        maximum_bytes: usize,
        output: u64,
        final_state: u64,
        stream: sys::CUstream,
    ) -> Result<(), String> {
        let [_, _, _, _, sb, ob] = self.admit(maximum_bytes)?;
        if output == 0
            || final_state == 0
            || output.checked_add(ob as u64).is_none()
            || final_state.checked_add(sb as u64).is_none()
        {
            return Err("invalid GDN output/state pointer".into());
        }
        target.device("output.bf16", output, ob, stream)?;
        target.device("final-state.f32", final_state, sb, stream)?;
        target.write("complete.json", &serde_json::to_vec_pretty(&serde_json::json!({
            "geometry":self,"output_rows":self.real_rows,"final_state_after_padded_rows":self.padded_rows,
            "scope":"Native FLA output and FP32 final state before persistent-state copyback",
            "diagnostic_gpu_allocation_bytes":0
        })).map_err(|e|e.to_string())?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gdn_observation_bounds_follow_actual_geometry() {
        let g = Geometry {
            real_rows: 7,
            padded_rows: 8,
            block_rows: 4,
            absolute_start: 19,
            heads: 3,
            key_dim: 5,
            value_dim: 6,
        };
        assert_eq!(g.extents().unwrap(), [240, 288, 48, 96, 360, 252]);
        assert_eq!(g.bundle_bytes().unwrap(), 2088);
        assert!(g.admit(2088).is_ok());
        assert!(g.admit(2087).is_err());
        assert!(g.admit(0).is_err());
        for bad in [
            Geometry { real_rows: 0, ..g },
            Geometry { block_rows: 0, ..g },
            Geometry {
                padded_rows: 12,
                ..g
            },
            Geometry {
                padded_rows: 7,
                ..g
            },
            Geometry { heads: 0, ..g },
            Geometry {
                key_dim: usize::MAX,
                ..g
            },
            Geometry {
                absolute_start: usize::MAX,
                ..g
            },
        ] {
            assert!(bad.extents().is_err());
        }
    }
}
