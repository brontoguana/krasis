//! Actual scalar-GDN storage geometry. No model names, GPU indices or budgets.
use std::{collections::BTreeMap, mem::size_of};
pub const NATIVE_BLOCK_ROWS: usize = 64;
pub const METADATA_WORDS: usize = 8;
#[derive(Debug, Clone)]
pub struct Layout {
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub maximum_new_rows: usize,
    pub padded_capacity: usize,
    pub persistent: BTreeMap<&'static str, usize>,
    pub workspace: BTreeMap<&'static str, usize>,
}
fn product(values: &[usize]) -> Result<usize, String> {
    values.iter().try_fold(1usize, |x, &y| {
        if y == 0 {
            return Err("empty scalar-GDN storage extent".into());
        }
        x.checked_mul(y)
            .ok_or_else(|| "scalar-GDN storage extent overflow".into())
    })
}
fn sum(map: &BTreeMap<&'static str, usize>) -> Result<usize, String> {
    map.values().try_fold(0usize, |n, &x| {
        n.checked_add(x)
            .ok_or_else(|| "scalar-GDN inventory overflow".into())
    })
}
impl Layout {
    pub fn checked(
        heads: usize,
        key_dim: usize,
        value_dim: usize,
        maximum_new_rows: usize,
    ) -> Result<Self, String> {
        if [heads, key_dim, value_dim, maximum_new_rows].contains(&0) {
            return Err("empty scalar-GDN dimensions".into());
        }
        // An arbitrary request can arrive with at most BT-1 unfinished rows.
        let combined = maximum_new_rows
            .checked_add(NATIVE_BLOCK_ROWS - 1)
            .ok_or("pending append overflow")?;
        let padded_capacity = combined
            .checked_add(NATIVE_BLOCK_ROWS - 1)
            .ok_or("native padded capacity overflow")?
            / NATIVE_BLOCK_ROWS
            * NATIVE_BLOCK_ROWS;
        for x in [heads, key_dim, value_dim, padded_capacity] {
            i32::try_from(x).map_err(|_| "scalar-GDN signed CUDA ABI exceeded")?;
        }
        let q_row = product(&[heads, key_dim])?;
        let v_row = product(&[heads, value_dim])?;
        // CUDA transport currently accepts signed per-row element counts.
        for n in [q_row, v_row] {
            i32::try_from(n).map_err(|_| "scalar-GDN row ABI exceeded")?;
        }
        let planes = |rows: usize| -> Result<BTreeMap<&'static str, usize>, String> {
            Ok(BTreeMap::from([
                ("q.bf16", product(&[rows, q_row, size_of::<u16>()])?),
                ("k.bf16", product(&[rows, q_row, size_of::<u16>()])?),
                ("v.bf16", product(&[rows, v_row, size_of::<u16>()])?),
                ("beta.bf16", product(&[rows, heads, size_of::<u16>()])?),
                ("gate.f32", product(&[rows, heads, size_of::<f32>()])?),
            ]))
        };
        let mut persistent = planes(NATIVE_BLOCK_ROWS)?;
        let state = product(&[heads, key_dim, value_dim, size_of::<f32>()])?;
        persistent.insert("boundary-state.f32", state);
        persistent.insert(
            "metadata.u64",
            product(&[METADATA_WORDS, size_of::<u64>()])?,
        );
        // Latest recurrent state remains the model's actual registered tensor,
        // borrowed by the owner; it is not duplicated or omitted from the
        // model registry/calibration. This map lists ONLY new owned regions.
        let mut workspace = planes(padded_capacity)?;
        workspace.extend([
            (
                "cumulative-gate.f32",
                product(&[padded_capacity, heads, size_of::<f32>()])?,
            ),
            (
                "A.f32",
                product(&[padded_capacity, heads, NATIVE_BLOCK_ROWS, size_of::<f32>()])?,
            ),
            (
                "Ai.bf16",
                product(&[padded_capacity, heads, NATIVE_BLOCK_ROWS, size_of::<u16>()])?,
            ),
            (
                "w.bf16",
                product(&[padded_capacity, q_row, size_of::<u16>()])?,
            ),
            (
                "u.bf16",
                product(&[padded_capacity, v_row, size_of::<u16>()])?,
            ),
            (
                "v-new.bf16",
                product(&[padded_capacity, v_row, size_of::<u16>()])?,
            ),
            (
                "chunk-state.bf16",
                product(&[
                    padded_capacity / NATIVE_BLOCK_ROWS,
                    heads,
                    key_dim,
                    value_dim,
                    size_of::<u16>(),
                ])?,
            ),
            (
                "output.bf16",
                product(&[padded_capacity, v_row, size_of::<u16>()])?,
            ),
            ("completed-state.f32", state),
            (
                "emitted.bf16",
                product(&[maximum_new_rows, v_row, size_of::<u16>()])?,
            ),
        ]);
        sum(&persistent)?;
        sum(&workspace)?;
        Ok(Self {
            heads,
            key_dim,
            value_dim,
            maximum_new_rows,
            padded_capacity,
            persistent,
            workspace,
        })
    }
    pub fn persistent_bytes(&self) -> Result<usize, String> {
        sum(&self.persistent)
    }
    pub fn workspace_bytes(&self) -> Result<usize, String> {
        sum(&self.workspace)
    }
    pub fn owned_bytes(&self) -> Result<usize, String> {
        self.persistent_bytes()?
            .checked_add(self.workspace_bytes()?)
            .ok_or_else(|| "scalar-GDN total bytes overflow".into())
    }
    pub fn prefill_shape(&self, start: u64, rows: usize) -> Result<(usize, usize, usize), String> {
        if rows == 0 || rows > self.maximum_new_rows {
            return Err("scalar-GDN append capacity exceeded".into());
        }
        start
            .checked_add(rows as u64)
            .ok_or("scalar-GDN cursor overflow")?;
        let pending = (start % NATIVE_BLOCK_ROWS as u64) as usize;
        let combined = pending
            .checked_add(rows)
            .ok_or("scalar-GDN combined rows overflow")?;
        let padded = combined
            .checked_add(NATIVE_BLOCK_ROWS - 1)
            .ok_or("scalar-GDN padding overflow")?
            / NATIVE_BLOCK_ROWS
            * NATIVE_BLOCK_ROWS;
        if padded > self.padded_capacity {
            return Err("scalar-GDN padded capacity exceeded".into());
        }
        Ok((pending, combined, padded))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nonsquare_value_planes_and_state_have_distinct_real_extents() {
        let l = Layout::checked(3, 2, 5, 1).unwrap();
        assert_eq!(l.persistent["q.bf16"], 768);
        assert_eq!(l.persistent["k.bf16"], 768);
        assert_eq!(l.persistent["v.bf16"], 1920);
        assert_eq!(l.persistent["beta.bf16"], 384);
        assert_eq!(l.persistent["gate.f32"], 768);
        assert_eq!(l.persistent["boundary-state.f32"], 120);
        assert_eq!(l.persistent_bytes().unwrap(), 4792);
        assert_eq!(l.padded_capacity, 64);
        assert_eq!(l.prefill_shape(63, 1).unwrap(), (63, 64, 64));
    }
    #[test]
    fn pending_extension_and_overflow_fail_before_allocation() {
        let l = Layout::checked(2, 7, 3, 65).unwrap();
        assert_eq!(l.padded_capacity, 128);
        assert_eq!(l.prefill_shape(63, 65).unwrap(), (63, 128, 128));
        assert!(l.prefill_shape(0, 66).is_err());
        assert!(l.prefill_shape(u64::MAX, 1).is_err());
        for dims in [
            (0, 1, 1, 1),
            (1, 0, 1, 1),
            (usize::MAX, 1, 1, 1),
            (1, 1, 1, usize::MAX),
        ] {
            assert!(Layout::checked(dims.0, dims.1, dims.2, dims.3).is_err());
        }
    }
}
