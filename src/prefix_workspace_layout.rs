/// Checked element layout for the compiled fixed-block arithmetic contract.
/// Counts come from kernel planes and their registered Rust element types.
struct PrefixWorkspaceLayout {
    hd: usize,
    state: usize,
    padded: usize,
    rows: usize,
    beta_rows: usize,
    matrices: usize,
    bytes: usize,
}
impl PrefixWorkspaceLayout {
    fn checked(heads: usize, dim: usize, max_new_rows: usize) -> Result<Self, String> {
        if heads == 0 || dim == 0 || dim > 128 || max_new_rows == 0 {
            return Err("fixed16 unsupported workspace dimensions/capacity".into());
        }
        let hd = heads
            .checked_mul(dim)
            .ok_or("fixed16 workspace width overflow")?;
        let state = hd
            .checked_mul(dim)
            .ok_or("fixed16 workspace state overflow")?;
        let capacity = max_new_rows
            .checked_add(BLOCK_ROWS - 1)
            .ok_or("fixed16 prepend capacity overflow")?;
        let padded = capacity
            .div_ceil(BLOCK_ROWS)
            .checked_mul(BLOCK_ROWS)
            .ok_or("fixed16 padded rows overflow")?;
        let rows = padded
            .checked_mul(hd)
            .ok_or("fixed16 workspace rows overflow")?;
        let beta_rows = padded
            .checked_mul(heads)
            .ok_or("fixed16 workspace beta rows overflow")?;
        let matrices = (padded / BLOCK_ROWS)
            .checked_mul(heads)
            .and_then(|x| x.checked_mul(BLOCK_ROWS * BLOCK_ROWS))
            .ok_or("fixed16 workspace matrix overflow")?;
        for n in [state, rows, beta_rows, matrices] {
            i32::try_from(n).map_err(|_| "fixed16 workspace CUDA i32 extent exceeded")?;
        }
        let bf16 = 4usize
            .checked_mul(rows)
            .and_then(|x| x.checked_add(beta_rows))
            .ok_or("fixed16 workspace BF16 count overflow")?;
        let f32 = 9usize
            .checked_mul(rows)
            .and_then(|x| x.checked_add(beta_rows))
            .and_then(|x| 3usize.checked_mul(matrices).and_then(|n| x.checked_add(n)))
            .ok_or("fixed16 workspace F32 count overflow")?;
        let bytes = bf16
            .checked_mul(std::mem::size_of::<u16>())
            .and_then(|x| {
                f32.checked_mul(std::mem::size_of::<f32>())
                    .and_then(|n| x.checked_add(n))
            })
            .ok_or("fixed16 workspace byte overflow")?;
        Ok(Self {
            hd,
            state,
            padded,
            rows,
            beta_rows,
            matrices,
            bytes,
        })
    }
}
/// Pure checked batch payload, shared by the actual allocator and estimator.
pub fn workspace_bytes(heads: usize, dim: usize, max_new_rows: usize) -> Result<usize, String> {
    Ok(PrefixWorkspaceLayout::checked(heads, dim, max_new_rows)?.bytes)
}
#[cfg(test)]
mod prefix_workspace_layout_tests {
    use super::*;
    #[test]
    fn invalid_and_overflowing_layouts_reject_before_allocation() {
        for (heads, dim, rows) in [
            (0, 128, 1),
            (64, 0, 1),
            (64, 129, 1),
            (64, 128, 0),
            (usize::MAX, 128, 1),
            (64, 128, usize::MAX),
        ] {
            assert!(PrefixWorkspaceLayout::checked(heads, dim, rows).is_err());
        }
        // Exact boundary comes from the compiled signed tensor-index ABI.
        for (heads, dim) in [(1, 32), (32, 64), (64, 128)] {
            let maximum_padded = (i32::MAX as usize / (heads * dim) / BLOCK_ROWS) * BLOCK_ROWS;
            let maximum_new = maximum_padded - (BLOCK_ROWS - 1);
            assert!(PrefixWorkspaceLayout::checked(heads, dim, maximum_new).is_ok());
            assert!(PrefixWorkspaceLayout::checked(heads, dim, maximum_new + 1).is_err());
        }
    }
    #[test]
    fn capacity_covers_every_original_partial_tail_without_smaller_arena() {
        for new_rows in 1..=BLOCK_ROWS * 3 {
            let layout = PrefixWorkspaceLayout::checked(64, 128, new_rows).unwrap();
            for partial in 0..BLOCK_ROWS {
                assert!(layout.padded >= partial + new_rows);
                assert_eq!(layout.padded % BLOCK_ROWS, 0);
            }
            assert!(layout.padded < new_rows + 2 * BLOCK_ROWS);
        }
    }
}
