use cudarc::driver::sys as cuda;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Geometry {
    pub input: usize,
    pub intermediate: usize,
    pub topk: usize,
}

impl Geometry {
    fn strides(self) -> Result<[usize; 4], String> {
        if self.input == 0 || self.intermediate == 0 || self.topk == 0 {
            return Err("bounded capture requires nonzero tensor dimensions".into());
        }
        let mul = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or("capture geometry overflow".to_string())
        };
        Ok([
            mul(self.input, 2)?,
            mul(self.topk, 4)?,
            mul(self.topk, 4)?,
            mul(mul(self.topk, self.intermediate)?, 2)?,
        ])
    }
}

#[derive(Debug)]
struct Layout {
    offsets: [usize; 4],
    lengths: [usize; 4],
    bytes: usize,
}

impl Layout {
    fn new(geometry: Geometry, rows: usize) -> Result<Self, String> {
        let strides = geometry.strides()?;
        let mut offsets = [0; 4];
        let mut lengths = [0; 4];
        let mut bytes = 0usize;
        for index in 0..4 {
            // Every segment starts at a 4-byte boundary, including odd BF16 widths.
            bytes = bytes.checked_add(3).ok_or("capture layout overflow")? & !3;
            offsets[index] = bytes;
            lengths[index] = strides[index]
                .checked_mul(rows)
                .ok_or("capture layout overflow")?;
            bytes = bytes
                .checked_add(lengths[index])
                .ok_or("capture layout overflow")?;
        }
        Ok(Self {
            offsets,
            lengths,
            bytes,
        })
    }
}

#[derive(Debug)]
pub(crate) struct Batch<'a> {
    pub inputs: &'a [u16],
    pub ids: &'a [i32],
    pub weights: &'a [f32],
    pub down: &'a [u16],
}

pub(crate) struct CaptureWindow {
    // u64 storage supplies alignment for all four tensor types. Allocation is
    // retained across windows, layers and requests; no GPU allocation is added.
    storage: Vec<u64>,
}

impl CaptureWindow {
    pub fn new(bytes: usize, available: u64) -> Result<Self, String> {
        let words = bytes / std::mem::size_of::<u64>();
        let allocated = words * std::mem::size_of::<u64>();
        if words == 0 || allocated as u64 > available {
            return Err(format!(
                "capture window {bytes} bytes exceeds available RAM {available} or is too small"
            ));
        }
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(words)
            .map_err(|e| format!("capture staging allocation failed: {e}"))?;
        storage.resize(words, 0);
        Ok(Self { storage })
    }

    pub fn rows(&self, geometry: Geometry, limit: usize) -> Result<usize, String> {
        let mut low = 0;
        let mut high = limit.min(u32::MAX as usize);
        while low < high {
            let mid = low + (high - low) / 2 + 1;
            if Layout::new(geometry, mid)?.bytes <= self.bytes() {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        if low == 0 {
            return Err(format!(
                "capture window {} bytes cannot fit one row of {geometry:?}",
                self.bytes()
            ));
        }
        Ok(low)
    }

    pub fn bytes(&self) -> usize {
        self.storage.len() * std::mem::size_of::<u64>()
    }

    pub fn copy_device(
        &mut self,
        geometry: Geometry,
        rows: usize,
        offset: usize,
        pointers: [u64; 4],
    ) -> Result<Batch<'_>, String> {
        self.copy_with(geometry, rows, offset, pointers, |dest, src| {
            let result =
                unsafe { cuda::lib().cuMemcpyDtoH_v2(dest.as_mut_ptr().cast(), src, dest.len()) };
            if result != cuda::CUresult::CUDA_SUCCESS {
                return Err(format!("bounded capture D2H failed: {result:?}"));
            }
            Ok(())
        })
    }

    fn copy_with(
        &mut self,
        geometry: Geometry,
        rows: usize,
        offset: usize,
        pointers: [u64; 4],
        mut copy: impl FnMut(&mut [u8], u64) -> Result<(), String>,
    ) -> Result<Batch<'_>, String> {
        let layout = Layout::new(geometry, rows)?;
        if rows == 0 || layout.bytes > self.bytes() {
            return Err("capture window copy exceeds staging capacity".into());
        }
        let strides = geometry.strides()?;
        let byte_len = self.bytes();
        // Exclusive access to initialized, aligned storage, bounded by layout.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(self.storage.as_mut_ptr().cast::<u8>(), byte_len)
        };
        for index in 0..4 {
            let source_offset = offset
                .checked_mul(strides[index])
                .ok_or("capture source offset overflow")?;
            let source = pointers[index]
                .checked_add(u64::try_from(source_offset).map_err(|_| "capture pointer overflow")?)
                .ok_or("capture pointer overflow")?;
            source
                .checked_add(layout.lengths[index] as u64)
                .ok_or("capture pointer end overflow")?;
            copy(
                &mut bytes[layout.offsets[index]..layout.offsets[index] + layout.lengths[index]],
                source,
            )?;
        }
        // Layout aligns each offset to 4 bytes; lengths are multiples of the
        // corresponding element size. All bit patterns are valid for these types.
        unsafe {
            let base = self.storage.as_ptr().cast::<u8>();
            Ok(Batch {
                inputs: std::slice::from_raw_parts(
                    base.add(layout.offsets[0]).cast(),
                    layout.lengths[0] / 2,
                ),
                ids: std::slice::from_raw_parts(
                    base.add(layout.offsets[1]).cast(),
                    layout.lengths[1] / 4,
                ),
                weights: std::slice::from_raw_parts(
                    base.add(layout.offsets[2]).cast(),
                    layout.lengths[2] / 4,
                ),
                down: std::slice::from_raw_parts(
                    base.add(layout.offsets[3]).cast(),
                    layout.lengths[3] / 2,
                ),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_geometry_capacity_alignment_and_errors() {
        let w = CaptureWindow::new(120, 120).unwrap();
        let g = Geometry {
            input: 3,
            intermediate: 3,
            topk: 1,
        };
        assert_eq!(w.rows(g, 99).unwrap(), 6);
        assert_eq!(w.rows(g, 2).unwrap(), 2);
        assert!(CaptureWindow::new(121, 119).is_err());
        assert!(CaptureWindow::new(7, 99).is_err());
        assert!(w
            .rows(
                Geometry {
                    input: usize::MAX,
                    ..g
                },
                1
            )
            .is_err());
        assert!(w.rows(Geometry { topk: 0, ..g }, 1).is_err());
        assert!(w
            .rows(
                Geometry {
                    input: 4096,
                    intermediate: 2048,
                    topk: 8
                },
                1
            )
            .is_err());
    }

    #[test]
    fn bounded_offsets_reuse_partial_window_and_copy_failure() {
        let mut w = CaptureWindow::new(64, 64).unwrap();
        let base = w.storage.as_ptr();
        let g = Geometry {
            input: 3,
            intermediate: 3,
            topk: 1,
        };
        for offset in [0usize, 3, 6] {
            let batch = w
                .copy_with(g, 1, offset, [100, 200, 300, 400], |dest, src| {
                    dest.fill(src as u8);
                    Ok(())
                })
                .unwrap();
            assert_eq!(
                batch.inputs,
                &[u16::from_ne_bytes([(100 + offset * 6) as u8; 2]); 3]
            );
            assert_eq!(batch.ids.len(), 1);
            assert_eq!(
                batch.weights[0].to_bits(),
                u32::from_ne_bytes([(300 + offset * 4) as u8; 4])
            );
            assert_eq!(batch.down.len(), 3);
            assert_eq!(w.storage.as_ptr(), base);
        }
        assert!(w
            .copy_with(g, 1, 0, [0; 4], |_, _| Err("test copy fault".into()))
            .unwrap_err()
            .contains("test copy fault"));
        assert!(w
            .copy_with(g, 1, usize::MAX, [0; 4], |_, _| Ok(()))
            .is_err());
    }
}
