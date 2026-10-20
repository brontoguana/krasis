// Concrete current-state ownership passed to the repaired generic Store DMA.
impl PreparedOwner {
    fn external_dma_regions(
        &self,
    ) -> Result<Vec<crate::sequence_external_owner::BorrowedGpuRegion<'_>>, String> {
        use crate::sequence_external_owner::BorrowedGpuRegion as Region;
        let mut regions = Vec::new();
        for b in self.arrays.values() {
            regions.push(Region::checked(&self.device, b)?)
        }
        for b in &self.pending_raw.buffers {
            match b {
                NativeBuffer::ScalarF32(b) => regions.push(Region::checked(&self.device, b)?),
                NativeBuffer::Bf16(b) => regions.push(Region::checked(&self.device, b)?),
                NativeBuffer::Bf16View(_) => {
                    return Err(
                        "persistent current-state DMA requires actual owned pending buffers".into(),
                    )
                }
            }
        }
        regions.push(Region::checked(&self.device, &self.pending)?);
        regions.push(Region::checked(&self.device, &self.error)?);
        regions.push(Region::checked(&self.device, &self.absolute)?);
        Ok(regions)
    }
}
