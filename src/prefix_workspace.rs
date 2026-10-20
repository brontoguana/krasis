impl PrefixWorkspace {
    /// Cold allocation from actual dimensions; this arena is counted once.
    /// max_new_rows is explicit capacity, including room for the original tail.
    pub fn new(
        device: Arc<CudaDevice>,
        geometry: Geometry,
        max_new_rows: usize,
        reserve_bytes: usize,
    ) -> Result<Self, String> {
        geometry.checked(max_new_rows)?;
        let layout = PrefixWorkspaceLayout::checked(geometry.heads, geometry.dim, max_new_rows)?;
        let (rows, beta_rows, matrices, padded, allocation_bytes) = (
            layout.rows,
            layout.beta_rows,
            layout.matrices,
            layout.padded,
            layout.bytes,
        );
        device.bind_to_thread().map_err(|e| e.to_string())?;
        let (mut free, mut total) = (0, 0);
        status(
            unsafe { sys::lib().cuMemGetInfo_v2(&mut free, &mut total) },
            "workspace free memory",
        )?;
        if free
            < allocation_bytes
                .checked_add(reserve_bytes)
                .ok_or("fixed16 workspace reserve overflow")?
        {
            return Err(format!("fixed16 batch workspace {allocation_bytes} plus reserve {reserve_bytes} exceeds measured free {free}"));
        }
        let alloc = |n| device.alloc_zeros::<f32>(n).map_err(|e| e.to_string());
        let raw = (0..5)
            .map(|i| {
                device
                    .alloc_zeros::<u16>(if i == 4 { beta_rows } else { rows })
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let arena = Self {
            device: device.clone(),
            heads: geometry.heads,
            dim: geometry.dim,
            max_new_rows,
            padded_rows: padded,
            allocation_bytes,
            raw,
            q: alloc(rows)?,
            k: alloc(rows)?,
            v: alloc(rows)?,
            g: alloc(rows)?,
            beta: alloc(beta_rows)?,
            gs: alloc(rows)?,
            kg: alloc(rows)?,
            qg: alloc(rows)?,
            pair: alloc(rows)?,
            a: alloc(matrices)?,
            qk: alloc(matrices)?,
            inverse: alloc(matrices)?,
            out: alloc(rows)?,
        };
        let actual_bytes = arena
            .raw
            .iter()
            .map(|x| x.len() * std::mem::size_of::<u16>())
            .sum::<usize>()
            + [
                &arena.q,
                &arena.k,
                &arena.v,
                &arena.g,
                &arena.beta,
                &arena.gs,
                &arena.kg,
                &arena.qg,
                &arena.pair,
                &arena.a,
                &arena.qk,
                &arena.inverse,
                &arena.out,
            ]
            .iter()
            .map(|x| x.len() * std::mem::size_of::<f32>())
            .sum::<usize>();
        if actual_bytes != allocation_bytes {
            return Err("fixed16 workspace type-derived payload differs from allocations".into());
        }
        // Cold producer completion makes independent caller streams safe to use.
        device.synchronize().map_err(|e| e.to_string())?;
        Ok(arena)
    }
    fn validate(
        &self,
        device: &Arc<CudaDevice>,
        geometry: Geometry,
        rows: usize,
    ) -> Result<(), String> {
        if *self.device.cu_primary_ctx() != *device.cu_primary_ctx()
            || self.heads != geometry.heads
            || self.dim != geometry.dim
            || rows == 0
            || rows > self.max_new_rows
        {
            return Err("fixed16 workspace context/geometry/capacity mismatch".into());
        }
        Ok(())
    }
    pub fn diagnostic_output_ptr(&self) -> u64 {
        ptr(&self.out)
    }
    pub fn diagnostic_pointers(&self) -> Vec<u64> {
        let mut p = self.raw.iter().map(ptr).collect::<Vec<_>>();
        p.extend(
            [
                &self.q,
                &self.k,
                &self.v,
                &self.g,
                &self.beta,
                &self.gs,
                &self.kg,
                &self.qg,
                &self.pair,
                &self.a,
                &self.qk,
                &self.inverse,
                &self.out,
            ]
            .iter()
            .map(|x| ptr(x)),
        );
        p
    }
}
impl Drop for PrefixWorkspace {
    fn drop(&mut self) {
        if let Err(e) = self.device.bind_to_thread() {
            log::error!("fixed16 arena cleanup context bind: {e}");
            return;
        }
        let rc = unsafe { sys::lib().cuCtxSynchronize() };
        if rc != sys::CUresult::CUDA_SUCCESS {
            log::error!("fixed16 arena cleanup synchronization: {rc:?}");
        }
    }
}
