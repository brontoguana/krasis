// Diagnostic-only resident opaque TF32 component owners. No serving/cache dispatch.
struct ResidentComponents {
    high: cudarc::driver::CudaSlice<u32>,
    middle: cudarc::driver::CudaSlice<u32>,
    low: cudarc::driver::CudaSlice<u32>,
    device: std::sync::Arc<CudaDevice>,
    materialize_seconds: f64,
}
impl Drop for ResidentComponents {
    fn drop(&mut self) {
        if let Err(e) = self.device.synchronize() {
            eprintln!("ERROR: resident component teardown completion failed: {e}");
        }
    }
}
impl ResidentComponents {
    const GUARD_WORDS: usize = 32;
    const GUARD: u32 = 0x7fc1a123;
    fn allocation_bytes(rows: usize, cols: usize) -> Result<usize, String> {
        mul(mul(add(mul(rows, cols)?, Self::GUARD_WORDS)?, 4)?, 3)
    }
    fn validate_rows(rows: usize, selected: &[usize]) -> Result<(), String> {
        let unique = selected.iter().copied().collect::<std::collections::BTreeSet<_>>();
        if selected.is_empty() || unique.len() != selected.len() || selected.iter().any(|&r| r >= rows) {
            return Err("invalid resident component observation rows".into());
        }
        Ok(())
    }
    fn create(
        dev: &std::sync::Arc<CudaDevice>,
        packed: &cudarc::driver::CudaSlice<u8>,
        scales: &cudarc::driver::CudaSlice<f32>,
        zeros: &cudarc::driver::CudaSlice<f32>,
        rows: usize, cols: usize, group: usize, packed_stride: usize, scales_stride: usize,
        limits: [usize; 5],
    ) -> Result<Self, String> {
        if cols % 4 != 0 || group % 4 != 0 || limits[0] < 256 || limits[2] == 0 || limits[3] == 0 {
            return Err("unsupported resident component materialization geometry".into());
        }
        let count = add(mul(rows, cols)?, Self::GUARD_WORDS)?;
        let high = dev.htod_copy(vec![Self::GUARD; count]).map_err(|e| e.to_string())?;
        let _high_completion = Completion(dev);
        let middle = dev.htod_copy(vec![Self::GUARD; count]).map_err(|e| e.to_string())?;
        let _middle_completion = Completion(dev);
        let low = dev.htod_copy(vec![Self::GUARD; count]).map_err(|e| e.to_string())?;
        let _low_completion = Completion(dev);
        dev.synchronize().map_err(|e| e.to_string())?;
        let blocks = add(mul(rows, cols / 4)?, 255)? / 256;
        let gx = blocks.min(limits[2]);
        let gy = add(blocks, gx - 1)? / gx;
        if gy > limits[3] {
            return Err("actual resident materialization grid exceeds device limit".into());
        }
        let config = LaunchConfig { grid_dim: (gx as u32, gy as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let kernel = dev.get_func("row_operator", "hqq6_materialize_tf32_components").ok_or("component materializer missing")?;
        let start = Instant::now();
        unsafe {
            kernel.launch(config, (
                *packed.device_ptr(), *scales.device_ptr(), *zeros.device_ptr(),
                *high.device_ptr(), *middle.device_ptr(), *low.device_ptr(),
                rows as i32, cols as i32, group as i32, packed_stride as i32, scales_stride as i32,
            )).map_err(|e| e.to_string())?;
        }
        dev.synchronize().map_err(|e| e.to_string())?;
        let materialize_seconds = start.elapsed().as_secs_f64();
        Ok(Self { high, middle, low, device: dev.clone(), materialize_seconds })
    }
    fn evidence(&self, out: &Path, rows: usize, cols: usize, selected: &[usize], phase: &str) -> Result<Value, String> {
        Self::validate_rows(rows, selected)?;
        let n = mul(rows, cols)?;
        let mut records = Vec::new();
        for (name, plane) in [("high", &self.high), ("middle", &self.middle), ("low", &self.low)] {
            let words = self.device.dtoh_sync_copy(plane).map_err(|e| e.to_string())?;
            if words.len() != add(n, Self::GUARD_WORDS)? || words[n..].iter().any(|&w| w != Self::GUARD) || words[..n].contains(&Self::GUARD) {
                let raw = words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>();
                fs::write(out.join(format!("component-{name}-{phase}-guard-failure.u32")), raw).map_err(|e| e.to_string())?;
                return Err("resident component guard/unwritten payload failure retained".into());
            }
            let bytes = words[..n].iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>();
            let observed = selected.iter().flat_map(|&row| words[row*cols..(row+1)*cols].iter().flat_map(|w| w.to_le_bytes())).collect::<Vec<_>>();
            let path = out.join(format!("component-{name}-{phase}-selected.u32"));
            fs::write(&path, &observed).map_err(|e| e.to_string())?;
            records.push(json!({"component":name,"payload_bytes":bytes.len(),"payload_sha256":format!("{:x}",Sha256::digest(&bytes)),"guard_words":Self::GUARD_WORDS,"guard_exact":true,"selected":Identity{path,bytes:observed.len(),sha256:format!("{:x}",Sha256::digest(&observed))}}));
        }
        Ok(json!({"opaque_words_not_scalar_f32":true,"selected_rows":selected,"rows":rows,"cols":cols,"allocated_bytes":Self::allocation_bytes(rows,cols)?,"materialize_host_launch_completion_seconds":self.materialize_seconds,"planes":records}))
    }
}
