// Rust-created uploads own their allocations separately from raw descriptors
// for externally supplied tensors. All release is under Graph/Store completion.
impl GpuDecodeStore {
    fn register_owned_bf16_weight_rust(
        &mut self,
        values: Vec<u16>,
        rows: usize,
        cols: usize,
    ) -> PyResult<usize> {
        if rows.checked_mul(cols) != Some(values.len()) || values.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "owned BF16 weight geometry differs from data",
            ));
        }
        let graph = self.graph.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Call configure before owned weight upload")
        })?;
        let buffer = self
            .device
            .htod_copy(values)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let id = graph.weights.len();
        graph
            .weights
            .push(GpuWeight::new(*buffer.device_ptr(), rows, cols, 0));
        graph.owned_weight_bf16.push(buffer);
        Ok(id)
    }
    fn register_owned_f32_weight_rust(
        &mut self,
        values: Vec<f32>,
        rows: usize,
        cols: usize,
    ) -> PyResult<usize> {
        if rows.checked_mul(cols) != Some(values.len()) || values.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "owned F32 weight geometry differs from data",
            ));
        }
        let graph = self.graph.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Call configure before owned weight upload")
        })?;
        let buffer = self
            .device
            .htod_copy(values)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let id = graph.weights.len();
        graph
            .weights
            .push(GpuWeight::new(*buffer.device_ptr(), rows, cols, 1));
        graph.owned_weight_f32.push(buffer);
        Ok(id)
    }
    #[cfg(feature = "scalar-gdn-test")]
    fn owned_weight_control_rust(
        &mut self,
        rows: usize,
        cols: usize,
        group: usize,
        fail_after_enqueue: bool,
    ) -> PyResult<String> {
        let graph = self.graph.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("configure the empty Store first")
        })?;
        if graph.num_layers != 0
            || !graph.weights.is_empty()
            || !graph.owned_weight_bf16.is_empty()
            || !graph.owned_weight_f32.is_empty()
            || !graph.owned_weight_packed.is_empty()
        {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "owned weight control requires an empty zero-layer Store; never a loaded model",
            ));
        }
        let count = rows.checked_mul(cols).filter(|&n| n > 0).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("control geometry overflow/empty")
        })?;
        if group == 0 || cols % group != 0 || count % 8 != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "control quantized geometry is not aligned",
            ));
        }
        self.register_owned_bf16_weight_rust(vec![0x3f80; count], rows, cols)?;
        self.register_owned_f32_weight_rust(vec![1.0; count], rows, cols)?;
        self.register_owned_bf16_weight_rust(vec![0], 1, 1)?;
        for bits in [4usize, 8] {
            let repacked = crate::weights::marlin::MarlinRepacked {
                packed: vec![0u32; count / (32 / bits)],
                scales: vec![0x3f80; rows * (cols / group)],
                k: cols,
                n: rows,
                group_size: group,
            };
            if bits == 4 {
                self._finish_register_marlin_int4(repacked, rows, cols, group)?;
            } else {
                self._finish_register_marlin_int8(repacked, rows, cols, group)?;
            }
        }
        // Actual allocations and queued work; configure/Store teardown must
        // retain them through completion even without captured executables.
        let graph = self.graph.as_ref().unwrap();
        let ready = ScopedCudaEvent::new("owned weight control upload readiness")
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        unsafe { cuda_sys::lib().cuEventRecord(ready.0, *self.device.cu_stream()) }
            .result()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        unsafe { cuda_sys::lib().cuStreamWaitEvent(self.copy_stream.0, ready.0, 0) }
            .result()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        for buffer in &graph.owned_weight_bf16 {
            unsafe {
                cuda_sys::lib().cuMemsetD8Async(
                    *buffer.device_ptr(),
                    0x37,
                    buffer.len() * std::mem::size_of::<u16>(),
                    self.copy_stream.0,
                )
            }
            .result()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }
        let bytes = graph
            .owned_weight_bf16
            .iter()
            .map(|x| x.len() * std::mem::size_of::<u16>())
            .sum::<usize>()
            + graph
                .owned_weight_f32
                .iter()
                .map(|x| x.len() * std::mem::size_of::<f32>())
                .sum::<usize>()
            + graph
                .owned_weight_packed
                .iter()
                .map(|x| x.len() * std::mem::size_of::<u32>())
                .sum::<usize>();
        if fail_after_enqueue {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "explicit owned-weight control: error after queued uploads/copy",
            ));
        }
        Ok(serde_json::json!({"model_layers":0,"actual_production_upload_helpers":true,"weight_descriptors":graph.weights.len(),"owned_bf16":graph.owned_weight_bf16.len(),"owned_f32":graph.owned_weight_f32.len(),"owned_packed":graph.owned_weight_packed.len(),"owned_bytes":bytes,"queued_real_copy":true}).to_string())
    }
}
