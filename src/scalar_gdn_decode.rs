// Included in gpu_decode. All real buffers/positions remain borrowed from the
// configured graph and whole owned invocation; no Python orchestration.
impl GpuDecodeStore {
    fn scalar_gdn_active_health(&self) -> Result<(), String> {
        #[cfg(feature = "scalar-gdn-test")]
        if let Some(storage) = &self.selected_scalar_gdn {
            storage.require_active_health()?;
        }
        Ok(())
    }
    pub(crate) fn scalar_gdn_publication_health(&self) -> Result<(), String> {
        #[cfg(feature = "scalar-gdn-test")]
        if let Some(storage) = &self.selected_scalar_gdn {
            storage.require_publication_health()?;
        }
        Ok(())
    }
    fn scalar_gdn_has_local_storage(&self) -> bool {
        #[cfg(feature = "scalar-gdn-test")]
        {
            self.selected_scalar_gdn.is_some()
        }
        #[cfg(not(feature = "scalar-gdn-test"))]
        {
            false
        }
    }
    fn scalar_decode_invocation<T>(
        &mut self,
        work: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        #[cfg(feature = "scalar-gdn-test")]
        {
            if self.scalar_gdn_requested && !self.scalar_gdn_initialized {
                return Err("scalar decode requires finalized actual device inventory".into());
            }
            if let Some(storage) = self.selected_scalar_gdn.clone() {
                return unsafe { storage.with_decode(|| work(self)) };
            }
        }
        // Default mode and a validated auxiliary split with no scalar layers
        // have no scalar execution domain to enter. No state owner is invented.
        work(self)
    }
    #[cfg(feature = "scalar-gdn-test")]
    fn scalar_native_attention_inputs(
        &self,
        layer: usize,
        input: &dyn crate::cuda_region::CudaRegion,
        input_offset: usize,
        output: &dyn crate::cuda_region::CudaRegion,
        output_offset: usize,
        host_position: Option<usize>,
        device_position: Option<&CudaSlice<i32>>,
    ) -> Result<(), String> {
        use crate::{
            scalar_gdn_owner::{DecodePosition, Plane},
            scalar_gdn_producer::Role,
        };
        let storage = self
            .selected_scalar_gdn
            .as_ref()
            .ok_or("executed scalar attention has no actual native storage")?;
        let producer = storage.producer()?;
        let hidden = producer.projection_layout(layer, Role::Qkvz)?.cols;
        // Real graph hidden/batch buffers are written and consumed as BF16;
        // the sealed regions supply actual bounds and CUDA context metadata.
        let input = unsafe { Plane::<u16>::from_region(input, input_offset, hidden)? };
        let output = unsafe { Plane::<u16>::from_region(output, output_offset, hidden)? };
        let position = match (host_position, device_position) {
            (Some(p), None) => DecodePosition::Host(u64::try_from(p).map_err(|e| e.to_string())?),
            (None, Some(p)) => DecodePosition::Device(Plane::checked(p, 0, 1)?),
            _ => {
                return Err(
                    "scalar attention requires exactly one real eager or device position".into(),
                )
            }
        };
        let stream = *self.device.cu_stream();
        unsafe {
            let prepared = producer.prepare_decode(layer, &input, stream)?;
            let inputs = prepared.inputs()?;
            let emitted = storage.decode_emitted(layer, &inputs, &position)?;
            prepared.finish(&emitted.plane()?, &output, stream)
        }
    }
    #[cfg(not(feature = "scalar-gdn-test"))]
    fn scalar_native_attention_inputs(
        &self,
        _layer: usize,
        _input: &dyn crate::cuda_region::CudaRegion,
        _input_offset: usize,
        _output: &dyn crate::cuda_region::CudaRegion,
        _output_offset: usize,
        _host_position: Option<usize>,
        _device_position: Option<&CudaSlice<i32>>,
    ) -> Result<(), String> {
        Err("native scalar attention was not built; cold admission must reject this request".into())
    }
}
