//! One-time setup admission for an actual Torch CUDA allocation.
//! The persistent region methods below never enter Python or acquire the GIL.
use crate::{
    cuda_region::{sealed, CudaRegion},
    stable_mla_gemm::Element,
};
use cudarc::driver::{sys, CudaDevice};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::sync::Arc;

pub(crate) struct ModelTensorOwner {
    _tensor: Py<PyAny>,
    _storage: Py<PyAny>,
    device: Arc<CudaDevice>,
    start: u64,
    end: u64,
    pub(crate) element: Element,
    pub(crate) shape: Vec<usize>,
}
impl sealed::Sealed for ModelTensorOwner {}
impl CudaRegion for ModelTensorOwner {
    fn extent(&self) -> Result<(u64, u64), String> {
        Ok((self.start, self.end))
    }
    fn context(&self) -> sys::CUcontext {
        *self.device.cu_primary_ctx()
    }
}
impl ModelTensorOwner {
    /// Called only during loading while the caller owns the real model tensor.
    /// Service setup must not resize/rebind its storage while native views exist.
    pub(crate) fn admit(
        py: Python<'_>,
        tensor: &Bound<'_, PyAny>,
        device: Arc<CudaDevice>,
        expected: Element,
    ) -> PyResult<Arc<Self>> {
        let torch = py.import("torch")?;
        if !tensor.is_instance(&torch.getattr("Tensor")?)? {
            return Err(PyValueError::new_err(
                "model allocation owner must be an actual Torch tensor",
            ));
        }
        let location = tensor.getattr("device")?;
        if location.getattr("type")?.extract::<String>()? != "cuda" {
            return Err(PyValueError::new_err(
                "model allocation owner is not a CUDA tensor",
            ));
        }
        let ordinal = location.getattr("index")?.extract::<usize>()?;
        if ordinal != device.ordinal() {
            return Err(PyValueError::new_err(
                "model tensor is on a different CUDA device",
            ));
        }
        if !tensor.call_method0("is_contiguous")?.extract::<bool>()? {
            return Err(PyValueError::new_err(
                "model tensor view is not contiguous; no implicit repacking is permitted",
            ));
        }
        let dtype = tensor.getattr("dtype")?.str()?.to_str()?.to_owned();
        let (element, element_bytes) = match dtype.as_str() {
            "torch.float32" => (Element::F32, 4usize),
            "torch.bfloat16" => (Element::Bf16, 2usize),
            "torch.int32" => (Element::I32, 4usize),
            "torch.uint8" => (Element::Bytes, 1usize),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported actual model tensor dtype: {dtype}"
                )))
            }
        };
        if element != expected {
            return Err(PyValueError::new_err(format!(
                "actual model tensor dtype {dtype} differs from its native registration"
            )));
        }
        if tensor.call_method0("element_size")?.extract::<usize>()? != element_bytes {
            return Err(PyValueError::new_err(
                "model tensor element width is inconsistent",
            ));
        }
        let shape = tensor.getattr("shape")?.extract::<Vec<usize>>()?;
        let elements = shape.iter().try_fold(1usize, |n, &d| {
            n.checked_mul(d)
                .ok_or_else(|| PyValueError::new_err("model tensor shape overflow"))
        })?;
        if elements == 0 || elements != tensor.call_method0("numel")?.extract::<usize>()? {
            return Err(PyValueError::new_err(
                "model tensor owner has an empty/inconsistent shape",
            ));
        }
        let bytes = elements
            .checked_mul(element_bytes)
            .ok_or_else(|| PyValueError::new_err("model tensor bytes overflow"))?;
        let start = tensor.call_method0("data_ptr")?.extract::<u64>()?;
        let end = start
            .checked_add(u64::try_from(bytes).map_err(|e| PyValueError::new_err(e.to_string()))?)
            .ok_or_else(|| PyValueError::new_err("model tensor address overflow"))?;
        let storage = tensor.call_method0("untyped_storage")?;
        let base = storage.call_method0("data_ptr")?.extract::<u64>()?;
        let storage_bytes = storage.call_method0("nbytes")?.extract::<u64>()?;
        let limit = base
            .checked_add(storage_bytes)
            .ok_or_else(|| PyValueError::new_err("model storage address overflow"))?;
        if start == 0
            || base == 0
            || start < base
            || end > limit
            || start % element_bytes as u64 != 0
        {
            return Err(PyValueError::new_err(
                "model tensor view is not covered by its real retained storage",
            ));
        }
        let mut context = std::ptr::null_mut();
        unsafe {
            sys::lib().cuPointerGetAttribute(
                (&mut context as *mut sys::CUcontext).cast(),
                sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT,
                start,
            )
        }
        .result()
        .map_err(|e| {
            PyValueError::new_err(format!(
                "actual model allocation context query failed: {e:?}"
            ))
        })?;
        if context != *device.cu_primary_ctx() {
            return Err(PyValueError::new_err(
                "actual model allocation belongs to a different CUDA context",
            ));
        }
        Ok(Arc::new(Self {
            _tensor: tensor.clone().unbind(),
            _storage: storage.unbind(),
            device,
            start,
            end,
            element,
            shape,
        }))
    }
    pub(crate) fn view(
        &self,
        offset: usize,
        bytes: usize,
    ) -> Result<crate::stable_mla_gemm::Span<'_>, String> {
        // Element type is from the actual validated tensor, not the caller.
        unsafe { crate::stable_mla_gemm::Span::region(self, offset, bytes, self.element) }
    }
}

/// Installed setup-only ownership fixture, with no inference or numeric conversion.
#[cfg(feature = "shared-kda-test")]
#[pyfunction]
pub(crate) fn model_tensor_owner_fixture(
    py: Python<'_>,
    factory: &Bound<'_, PyAny>,
    device_index: usize,
    expected: &str,
    read_bytes: usize,
) -> PyResult<String> {
    let expected = match expected {
        "f32" => Element::F32,
        "bf16" => Element::Bf16,
        "i32" => Element::I32,
        "bytes" => Element::Bytes,
        _ => return Err(PyValueError::new_err("unknown explicit owner element type")),
    };
    let device = CudaDevice::new_with_stream(device_index)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut uuid = unsafe { std::mem::zeroed::<sys::CUuuid>() };
    unsafe { sys::lib().cuDeviceGetUuid_v2(&mut uuid, *device.cu_device()) }
        .result()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let hex = uuid
        .bytes
        .iter()
        .map(|&b| format!("{:02x}", b as u8))
        .collect::<String>();
    let physical = format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    );
    let tensor = factory.call0()?;
    let owner = ModelTensorOwner::admit(py, &tensor, device.clone(), expected)?;
    let weak = py.import("weakref")?.getattr("ref")?.call1((&tensor,))?;
    let bytes = usize::try_from(owner.end - owner.start)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let element_bytes = match expected {
        Element::F32 | Element::I32 => 4,
        Element::Bf16 => 2,
        Element::Bytes => 1,
    };
    if read_bytes == 0 || read_bytes > bytes || read_bytes % element_bytes != 0 {
        return Err(PyValueError::new_err(
            "explicit owner fixture window is empty, unaligned or exceeds source",
        ));
    }
    let kernel_bank =
        crate::stable_mla_ops::load_serving(device.clone()).map_err(PyValueError::new_err)?;
    let source = owner.view(0, read_bytes).map_err(PyValueError::new_err)?;
    drop(tensor);
    if weak.call0()?.is_none() {
        return Err(PyValueError::new_err(
            "registered tensor owner did not retain its real source",
        ));
    }
    let mut context = std::ptr::null_mut();
    unsafe {
        sys::lib().cuPointerGetAttribute(
            (&mut context as *mut sys::CUcontext).cast(),
            sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT,
            source.pointer(),
        )
    }
    .result()
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if context != *device.cu_primary_ctx() {
        return Err(PyValueError::new_err(
            "retained source allocation no longer has its admitted CUDA context",
        ));
    }
    let shape = owner.shape.clone();
    drop(source);
    drop(owner);
    if !weak.call0()?.is_none() {
        return Err(PyValueError::new_err(
            "fixture factory must return a fresh tensor without another persistent reference",
        ));
    }
    Ok(serde_json::json!({"passed":true,"actual_device_uuid":physical,"element":format!("{expected:?}"),"shape":shape,"allocation_bytes":bytes,"admitted_window_bytes":read_bytes,"real_owner_retained_after_original_reference_drop":true,"tensor_reference_released_after_owner_drop":true,"actual_cuda_context_rechecked":true,"installed_kernel_bank_identity":kernel_bank.identity,"scope":"Installed one-time setup ownership/pointer/type/window proof only; no GPU transfer, kernel, model state or serving qualification"}).to_string())
}

impl ModelTensorOwner {
    pub(crate) fn complete_setup_source(&self) -> Result<(), String> {
        self.device
            .bind_to_thread()
            .map_err(|e| format!("cold source context bind: {e}"))?;
        if let Err(error) = unsafe { sys::lib().cuCtxSynchronize() }.result() {
            crate::vram_monitor::fatal_cuda_context_error(
                "cold native source readiness",
                &error.to_string(),
            );
        }
        Ok(())
    }
    /// Exact one-time BF16 weight widening. Allocate before runtime memory
    /// calibration; callers must account the actual returned tensor extent.
    pub(crate) fn widen_bf16(
        &self,
        ops: &Arc<crate::stable_mla_ops::Ops>,
    ) -> Result<cudarc::driver::CudaSlice<f32>, String> {
        self.widen_bf16_then(ops, || Ok(()))
    }
    fn widen_bf16_then(
        &self,
        ops: &Arc<crate::stable_mla_ops::Ops>,
        after: impl FnOnce() -> Result<(), String>,
    ) -> Result<cudarc::driver::CudaSlice<f32>, String> {
        if self.element != Element::Bf16 {
            return Err("cold native weight widening requires actual BF16 source".into());
        }
        let elements = usize::try_from((self.end - self.start) / 2).map_err(|e| e.to_string())?;
        let stream = *self.device.cu_stream();
        if stream.is_null() {
            return Err("cold native weight widening requires retained nondefault stream".into());
        }
        // Setup source tensors may have been produced on Torch's other stream.
        // This is a one-time cold boundary, never a decode/prefill token loop.
        self.complete_setup_source()?;
        let output = self
            .device
            .alloc_zeros::<f32>(elements)
            .map_err(|e| format!("cold F32 weight allocation: {e}"))?;
        let retained = (self, &output, ops);
        unsafe {
            crate::sequence_copy_completion::with_copy_completion(
                &self.device,
                stream,
                &retained,
                |(source, output, ops)| {
                    let input = source.view(
                        0,
                        elements.checked_mul(2).ok_or("cold BF16 extent overflow")?,
                    )?;
                    let target = crate::stable_mla_gemm::Span::f32(output, 0, elements)?;
                    ops.widen(&input, &target, elements, stream)?;
                    after()
                },
            )
        }?;
        Ok(output)
    }
}

/// Installed cold actual-owner/temporary-allocation completion controls.
#[cfg(feature = "shared-kda-test")]
#[pyfunction]
pub(crate) fn model_tensor_widen_fixture(
    py: Python<'_>,
    factory: &Bound<'_, PyAny>,
    device_index: usize,
) -> PyResult<String> {
    let device = CudaDevice::new_with_stream(device_index)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let tensor = factory.call0()?;
    let owner = ModelTensorOwner::admit(py, &tensor, device.clone(), Element::Bf16)?;
    let weak = py.import("weakref")?.getattr("ref")?.call1((&tensor,))?;
    let elements = usize::try_from((owner.end - owner.start) / 2)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut original = vec![0u16; elements];
    owner
        .complete_setup_source()
        .map_err(PyValueError::new_err)?;
    unsafe { sys::lib().cuMemcpyDtoH_v2(original.as_mut_ptr().cast(), owner.start, elements * 2) }
        .result()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let ops = crate::stable_mla_ops::load_serving(device.clone()).map_err(PyValueError::new_err)?;
    drop(tensor);
    let error = owner
        .widen_bf16_then(&ops, || {
            Err("injected cold post-enqueue callback error".into())
        })
        .err()
        .ok_or_else(|| PyValueError::new_err("cold post-enqueue error was lost"))?;
    if !error.contains("injected cold post-enqueue") {
        return Err(PyValueError::new_err(error));
    }
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = owner.widen_bf16_then(&ops, || -> Result<(), String> {
            panic!("injected cold post-enqueue callback unwind")
        });
    }));
    if unwound.is_ok() {
        return Err(PyValueError::new_err(
            "cold post-enqueue callback did not unwind",
        ));
    }
    if weak.call0()?.is_none() {
        return Err(PyValueError::new_err(
            "actual source released during cold error/unwind",
        ));
    }
    let output = owner.widen_bf16(&ops).map_err(PyValueError::new_err)?;
    let values = device
        .dtoh_sync_copy(&output)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let expected: Vec<_> = original.iter().map(|&u| (u as u32) << 16).collect();
    let bits: Vec<_> = values.iter().map(|f| f.to_bits()).collect();
    if bits != expected {
        return Err(PyValueError::new_err("actual cold widened bytes differ"));
    }
    let mut after = vec![0u16; elements];
    unsafe { sys::lib().cuMemcpyDtoH_v2(after.as_mut_ptr().cast(), owner.start, elements * 2) }
        .result()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if after != original {
        return Err(PyValueError::new_err(
            "actual source bytes changed during cold conversion",
        ));
    }
    let native = crate::gpu_prefill::GpuBuf::<u8>::alloc_uninit(elements * 2)
        .map_err(PyValueError::new_err)?;
    let region = crate::sequence_external_owner::BorrowedGpuRegion::retained(&device, &native)
        .map_err(PyValueError::new_err)?;
    let region_view =
        unsafe { crate::stable_mla_gemm::Span::region(&region, 0, elements * 2, Element::Bytes) }
            .map_err(PyValueError::new_err)?;
    if region_view.context() != *device.cu_primary_ctx() {
        return Err(PyValueError::new_err(
            "native retained GpuBuf context differs",
        ));
    }
    drop(region_view);
    drop(region);
    drop(native);
    drop(output);
    drop(owner);
    if !weak.call0()?.is_none() {
        return Err(PyValueError::new_err(
            "actual cold source not released after completion and owner drop",
        ));
    }
    Ok(serde_json::json!({"passed":true,"elements":elements,"f32_bits":bits,"post_enqueue_error_completed":true,"post_enqueue_unwind_completed":true,"recovery_exact":true,"source_readonly_exact":true,"source_retained_through_error_unwind":true,"source_released_after_owner_drop":true,"real_gpu_buf_retained_context":true,"scope":"Installed cold BF16 conversion and real allocation lifetime/completion controls; no per-token synchronization or model serving qualification"}).to_string())
}
