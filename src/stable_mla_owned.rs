//! Cold setup owners for existing model tensors and one exact F32 WVC mirror.
use crate::{
    cuda_region::CudaRegion,
    model_tensor_owner::ModelTensorOwner,
    stable_mla_gemm::{Element, Span},
    stable_mla_ops::Ops,
};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::sync::Arc;
#[derive(Clone, Copy, Debug)]
pub(crate) struct Dimensions {
    pub heads: usize,
    pub values: usize,
    pub ccd: usize,
    pub rope: usize,
}
pub(crate) struct Layer {
    pub dimensions: Dimensions,
    _wvc_source: Arc<ModelTensorOwner>,
    pub ckv: Arc<ModelTensorOwner>,
    pub kpe: Option<Arc<ModelTensorOwner>>,
    pub wvc_f32: CudaSlice<f32>,
}
impl Layer {
    pub(crate) fn admit(
        py: Python<'_>,
        device: Arc<CudaDevice>,
        ops: &Arc<Ops>,
        dimensions: Dimensions,
        pointers: [u64; 3],
        wvc: &Bound<'_, PyAny>,
        ckv: &Bound<'_, PyAny>,
        kpe: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Arc<Self>> {
        let d = dimensions;
        if d.heads == 0 || d.values == 0 || d.ccd == 0 || d.ccd % 16 != 0 || d.rope % 16 != 0 {
            return Err(PyValueError::new_err(
                "loaded common MLA owner geometry is invalid",
            ));
        }
        let source = ModelTensorOwner::admit(py, wvc, device.clone(), Element::Bf16)?;
        let ckv = ModelTensorOwner::admit(py, ckv, device.clone(), Element::Bytes)?;
        let wbytes = d
            .heads
            .checked_mul(d.values)
            .and_then(|n| n.checked_mul(d.ccd))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| PyValueError::new_err("loaded WVC owner extent overflow"))?;
        let w = source.extent().map_err(PyValueError::new_err)?;
        let c = ckv.extent().map_err(PyValueError::new_err)?;
        if w.0 != pointers[0] || w.1 - w.0 != wbytes as u64 || c.0 != pointers[1] {
            return Err(PyValueError::new_err(
                "actual retained WVC/cache tensors do not cover loaded runtime descriptors",
            ));
        }
        let kpe = if d.rope == 0 {
            if kpe.is_some() {
                return Err(PyValueError::new_err(
                    "zero-RoPE runtime has a nonempty positional owner",
                ));
            }
            None
        } else {
            let tensor = kpe
                .ok_or_else(|| PyValueError::new_err("actual positional cache owner is missing"))?;
            let owner = ModelTensorOwner::admit(py, tensor, device, Element::Bytes)?;
            let k = owner.extent().map_err(PyValueError::new_err)?;
            if k.0 != pointers[2] {
                return Err(PyValueError::new_err(
                    "actual positional cache owner differs from loaded descriptor",
                ));
            }
            Some(owner)
        };
        let wvc_f32 = source.widen_bf16(ops).map_err(PyValueError::new_err)?;
        Ok(Arc::new(Self {
            dimensions,
            _wvc_source: source,
            ckv,
            kpe,
            wvc_f32,
        }))
    }
    /// Called only after the native loader has finalized its real cache contract.
    pub(crate) fn validate_cache(&self, context: usize, format: u32) -> Result<(), String> {
        if context == 0 {
            return Err("native MLA cache capacity is not finalized".into());
        }
        let block = match format {
            7 => 14usize,
            9 => 10usize,
            _ => return Err("common MLA cache format is not K4/K6".into()),
        };
        for (width, owner) in [
            (self.dimensions.ccd, Some(&self.ckv)),
            (self.dimensions.rope, self.kpe.as_ref()),
        ] {
            if width == 0 {
                continue;
            }
            let owner = owner.ok_or("actual MLA positional owner missing")?;
            let (start, end) = owner.extent()?;
            let needed = context
                .checked_mul(width / 16)
                .and_then(|n| n.checked_mul(block))
                .ok_or("actual MLA cache extent overflow")?;
            if end - start < needed as u64 {
                return Err("actual MLA tensor storage below finalized cache capacity".into());
            }
        }
        Ok(())
    }
    pub(crate) fn weight_view(&self) -> Result<Span<'_>, String> {
        Span::f32(&self.wvc_f32, 0, self.wvc_f32.len())
    }
    pub(crate) fn compressed_view(&self) -> Result<Span<'_>, String> {
        let (start, end) = self.ckv.extent()?;
        self.ckv
            .view(0, usize::try_from(end - start).map_err(|e| e.to_string())?)
    }
    pub(crate) fn positional_view(&self) -> Result<Option<Span<'_>>, String> {
        self.kpe
            .as_ref()
            .map(|owner| {
                let (start, end) = owner.extent()?;
                owner.view(0, usize::try_from(end - start).map_err(|e| e.to_string())?)
            })
            .transpose()
    }
    pub(crate) fn resident_bytes(&self) -> Result<usize, String> {
        self.wvc_f32
            .len()
            .checked_mul(4)
            .ok_or("actual F32 WVC residency overflow".into())
    }
}
