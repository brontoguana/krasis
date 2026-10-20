// Existing canonical pending-F32 invalid NaN ABI; never a tuning value.
const KDA_PENDING_INVALID_F32_BITS: u32 = 0x7fc1a123;
use cudarc::driver::{sys, CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use serde::{Deserialize, Serialize};
use std::{mem::size_of, sync::Arc};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RawContract {
    KdaBf16VectorGate,
    GdnPreparedF32ScalarGate,
    KdaPreparedF32VectorGate,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Dtype {
    Bfloat16,
    Float32,
}
impl Dtype {
    fn bytes(self) -> usize {
        match self {
            Self::Bfloat16 => size_of::<u16>(),
            Self::Float32 => size_of::<f32>(),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Plane {
    role: String,
    dtype: Dtype,
    shape: Vec<usize>,
    strides_bytes: Vec<usize>,
    elements: usize,
    storage_bytes: usize,
}
impl Plane {
    fn checked(role: &str, dtype: Dtype, shape: Vec<usize>) -> Result<Self, String> {
        if shape.iter().any(|&n| n == 0) {
            return Err("zero raw extent".into());
        }
        let mut stride = dtype.bytes();
        let mut strides = vec![0; shape.len()];
        let mut elements = 1usize;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride = stride.checked_mul(shape[i]).ok_or("raw byte overflow")?;
            elements = elements
                .checked_mul(shape[i])
                .ok_or("raw element overflow")?;
        }
        i32::try_from(elements).map_err(|_| "raw CUDA signed index ABI exceeded")?;
        Ok(Self {
            role: role.into(),
            dtype,
            shape,
            strides_bytes: strides,
            elements,
            storage_bytes: stride,
        })
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Schema {
    contract: RawContract,
    planes: Vec<Plane>,
    requested_bytes: usize,
}
impl Schema {
    fn checked(
        contract: RawContract,
        rows: usize,
        heads: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> Result<Self, String> {
        if key_dim == 0 || key_dim != value_dim || key_dim > 128 {
            return Err("current fixed16 arithmetic geometry unsupported".into());
        }
        let mut planes = Vec::new();
        for role in ["q", "k", "v"] {
            planes.push(Plane::checked(
                role,
                if contract == RawContract::KdaPreparedF32VectorGate {
                    Dtype::Float32
                } else {
                    Dtype::Bfloat16
                },
                vec![rows, heads, key_dim],
            )?);
        }
        let gate = match contract {
            RawContract::KdaPreparedF32VectorGate => {
                Plane::checked("g", Dtype::Float32, vec![rows, heads, key_dim])?
            }
            RawContract::KdaBf16VectorGate => {
                Plane::checked("g", Dtype::Bfloat16, vec![rows, heads, key_dim])?
            }
            RawContract::GdnPreparedF32ScalarGate => {
                Plane::checked("g", Dtype::Float32, vec![rows, heads])?
            }
        };
        planes.push(gate);
        planes.push(Plane::checked(
            "beta",
            if contract == RawContract::KdaPreparedF32VectorGate {
                Dtype::Float32
            } else {
                Dtype::Bfloat16
            },
            vec![rows, heads],
        )?);
        let requested_bytes = planes.iter().try_fold(0usize, |sum, p| {
            sum.checked_add(p.storage_bytes)
                .ok_or("raw payload overflow")
        })?;
        Ok(Self {
            contract,
            planes,
            requested_bytes,
        })
    }
    fn validate(&self) -> Result<(), String> {
        let q = &self.planes.first().ok_or("missing raw Q plane")?.shape;
        if q.len() != 3 {
            return Err("raw Q rank".into());
        }
        let expected = Self::checked(self.contract, q[0], q[1], q[2], q[2])?;
        if expected != *self {
            return Err("invalid raw contract descriptor accounting".into());
        }
        Ok(())
    }
    fn admits(&self, saved: &Self) -> Result<(), String> {
        self.validate()?;
        saved.validate()?;
        if self != saved {
            return Err("raw contract/type/shape/stride identity mismatch".into());
        }
        Ok(())
    }
}
#[allow(dead_code)]
enum TypedBuffer {
    Bf16(CudaSlice<u16>),
    ScalarF32(CudaSlice<f32>),
}
#[allow(dead_code)]
impl TypedBuffer {
    fn elements(&self) -> usize {
        match self {
            Self::Bf16(x) => x.len(),
            Self::ScalarF32(x) => x.len(),
        }
    }
    fn address(&self) -> u64 {
        match self {
            Self::Bf16(x) => *x.device_ptr(),
            Self::ScalarF32(x) => *x.device_ptr(),
        }
    }
    fn dtype(&self) -> Dtype {
        match self {
            Self::Bf16(_) => Dtype::Bfloat16,
            Self::ScalarF32(_) => Dtype::Float32,
        }
    }
}
#[allow(dead_code)]
struct RawOwner {
    device: Arc<CudaDevice>,
    schema: Schema,
    buffers: Vec<TypedBuffer>,
    memory_before: [usize; 2],
    memory_after: [usize; 2],
}
fn check(rc: sys::CUresult, label: &str) -> Result<(), String> {
    if rc == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{label}:{rc:?}"))
    }
}
fn memory() -> Result<[usize; 2], String> {
    let (mut f, mut t) = (0, 0);
    check(
        unsafe { sys::lib().cuMemGetInfo_v2(&mut f, &mut t) },
        "actual rawowner memory",
    )?;
    Ok([f, t])
}
#[allow(dead_code)]
impl RawOwner {
    fn allocate(
        device: Arc<CudaDevice>,
        schema: Schema,
        reserve_bytes: usize,
    ) -> Result<Self, String> {
        schema.validate()?;
        device.bind_to_thread().map_err(|e| e.to_string())?;
        let before = memory()?;
        if before[0]
            < schema
                .requested_bytes
                .checked_add(reserve_bytes)
                .ok_or("raw reserve overflow")?
        {
            return Err("measured free cannot cover typed raw payload and caller reserve".into());
        }
        let buffers = schema
            .planes
            .iter()
            .map(|p| match p.dtype {
                Dtype::Bfloat16 => device.alloc_zeros::<u16>(p.elements).map(TypedBuffer::Bf16),
                Dtype::Float32 => device
                    .alloc_zeros::<f32>(p.elements)
                    .map(TypedBuffer::ScalarF32),
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        device.synchronize().map_err(|e| e.to_string())?;
        let mut actual_bytes = 0usize;
        for (p, b) in schema.planes.iter().zip(&buffers) {
            if p.elements != b.elements() || p.dtype != b.dtype() {
                return Err("allocated raw type/extent mismatch".into());
            }
            actual_bytes = actual_bytes
                .checked_add(
                    b.elements()
                        .checked_mul(b.dtype().bytes())
                        .ok_or("actual raw byte overflow")?,
                )
                .ok_or("actual raw sum overflow")?;
        }
        if actual_bytes != schema.requested_bytes {
            return Err("allocator raw payload accounting mismatch".into());
        }
        let after = memory()?;
        if after[0] < reserve_bytes {
            return Err("typed raw allocation exceeded measured reserve".into());
        }
        Ok(Self {
            device,
            schema,
            buffers,
            memory_before: before,
            memory_after: after,
        })
    }
    fn drain(&self) -> Result<(), String> {
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        self.device.synchronize().map_err(|e| e.to_string())
    }
    fn inventory(&self) -> Vec<(Plane, u64)> {
        self.schema
            .planes
            .iter()
            .cloned()
            .zip(self.buffers.iter().map(TypedBuffer::address))
            .collect()
    }
    unsafe fn invalidate(&self, stream: sys::CUstream) -> Result<(), String> {
        self.device.bind_to_thread().map_err(|e| e.to_string())?;
        let mut context = std::ptr::null_mut();
        check(
            sys::lib().cuStreamGetCtx(stream, &mut context),
            "raw invalidation stream context",
        )?;
        if context != *self.device.cu_primary_ctx() {
            return Err("foreign raw invalidation stream owner".into());
        }
        for b in &self.buffers {
            match b {
                TypedBuffer::Bf16(x) => check(
                    sys::lib().cuMemsetD16Async(*x.device_ptr(), 0x7fc1, x.len(), stream),
                    "BF16 raw invalidation",
                )?,
                TypedBuffer::ScalarF32(x) => check(
                    sys::lib().cuMemsetD32Async(*x.device_ptr(), KDA_PENDING_INVALID_F32_BITS, x.len(), stream),
                    "F32 scalar raw invalidation",
                )?,
            }
        }
        Ok(())
    }
}
fn main() {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        eprintln!("Use ./dev");
        std::process::exit(2);
    }
    eprintln!("Private cold typed raw contract only. Use ./dev test-contract. No serving/model admission.");
    std::process::exit(2);
}
