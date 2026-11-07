//! DLPack retains the native allocation owner and supplies an explicit CUDA
//! device even when a virtual address is physically backed by host memory.
use super::Backing;
use pyo3::{ffi, prelude::*};
use std::{ffi::c_void, sync::Arc};
#[repr(C)]
struct Device {
    kind: i32,
    id: i32,
}
#[repr(C)]
struct Dtype {
    code: u8,
    bits: u8,
    lanes: u16,
}
#[repr(C)]
struct Tensor {
    data: *mut c_void,
    device: Device,
    ndim: i32,
    dtype: Dtype,
    shape: *mut i64,
    strides: *mut i64,
    offset: u64,
}
#[repr(C)]
struct Managed {
    tensor: Tensor,
    context: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut Managed)>,
}
struct Owner {
    managed: Managed,
    _shape: Vec<i64>,
    _backing: Arc<Backing>,
}
unsafe extern "C" fn delete(t: *mut Managed) {
    if !t.is_null() {
        let p = (*t).context as *mut Owner;
        if !p.is_null() {
            drop(Box::from_raw(p));
        }
    }
}
unsafe extern "C" fn capsule_delete(c: *mut ffi::PyObject) {
    if ffi::PyCapsule_IsValid(c, b"dltensor\0".as_ptr().cast()) != 0 {
        let t = ffi::PyCapsule_GetPointer(c, b"dltensor\0".as_ptr().cast()) as *mut Managed;
        if !t.is_null() {
            delete(t);
        }
    }
}
pub(super) fn export(py: Python<'_>, backing: Arc<Backing>, shape: &[usize]) -> PyResult<PyObject> {
    backing
        .device
        .bind_to_thread()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    // Export is setup/diagnostic only. Consumer streams observe initialized KV.
    backing
        .device
        .synchronize()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let mut dims: Vec<i64> = shape
        .iter()
        .map(|&n| i64::try_from(n))
        .collect::<Result<_, _>>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("DLPack shape exceeds i64"))?;
    let t = Tensor {
        data: backing.ptr as *mut c_void,
        device: Device {
            kind: 2,
            id: backing.ordinal,
        },
        ndim: dims.len() as i32,
        dtype: Dtype {
            code: 1,
            bits: 8,
            lanes: 1,
        },
        shape: dims.as_mut_ptr(),
        strides: std::ptr::null_mut(),
        offset: 0,
    };
    let mut owner = Box::new(Owner {
        managed: Managed {
            tensor: t,
            context: std::ptr::null_mut(),
            deleter: Some(delete),
        },
        _shape: dims,
        _backing: backing,
    });
    let ptr = &mut *owner as *mut Owner;
    owner.managed.context = ptr.cast();
    let tensor = &mut owner.managed as *mut Managed;
    let _ = Box::into_raw(owner);
    unsafe {
        let capsule = ffi::PyCapsule_New(
            tensor.cast(),
            b"dltensor\0".as_ptr().cast(),
            Some(capsule_delete),
        );
        if capsule.is_null() {
            delete(tensor);
            return Err(PyErr::fetch(py));
        }
        Ok(PyObject::from_owned_ptr(py, capsule))
    }
}
