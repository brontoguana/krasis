//! Explicit per-CUDA-context native FLA scalar-gate ABI; unchanged cubins.
use cudarc::driver::sys;
use std::{
    ffi::{c_char, c_void, CStr, CString},
    path::Path,
    sync::Arc,
};
pub type Kkt = unsafe extern "C" fn(
    *mut c_void,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    i32,
    u32,
    u32,
    u32,
    *mut c_void,
) -> u32;
pub type Solve =
    unsafe extern "C" fn(*mut c_void, u64, u64, u64, u64, i32, u32, u32, u32, *mut c_void) -> u32;
pub type Wy = unsafe extern "C" fn(
    *mut c_void,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    i32,
    u32,
    u32,
    u32,
    *mut c_void,
) -> u32;
pub type State = unsafe extern "C" fn(
    *mut c_void,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    i32,
    u32,
    u32,
    u32,
    *mut c_void,
) -> u32;
pub type Output = unsafe extern "C" fn(
    *mut c_void,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    f32,
    i32,
    u32,
    u32,
    u32,
    *mut c_void,
) -> u32;
type Destroy = unsafe extern "C" fn(*mut c_void) -> u32;
struct Context {
    opaque: *mut c_void,
    cuda_context: sys::CUcontext,
    destroy: Destroy,
}
// SAFETY: after successful cold creation the ABI2 wrapper reads only immutable
// module/function/context handles; launch argument arrays are call-local. CUDA
// driver handles are not thread-affine. Every caller binds the retained actual
// context on the receiving thread, serializes conflicting state/workspace uses,
// and retains that context through completion and final destruction (load's
// unsafe contract). Destruction requires the final Arc and binds its context.
// This grants handle transport, not concurrent access to recurrent state.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}
impl Drop for Context {
    fn drop(&mut self) {
        let cleanup = (|| -> Result<(), String> {
            let mut previous = std::ptr::null_mut();
            unsafe { sys::lib().cuCtxGetCurrent(&mut previous) }
                .result()
                .map_err(|e| e.to_string())?;
            unsafe { sys::lib().cuCtxSetCurrent(self.cuda_context) }
                .result()
                .map_err(|e| e.to_string())?;
            let code = unsafe { (self.destroy)(self.opaque) };
            if code != 0 {
                return Err(format!("native FLA context destroy returned{code}"));
            }
            self.opaque = std::ptr::null_mut();
            unsafe { sys::lib().cuCtxSetCurrent(previous) }
                .result()
                .map_err(|e| e.to_string())?;
            Ok(())
        })();
        if let Err(e) = cleanup {
            eprintln!("KRASIS FATAL: native FLA context release unknown:{e}; library/context owners cannot be safely released");
            std::process::exit(crate::cuda_failure_code::CUDA_CONTEXT_POISONED_EXIT_CODE);
        }
    }
}
struct Library {
    context: Context,
    artifact_identity: String,
    _handle: libloading::Library,
}
#[derive(Clone)]
pub struct Native {
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub kkt: Kkt,
    pub solve: Solve,
    pub wy: Wy,
    pub state: State,
    pub output: Output,
    // Context destroys modules before releasing their library code.
    shared: Arc<Library>,
}
impl Native {
    pub fn artifact_identity(&self) -> &str {
        &self.shared.artifact_identity
    }
    pub fn handle(&self) -> *mut c_void {
        self.shared.context.opaque
    }
    pub fn context(&self) -> sys::CUcontext {
        self.shared.context.cuda_context
    }
    unsafe fn symbol<T: Copy>(h: &libloading::Library, name: &str) -> Result<T, String> {
        let n = CString::new(name).map_err(|e| e.to_string())?;
        h.get::<T>(n.as_bytes_with_nul())
            .map(|symbol| *symbol)
            .map_err(|e| format!("required explicit-context FLA symbol absent:{name}: {e}"))
    }
    /// Native scalar64/key128/value128 capabilities are unchanged. Missing new
    /// context API is an explicit error; legacy global init is never selected.
    ///
    /// # Safety
    /// Caller retains the actual current CUDA context through this Native's Drop,
    /// and completes every queued use before its module/library release. Owner
    /// retains its real CudaDevice last; the raw-secondary test retains its actual
    /// context guard through explicit Native destruction. On any receiving
    /// thread bind this actual context before dispatch. Serialize conflicting
    /// state/workspace uses and retain their owners through queued completion.
    pub unsafe fn load(
        path: &Path,
        heads: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> Result<Self, String> {
        if heads == 0 || key_dim != 128 || value_dim != 128 {
            return Err("loaded scalar FLA ABI requires positive heads and key/value128".into());
        }
        let mut cuda_context = std::ptr::null_mut();
        sys::lib()
            .cuCtxGetCurrent(&mut cuda_context)
            .result()
            .map_err(|e| e.to_string())?;
        if cuda_context.is_null() {
            return Err("native FLA owner requires real current CUDA context".into());
        }
        let handle = crate::gpu_prefill::open_owned_sidecar_library(path, "scalar native64 FLA")?;
        let h = &handle;
        let version: unsafe extern "C" fn() -> u32 =
            Self::symbol(h, "krasis_fla_context_abi_version")?;
        let actual = version();
        if actual != 2 {
            return Err(format!("unsupported FLA context ABI:{actual}"));
        }
        let identity: unsafe extern "C" fn() -> *const c_char =
            Self::symbol(h, "krasis_fla_artifact_identity")?;
        let identity = identity();
        if identity.is_null() {
            return Err("loaded FLA artifact identity is null".into());
        }
        let artifact_identity = CStr::from_ptr(identity)
            .to_str()
            .map_err(|e| format!("loaded FLA artifact identity:{e}"))?
            .to_owned();
        if artifact_identity.len() != 64
            || !artifact_identity
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("loaded FLA artifact identity is not a SHA256 contract digest".into());
        }
        let create: unsafe extern "C" fn(*mut *mut c_void) -> u32 =
            Self::symbol(h, "krasis_fla_context_create")?;
        let destroy: Destroy = Self::symbol(h, "krasis_fla_context_destroy")?;
        let mut opaque = std::ptr::null_mut();
        let code = create(&mut opaque);
        if code != 0 {
            if !opaque.is_null() {
                eprintln!("KRASIS FATAL: native FLA initialization returned an incompletely cleaned resource; context/library release is unsafe");
                std::process::exit(crate::cuda_failure_code::CUDA_CONTEXT_POISONED_EXIT_CODE);
            }
            return Err(format!("native FLA context creation returned{code}"));
        }
        if opaque.is_null() {
            return Err("native FLA context creation returned null after success".into());
        }
        let context = Context {
            opaque,
            cuda_context,
            destroy,
        };
        let shared = Arc::new(Library {
            context,
            artifact_identity,
            _handle: handle,
        });
        Self::bind(shared, heads, key_dim, value_dim)
    }
    pub fn for_geometry(
        &self,
        heads: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> Result<Self, String> {
        unsafe { Self::bind(self.shared.clone(), heads, key_dim, value_dim) }
    }
    unsafe fn bind(
        shared: Arc<Library>,
        heads: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> Result<Self, String> {
        if heads == 0 || key_dim != 128 || value_dim != 128 {
            return Err("loaded scalar FLA geometry is outside native capability".into());
        }
        let h = &shared._handle;
        Ok(Self {
            heads,
            key_dim,
            value_dim,
            kkt: Self::symbol(
                h,
                &format!("krasis_fla_context_chunk_scaled_dot_kkt_fwd_kernel_h{heads}"),
            )?,
            solve: Self::symbol(
                h,
                &format!("krasis_fla_context_merge_16x16_to_64x64_inverse_kernel_h{heads}"),
            )?,
            wy: Self::symbol(
                h,
                &format!("krasis_fla_context_recompute_w_u_fwd_kernel_h{heads}"),
            )?,
            state: Self::symbol(
                h,
                &format!(
                    "krasis_fla_context_chunk_gated_delta_rule_fwd_kernel_h_blockdim64_h{heads}"
                ),
            )?,
            output: Self::symbol(
                h,
                &format!("krasis_fla_context_chunk_fwd_kernel_o_h{heads}"),
            )?,
            shared,
        })
    }
}
