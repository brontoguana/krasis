// Exact retained native Marlin ABI and workspace clearing policy, for cost controls only.
type MarlinMmFn = unsafe extern "C" fn(
    *const std::ffi::c_void,
    *const std::ffi::c_void,
    *mut std::ffi::c_void,
    *mut std::ffi::c_void,
    *const std::ffi::c_void,
    *const std::ffi::c_void,
    *const std::ffi::c_void,
    *const std::ffi::c_void,
    *const std::ffi::c_void,
    *mut std::ffi::c_void,
    i32,
    i32,
    i32,
    i32,
    *mut std::ffi::c_void,
    *const ScalarType,
    bool,
    bool,
    bool,
    i32,
    i32,
    i32,
    u64,
    i32,
    i32,
    i32,
    bool,
    bool,
    bool,
);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ScalarType {
    pub exponent: u8,
    pub mantissa: u8,
    pub is_signed: u8,
    pub _pad0: u8,
    pub bias: i32,
    pub finite_values_only: u8,
    pub nan_repr: u8,
    pub _pad1: [u8; 2],
}

impl ScalarType {
    pub const U4: ScalarType = ScalarType {
        exponent: 0,
        mantissa: 4,
        is_signed: 0,
        _pad0: 0,
        bias: 0,
        finite_values_only: 0,
        nan_repr: 1,
        _pad1: [0; 2], // nan_repr=1 = NAN_IEEE_754
    };
    pub const U8: ScalarType = ScalarType {
        exponent: 0,
        mantissa: 8,
        is_signed: 0,
        _pad0: 0,
        bias: 0,
        finite_values_only: 0,
        nan_repr: 1,
        _pad1: [0; 2], // nan_repr=1 = NAN_IEEE_754
    };
    pub const U4B8: ScalarType = ScalarType {
        exponent: 0,
        mantissa: 4,
        is_signed: 0,
        _pad0: 0,
        bias: 8,
        finite_values_only: 0,
        nan_repr: 1,
        _pad1: [0; 2], // nan_repr=1 = NAN_IEEE_754
    };
    pub const U8B128: ScalarType = ScalarType {
        exponent: 0,
        mantissa: 8,
        is_signed: 0,
        _pad0: 0,
        bias: 128,
        finite_values_only: 0,
        nan_repr: 1,
        _pad1: [0; 2], // nan_repr=1 = NAN_IEEE_754
    };
}


    fn marlin_ctmp_clear_required(m: usize, n: usize) -> bool {
        if m == 0 || n == 0 {
            return false;
        }

        // The vendored Marlin wrapper enables the atomic-add reduction path only
        // for small split shapes. In the non-atomic path, the first reduction
        // slice writes C_tmp before later slices read it, so stale C_tmp contents
        // are irrelevant and clearing m*n FP32 elements is pure overhead.
        let max_thread_m_blocks = 4usize;
        let split_granularity = max_thread_m_blocks * 16;
        let max_par_local = if n <= 4096 { 16 * 8 } else { 16 };
        let mut rest_m = m;
        while rest_m > 0 {
            let par_count = (rest_m / split_granularity).min(max_par_local);
            let split_m = if par_count > 0 {
                par_count * split_granularity
            } else {
                rest_m
            };
            if split_m.div_ceil(64) * n <= 2048 {
                return true;
            }
            rest_m -= split_m;
        }
        false
    }
