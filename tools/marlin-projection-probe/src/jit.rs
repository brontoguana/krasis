use super::*;
pub(super) fn inspect(path: &Path, out: &Path) -> Result<Value, String> {
    let plan: Plan = read_json(path)?;
    plan.source.read()?;
    let ptx = plan.ptx.read()?;
    let text = std::ffi::CString::new(ptx).map_err(|e| e.to_string())?;
    let dev = CudaDevice::new_with_stream(plan.gpu_ordinal).map_err(|e| e.to_string())?;
    let mut uuid = sys::CUuuid { bytes: [0; 16] };
    unsafe { sys::lib().cuDeviceGetUuid_v2(&mut uuid, *dev.cu_device()) }
        .result()
        .map_err(|e| e.to_string())?;
    let h = uuid
        .bytes
        .iter()
        .map(|v| format!("{:02x}", *v as u8))
        .collect::<String>();
    let actual = format!(
        "GPU-{}-{}-{}-{}-{}",
        &h[..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..]
    );
    if actual != plan.gpu_uuid {
        return Err("actual UUID mismatch".into());
    }
    let mut limits = Vec::new();
    for attribute in [
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
    ] {
        let mut v = 0;
        unsafe { sys::lib().cuDeviceGetAttribute(&mut v, attribute, *dev.cu_device()) }
            .result()
            .map_err(|e| e.to_string())?;
        limits.push(json!({"attribute":format!("{attribute:?}"),"value":v}))
    }
    let mut error = vec![0u8; 65536];
    let mut info = vec![0u8; 65536];
    let mut options = [
        sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER,
        sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        sys::CUjit_option::CU_JIT_INFO_LOG_BUFFER,
        sys::CUjit_option::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
    ];
    let mut values = [
        error.as_mut_ptr().cast(),
        (error.len() as usize) as *mut std::ffi::c_void,
        info.as_mut_ptr().cast(),
        (info.len() as usize) as *mut std::ffi::c_void,
    ];
    let mut module = std::ptr::null_mut();
    let status = unsafe {
        sys::lib().cuModuleLoadDataEx(
            &mut module,
            text.as_ptr().cast(),
            options.len() as u32,
            options.as_mut_ptr(),
            values.as_mut_ptr(),
        )
    };
    let decode = |b: &[u8]| {
        String::from_utf8_lossy(&b[..b.iter().position(|&v| v == 0).unwrap_or(b.len())])
            .into_owned()
    };
    let error = decode(&error);
    let info = decode(&info);
    fs::write(out.join("JIT_ERROR.txt"), &error).map_err(|e| e.to_string())?;
    fs::write(out.join("JIT_INFO.txt"), &info).map_err(|e| e.to_string())?;
    let function_stats = (|| -> Result<Value, String> {
        if module.is_null() {
            return Ok(Value::Null);
        }
        let symbol = std::ffi::CString::new("marlin_projection_tensorcomp").unwrap();
        let mut function = std::ptr::null_mut();
        unsafe { sys::lib().cuModuleGetFunction(&mut function, module, symbol.as_ptr()) }
            .result()
            .map_err(|e| e.to_string())?;
        let mut values = Vec::new();
        for attribute in [
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS,
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        ] {
            let mut v = 0;
            unsafe { sys::lib().cuFuncGetAttribute(&mut v, attribute, function) }
                .result()
                .map_err(|e| e.to_string())?;
            values.push(json!({"attribute":format!("{attribute:?}"),"value":v}))
        }
        let mut blocks = 0;
        unsafe {
            sys::lib().cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut blocks, function, 256, 0)
        }
        .result()
        .map_err(|e| e.to_string())?;
        Ok(json!({"attributes":values,"active256ThreadBlocksPerSm":blocks}))
    })();
    if !module.is_null() {
        unsafe { sys::lib().cuModuleUnload(module) }
            .result()
            .map_err(|e| e.to_string())?
    }
    Ok(
        json!({"collectionPassed":true,"moduleAdmitted":status==sys::CUresult::CUDA_SUCCESS,"status":format!("{status:?}"),"gpuUuid":actual,"limits":limits,"error":error,"info":info,"functionStats":function_stats?,"scope":"Actual driver JIT diagnostics only; no kernel/model execution or fallback."}),
    )
}
