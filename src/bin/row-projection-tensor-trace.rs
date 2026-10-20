// Private matched operand observer; no serving dispatch.
mod observer {
    #![allow(dead_code)]
    include!("row-projection-tensor-probe.rs");
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct TraceParameters {
        packed: u64,
        scales: u64,
        zeros: u64,
        input: u64,
        output: u64,
        output_f32: u64,
        f32_components: u64,
        tf32_components: u64,
        tf32_input: u64,
        trajectory: u64,
        rows: i32,
        cols: i32,
        group: i32,
        packed_stride: i32,
        scales_stride: i32,
        reserved_zero: i32,
        tokens: u64,
    }
    // Plain integer/pointer-address fields; CUDA mirrors repr(C) and checks sizeof.
    unsafe impl cudarc::driver::DeviceRepr for TraceParameters {}
    const _: [(); 112] = [(); std::mem::size_of::<TraceParameters>()];
    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct TracePlan {
        schema: u32,
        source: Identity,
        ptx: Identity,
        build_report: Identity,
        selection: Identity,
        capture: BTreeMap<String, Identity>,
        expected_f32: Identity,
        expected_bf16: Identity,
        gpu_name: String,
        gpu_uuid: String,
        compute_capability: u32,
        safety_margin_bytes: usize,
        kernel_contract: String,
    }
    fn checked_identity(out: &Path, name: &str, bytes: &[u8]) -> Result<Identity, String> {
        let path = out.join(name);
        fs::write(&path, bytes).map_err(|e| e.to_string())?;
        Ok(Identity {
            path,
            bytes: bytes.len(),
            sha256: format!("{:x}", Sha256::digest(bytes)),
        })
    }
    fn trace(plan: &Path, out: &Path) -> Result<Value, String> {
        let p: TracePlan = serde_json::from_slice(&fs::read(plan).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        if p.schema != 1
            || p.kernel_contract != "tf32_three_component_fixed_k32_v1"
            || p.safety_margin_bytes != 600 * 1024 * 1024
        {
            return Err("explicit diagnostic arithmetic/reserve admission required".into());
        }
        p.source.read()?;
        let ptx = p.ptx.read()?;
        let built: Value =
            serde_json::from_slice(&p.build_report.read()?).map_err(|e| e.to_string())?;
        if built["plan"]["source"] != serde_json::to_value(&p.source).map_err(|e| e.to_string())?
            || built["ptx"] != serde_json::to_value(&p.ptx).map_err(|e| e.to_string())?
        {
            return Err("source/PTX/build identity mismatch".into());
        }
        let selection: Value =
            serde_json::from_slice(&p.selection.read()?).map_err(|e| e.to_string())?;
        let get_input = |key: &str| {
            p.capture
                .get(key)
                .ok_or(format!("missing {key}"))
                .and_then(Identity::read)
        };
        let desc: Value =
            serde_json::from_slice(&get_input("descriptor")?).map_err(|e| e.to_string())?;
        let (r, k, g, ps, ss, _zs, h, row) = dims(&desc)?;
        if r > 12
            || h != 1
            || row != 0
            || k % 32 != 0
            || selection["selected_coordinates"].as_array().map(Vec::len) != Some(r)
        {
            return Err("bounded selected-row observer geometry rejected".into());
        }
        let packed = get_input("packed")?;
        let sb = get_input("scales")?;
        let zb = get_input("zeros")?;
        let xb = get_input("selected_input")?;
        if packed.len() != mul(r, ps)?
            || sb.len() != mul(r, ss)?
            || zb.len() != mul(r, ss)?
            || xb.len() != mul(k, 2)?
        {
            return Err("operand allocation size mismatch".into());
        }
        let scales = f32s(&sb)?;
        let zeros = f32s(&zb)?;
        let x = u16s(&xb)?;
        let ef = p.expected_f32.read()?;
        let eb = p.expected_bf16.read()?;
        if ef.len() != mul(r, 4)? || eb.len() != mul(r, 2)? {
            return Err("qualification reference extent mismatch".into());
        }
        f32s(&ef)?;
        u16s(&eb)?;
        let dev = CudaDevice::new_with_stream(0).map_err(|e| e.to_string())?;
        if (*dev.cu_stream()).is_null() {
            return Err("observer requires explicit CUDA stream".into());
        }
        let mut uuid = unsafe { std::mem::zeroed::<cudarc::driver::sys::CUuuid>() };
        unsafe {
            cudarc::driver::sys::lib()
                .cuDeviceGetUuid_v2(&mut uuid, *dev.cu_device())
                .result()
                .map_err(|e| e.to_string())?
        };
        let hx = uuid
            .bytes
            .iter()
            .map(|&x| format!("{:02x}", x as u8))
            .collect::<String>();
        let actual_uuid = format!(
            "GPU-{}-{}-{}-{}-{}",
            &hx[..8],
            &hx[8..12],
            &hx[12..16],
            &hx[16..20],
            &hx[20..]
        );
        use cudarc::driver::sys::CUdevice_attribute as A;
        let attr = |a| dev.attribute(a).map_err(|e| e.to_string());
        let cc = attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? * 10
            + attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        if actual_uuid != p.gpu_uuid
            || dev.name().map_err(|e| e.to_string())? != p.gpu_name
            || cc as u32 != p.compute_capability
            || cc < 80
        {
            return Err("actual GPU identity differs from admission".into());
        }
        let limits = [
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
            attr(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
        ]
        .map(|x| x as usize);
        let cf = tensor_grid(r, 1, limits)?;
        if cf.grid_dim != (1, 1, 1) {
            return Err("observer must admit one CTA".into());
        }
        dev.load_ptx(
            cudarc::nvrtc::Ptx::from_src(std::str::from_utf8(&ptx).map_err(|e| e.to_string())?),
            "trace_operator",
            &["hqq6_trace_tf32x3"],
        )
        .map_err(|e| e.to_string())?;
        let kernel = dev
            .get_func("trace_operator", "hqq6_trace_tf32x3")
            .ok_or("observer kernel missing")?;
        let plane = mul(r, k)?;
        let components = mul(plane, 4)?;
        let converted = mul(plane, 3)?;
        let steps = k / 8;
        let trajectory = mul(mul(steps, 3)?, 256)?;
        let guard = 32usize;
        let allocation_bytes = [
            packed.len(),
            sb.len(),
            zb.len(),
            xb.len(),
            mul(add(r, guard)?, 6)?,
            mul(add(components, guard)?, 4)?,
            mul(add(converted, guard)?, 4)?,
            mul(add(k, guard)?, 4)?,
            mul(add(trajectory, guard)?, 4)?,
            112,
        ]
        .iter()
        .try_fold(0usize, |a, &b| add(a, b))?;
        let before = cudarc::driver::result::mem_get_info()
            .map_err(|e| e.to_string())?
            .0;
        if before < add(allocation_bytes, p.safety_margin_bytes)? {
            return Err("measured VRAM cannot cover observer allocations plus reserve".into());
        }
        let dp = dev.htod_copy(packed).map_err(|e| e.to_string())?;
        let ds = dev.htod_copy(scales).map_err(|e| e.to_string())?;
        let dz = dev.htod_copy(zeros).map_err(|e| e.to_string())?;
        let dx = dev.htod_copy(x).map_err(|e| e.to_string())?;
        let dy = dev
            .htod_copy(vec![0x7fc1u16; r + guard])
            .map_err(|e| e.to_string())?;
        let poison = f32::from_bits(0x7fc1a123);
        let df = dev
            .htod_copy(vec![poison; r + guard])
            .map_err(|e| e.to_string())?;
        let dw = dev
            .htod_copy(vec![poison; components + guard])
            .map_err(|e| e.to_string())?;
        let dt = dev
            .htod_copy(vec![0xdeadbeefu32; converted + guard])
            .map_err(|e| e.to_string())?;
        let db = dev
            .htod_copy(vec![0xdeadbeefu32; k + guard])
            .map_err(|e| e.to_string())?;
        let dc = dev
            .htod_copy(vec![poison; trajectory + guard])
            .map_err(|e| e.to_string())?;
        let params = TraceParameters {
            packed: *dp.device_ptr(),
            scales: *ds.device_ptr(),
            zeros: *dz.device_ptr(),
            input: *dx.device_ptr(),
            output: *dy.device_ptr(),
            output_f32: *df.device_ptr(),
            f32_components: *dw.device_ptr(),
            tf32_components: *dt.device_ptr(),
            tf32_input: *db.device_ptr(),
            trajectory: *dc.device_ptr(),
            rows: r as i32,
            cols: k as i32,
            group: g as i32,
            packed_stride: ps as i32,
            scales_stride: ss as i32,
            reserved_zero: 0,
            tokens: 1,
        };
        let cfg = dev.htod_copy(vec![params]).map_err(|e| e.to_string())?;
        let _completion = Completion(&dev);
        dev.synchronize().map_err(|e| e.to_string())?;
        let after_allocation = cudarc::driver::result::mem_get_info()
            .map_err(|e| e.to_string())?
            .0;
        if after_allocation < p.safety_margin_bytes {
            return Err("observer allocations breach measured reserve".into());
        }
        unsafe { kernel.launch(cf, (&cfg,)).map_err(|e| e.to_string())? };
        dev.synchronize().map_err(|e| e.to_string())?;
        let after_run = cudarc::driver::result::mem_get_info()
            .map_err(|e| e.to_string())?
            .0;
        let y = dev.dtoh_sync_copy(&dy).map_err(|e| e.to_string())?;
        let f = dev.dtoh_sync_copy(&df).map_err(|e| e.to_string())?;
        let w = dev.dtoh_sync_copy(&dw).map_err(|e| e.to_string())?;
        let t = dev.dtoh_sync_copy(&dt).map_err(|e| e.to_string())?;
        let b = dev.dtoh_sync_copy(&db).map_err(|e| e.to_string())?;
        let c = dev.dtoh_sync_copy(&dc).map_err(|e| e.to_string())?;
        // Preserve every raw allocation including guards before qualification or rejection.
        let raw_y = save(out, "observed-with-guard.bf16", &y)?;
        let raw_f = save_f32(out, "observed-with-guard.f32", &f)?;
        let raw_w = save_f32(out, "components-with-guard.f32", &w)?;
        let raw_c = save_f32(out, "trajectory-with-guard.f32", &c)?;
        let raw_t = checked_identity(
            out,
            "converted-components-with-guard.u32",
            &t.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
        )?;
        let raw_b = checked_identity(
            out,
            "converted-input-with-guard.u32",
            &b.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
        )?;
        let fbytes = f[..r]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>();
        let ybytes = y[..r]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>();
        if fbytes != ef || ybytes != eb {
            return Err("observer final F32/BF16 differs from 505; attribution prohibited".into());
        }
        if y[r..].iter().any(|&x| x != 0x7fc1)
            || [(&f, r), (&w, components), (&c, trajectory)]
                .iter()
                .any(|(xs, n)| xs[*n..].iter().any(|x| x.to_bits() != 0x7fc1a123))
        {
            return Err("F32/BF16 observer guard overwritten".into());
        }
        if [(&t, converted), (&b, k)].iter().any(|(xs, n)| {
            xs[*n..].iter().any(|&x| x != 0xdeadbeef) || xs[..*n].contains(&0xdeadbeef)
        }) {
            return Err("opaque converted-storage guard/unwritten failure".into());
        }
        if f[..r]
            .iter()
            .chain(w[..components].iter())
            .chain(c[..trajectory].iter())
            .any(|x| !x.is_finite())
        {
            return Err("nonfinite or unwritten observer allocation".into());
        }
        if after_run < p.safety_margin_bytes {
            return Err("completed observer breached reserve".into());
        }
        p.source.read()?;
        p.ptx.read()?;
        p.selection.read()?;
        for x in p.capture.values() {
            x.read()?;
        }
        Ok(
            json!({"passed":true,"diagnostic_only":true,"adopted":false,"final_matches_505_f32_bf16_exact":true,"plan":p,"selection":selection,"geometry":{"rows":r,"cols":k,"steps_k8":steps,"matrix_rows":16,"matrix_cols":16},"raw":{"bf16":raw_y,"f32":raw_f,"components":raw_w,"converted_components":raw_t,"converted_input":raw_b,"trajectory":raw_c},"guards_exact":true,"finite_preconversion_and_accumulator_buffers":true,"memory":{"allocated_tensor_bytes":allocation_bytes,"before":before,"after_allocation":after_allocation,"after_run":after_run,"sampled_completed_boundary_minimum":before.min(after_allocation).min(after_run),"low_water_measurement":false,"safety_margin_bytes":p.safety_margin_bytes},"limitations":"instrumented single selected token/subset observer; opaque TF32 storage is not scalar F32; no performance, graph or serving acceptance"}),
        )
    }
    pub(super) fn run_observer() -> Result<(), String> {
        if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
            return Err("use built ./dev row-projection-tensor-trace".into());
        }
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        if args.len() != 3 {
            return Err("usage: build|trace PLAN NEW_OUTPUT".into());
        }
        let out = Path::new(&args[2]);
        if out.exists() {
            return Err("output attempt already exists".into());
        }
        fs::create_dir_all(out).map_err(|e| e.to_string())?;
        let result = match args[0].as_str() {
            "build" => build(Path::new(&args[1]), out),
            "trace" => trace(Path::new(&args[1]), out),
            _ => Err("unknown built command".into()),
        };
        match result {
            Ok(r) => {
                fs::write(
                    out.join("REPORT.json"),
                    serde_json::to_vec_pretty(&r).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                println!("{}", json!({"passed":true,"output":out}));
                Ok(())
            }
            Err(e) => {
                fs::write(out.join("FAILURE.txt"), &e).map_err(|x| x.to_string())?;
                Err(e)
            }
        }
    }
}
fn main() {
    if let Err(e) = observer::run_observer() {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
