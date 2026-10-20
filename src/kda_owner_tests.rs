fn vector_bf16_array(o: &PreparedOwner, n: usize) -> Result<Vec<u8>, String> {
    if n > o.arena.bf16_emitted.len() {
        return Err("vector BF16 read extent".into());
    }
    let mut b = vec![0u8; n * 2];
    check(
        unsafe {
            sys::lib().cuMemcpyDtoH_v2(
                b.as_mut_ptr().cast(),
                pointer(&o.arena.bf16_emitted),
                b.len(),
            )
        },
        "vector explicit BF16 diagnostic read",
    )?;
    Ok(b)
}
fn vector_rne_view(v: &[u8]) -> Result<Vec<u8>, String> {
    if v.len() % 4 != 0 {
        return Err("vector F32 emission extent".into());
    }
    let mut b = Vec::with_capacity(v.len() / 2);
    for x in v.chunks_exact(4) {
        let bits = u32::from_le_bytes(x.try_into().unwrap());
        if bits & 0x7f800000 == 0x7f800000 {
            return Err("vector nonfinite BF16 reference input".into());
        }
        let rounded = ((u64::from(bits) + 0x7fff + u64::from((bits >> 16) & 1)) >> 16) as u16;
        if rounded & 0x7f80 == 0x7f80 {
            return Err("vector BF16 reference overflow".into());
        }
        b.extend_from_slice(&rounded.to_le_bytes());
    }
    Ok(b)
}
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct OwnerId {
    path: PathBuf,
    bytes: usize,
    sha256: String,
}
fn owner_checked(x: &OwnerId) -> Result<Vec<u8>, String> {
    let b = fs::read(&x.path).map_err(|e| e.to_string())?;
    if b.len() != x.bytes || format!("{:x}", Sha256::digest(&b)) != x.sha256 {
        return Err(format!("owner identity {}", x.path.display()));
    }
    Ok(b)
}
fn owner_save(dir: &Path, name: &str, b: &[u8]) -> Result<OwnerId, String> {
    use std::io::Write;
    let path = dir.join(name);
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .and_then(|mut f| f.write_all(b))
        .map_err(|e| e.to_string())?;
    Ok(OwnerId {
        path,
        bytes: b.len(),
        sha256: format!("{:x}", Sha256::digest(b)),
    })
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnerPlan {
    diagnostic_only: bool,
    qualifies_candidate: bool,
    executable: OwnerId,
    driver: OwnerId,
    runtime_config: OwnerId,
    cuda_index: usize,
    uuid: String,
    compute: [i32; 2],
    heads: usize,
    dim: usize,
    rows: usize,
    inputs: Vec<OwnerId>,
    transport_ptx: OwnerId,
    math_ptx: OwnerId,
    admission: Vec<OwnerId>,
    references: BTreeMap<String, OwnerId>,
}
fn upload_raw(o: &mut RawOwner, bytes: &[Vec<u8>]) -> Result<(), String> {
    if bytes.len() != 5 {
        return Err("owner preparedplanes".into());
    }
    for (b, src) in o.buffers.iter_mut().zip(bytes) {
        match b {
            TypedBuffer::Bf16(x) => {
                if src.len() != x.len() * 2 {
                    return Err("prepared BF16 extent".into());
                }
                let v = src
                    .chunks_exact(2)
                    .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
                    .collect::<Vec<_>>();
                o.device
                    .htod_sync_copy_into(&v, x)
                    .map_err(|e| e.to_string())?;
            }
            TypedBuffer::ScalarF32(x) => {
                if src.len() != x.len() * 4 {
                    return Err("prepared F32 extent".into());
                }
                let v = src
                    .chunks_exact(4)
                    .map(|x| f32::from_bits(u32::from_le_bytes(x.try_into().unwrap())))
                    .collect::<Vec<_>>();
                o.device
                    .htod_sync_copy_into(&v, x)
                    .map_err(|e| e.to_string())?;
            }
        }
    }
    Ok(())
}
fn raw_slice(v: &[Vec<u8>], at: usize, rows: usize, h: usize, d: usize) -> Vec<Vec<u8>> {
    let widths = [h * d * 4, h * d * 4, h * d * 4, h * d * 4, h * 4];
    v.iter()
        .zip(widths)
        .map(|(b, w)| b[at * w..(at + rows) * w].to_vec())
        .collect()
}
fn owner_array(o: &PreparedOwner, name: &str, n: usize) -> Result<Vec<u8>, String> {
    let x = o.array(name);
    if n > x.len() {
        return Err("owner readback extent".into());
    }
    let mut b = vec![0; n * 4];
    check(
        unsafe { sys::lib().cuMemcpyDtoH_v2(b.as_mut_ptr().cast(), pointer(x), b.len()) },
        "owner selectedcomplete array readback",
    )?;
    Ok(b)
}
fn owner_compare(got: &[u8], want: &[u8], label: &str) -> Result<(), String> {
    if got != want {
        let index = got.iter().zip(want).position(|(a, b)| a != b);
        return Err(format!(
            "owner full-bit mismatch {label} {:?} gotbytes={} wantbytes={}",
            index,
            got.len(),
            want.len()
        ));
    }
    Ok(())
}
fn owner_metadata(o: &PreparedOwner) -> Result<(i32, i32, u64), String> {
    o.stream.drain()?;
    let mut p = 0i32;
    let mut e = 0i32;
    let mut at = 0u64;
    check(
        unsafe { sys::lib().cuMemcpyDtoH_v2((&mut p as *mut i32).cast(), pointer(&o.pending), 4) },
        "native pendingread",
    )?;
    check(
        unsafe { sys::lib().cuMemcpyDtoH_v2((&mut e as *mut i32).cast(), pointer(&o.error), 4) },
        "native errorread",
    )?;
    check(
        unsafe {
            sys::lib().cuMemcpyDtoH_v2((&mut at as *mut u64).cast(), pointer(&o.absolute), 8)
        },
        "native absoluteread",
    )?;
    Ok((p, e, at))
}
fn allocation_bytes(a: &crate::session_cache::SequenceStateAllocation) -> Result<Vec<u8>, String> {
    let mut b = vec![0; a.storage_bytes];
    check(
        unsafe { sys::lib().cuMemcpyDtoH_v2(b.as_mut_ptr().cast(), a.ptr, b.len()) },
        "owner complete actualregistry capture",
    )?;
    Ok(b)
}
#[derive(Serialize, Deserialize)]
struct OwnerSnapshot {
    arithmetic_sha256: String,
    layout: Value,
    blobs: Vec<crate::session_cache::SequenceStateBlob>,
}
fn capture_owner(
    o: &PreparedOwner,
    r: &crate::session_cache::SequenceStateRegistry,
) -> Result<OwnerSnapshot, String> {
    use crate::session_cache::{snapshot_blob_can_restore, SequenceStateBlob};
    o.stream.drain()?;
    let mut blobs = Vec::new();
    for a in r.allocations() {
        let blob = SequenceStateBlob {
            allocation_name: a.name.clone(),
            kind: a.kind.clone(),
            layer_idx: a.layer_idx,
            device_ordinal: a.device_ordinal,
            dtype: a.dtype.clone(),
            element_size: a.element_size,
            shape: a.shape.clone(),
            strides_bytes: a.strides_bytes.clone(),
            bytes: allocation_bytes(a)?,
        };
        blob.validate()?;
        if !snapshot_blob_can_restore(a, a.storage_bytes, &blob) {
            return Err("owner snapshot ownlayout".into());
        }
        blobs.push(blob);
    }
    Ok(OwnerSnapshot {
        arithmetic_sha256: o.arithmetic_sha256.clone(),
        layout: r.compatibility_layout_value(),
        blobs,
    })
}
fn restore_owner(
    o: &PreparedOwner,
    r: &crate::session_cache::SequenceStateRegistry,
    s: &OwnerSnapshot,
) -> Result<(), String> {
    use crate::session_cache::snapshot_blob_can_restore;
    if s.arithmetic_sha256 != o.arithmetic_sha256
        || s.layout != r.compatibility_layout_value()
        || s.blobs.len() != r.allocations().len()
    {
        return Err("owner snapshot arithmetic/fullschema mismatch beforewrite".into());
    }
    for (a, b) in r.allocations().iter().zip(&s.blobs) {
        b.validate()?;
        if !snapshot_blob_can_restore(a, a.storage_bytes, b) {
            return Err("owner snapshot allblob admission beforewrite".into());
        }
    }
    for (a, b) in r.allocations().iter().zip(&s.blobs) {
        check(
            unsafe {
                sys::lib().cuMemcpyHtoDAsync_v2(
                    a.ptr,
                    b.bytes.as_ptr().cast(),
                    b.bytes.len(),
                    o.stream.handle,
                )
            },
            "owner actual typedrestore",
        )?;
    }
    o.stream.drain()
}
struct NativeCaptured<'a> {
    graph: OwnerGraph<'a>,
    _input: &'a NativeRawOwner,
}
fn native_capture<'a>(
    o: &'a PreparedOwner,
    live: &'a NativeRawOwner,
) -> Result<NativeCaptured<'a>, String> {
    let lease = o.arena.lease()?;
    o.validate_lease(&lease)?;
    let mut graph = OwnerGraph {
        owner: o,
        graph: std::ptr::null_mut(),
        exec: std::ptr::null_mut(),
    };
    check(
        unsafe {
            sys::lib().cuStreamBeginCapture_v2(
                o.stream.handle,
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        },
        "shared capturebegin",
    )?;
    let view = live.view()?;
    let capture = o.decode_capture_in(&lease, &view);
    let end = check(
        unsafe { sys::lib().cuStreamEndCapture(o.stream.handle, &mut graph.graph) },
        "shared captureend",
    );
    capture?;
    end?;
    check(
        unsafe { sys::lib().cuGraphInstantiateWithFlags(&mut graph.exec, graph.graph, 0) },
        "shared graphinstantiate",
    )?;
    let mut n = 0;
    check(
        unsafe { sys::lib().cuGraphGetNodes(graph.graph, std::ptr::null_mut(), &mut n) },
        "shared graphnodes",
    )?;
    if n != 11 + 4 {
        // Transaction/transport stages plus four new arithmetic stages.
        return Err("shared graph node contract".into());
    }
    Ok(NativeCaptured {
        graph,
        _input: live,
    })
}
fn native_upload(
    o: &PreparedOwner,
    lease: &EnqueueLease<'_>,
    live: &NativeRawOwner,
    bytes: &[Vec<u8>],
) -> Result<(), String> {
    o.validate_lease(lease)?;
    let view = live.view()?;
    o.validate_view(&view, live.schema.planes[0].shape[0])?;
    if bytes.len() != 5 {
        return Err("shared upload planes".into());
    }
    for ((desc, ptr), b) in live.inventory().into_iter().zip(bytes) {
        if b.len() != desc.storage_bytes {
            return Err("shared upload dtype/extent".into());
        }
        check(
            unsafe {
                sys::lib().cuMemcpyHtoDAsync_v2(ptr, b.as_ptr().cast(), b.len(), o.stream.handle)
            },
            "shared diagnostic raw upload",
        )?;
    }
    // Diagnostic host readback only. Hot owner dispatch performs no synchronize.
    o.stream.drain()
}
fn native_snapshot_equal(
    o: &PreparedOwner,
    r: &crate::session_cache::SequenceStateRegistry,
    want: &OwnerSnapshot,
    label: &str,
) -> Result<(), String> {
    let actual = capture_owner(o, r)?;
    owner_compare(
        &serde_json::to_vec(&actual).map_err(|e| e.to_string())?,
        &serde_json::to_vec(want).map_err(|e| e.to_string())?,
        label,
    )
}
fn native_pointer_inventory(o: &PreparedOwner) -> Vec<Value> {
    let mut v=o.arena.raw.inventory().into_iter().enumerate().map(|(i,(x,p))|json!({"role":format!("raw{i}"),"ptr":p,"bytes":x.storage_bytes,"dtype":if x.dtype==Dtype::Float32{"float32"}else{"bfloat16"}})).collect::<Vec<_>>();
    for (n, x) in &o.arena.powers {
        v.push(json!({"role":n,"ptr":pointer(x),"bytes":x.len()*4,"dtype":"int32"}));
    }
    for (n, x) in &o.arena.arrays {
        v.push(json!({"role":n,"ptr":pointer(x),"bytes":x.len()*4,"dtype":"float32"}));
    }
    for (n, x) in [
        ("count", &o.arena.count),
        ("raw_count", &o.arena.raw_count),
        ("one", &o.arena.one),
    ] {
        v.push(json!({"role":n,"ptr":pointer(x),"bytes":4,"dtype":"int32"}));
    }
    v.push(json!({"role":"bf16_emitted","ptr":pointer(&o.arena.bf16_emitted),"bytes":o.arena.bf16_emitted.len()*2,"dtype":"bfloat16"}));
    v
}

fn vector_seed(o: &PreparedOwner, initial: &[u8]) -> Result<(), String> {
    o.reset()?;
    for role in ["incoming", "complete", "latest"] {
        if initial.len() != o.array(role).len() * 4 {
            return Err("vector initial extent".into());
        }
        check(
            unsafe {
                sys::lib().cuMemcpyHtoDAsync_v2(
                    o.p(role),
                    initial.as_ptr().cast(),
                    initial.len(),
                    o.stream.handle,
                )
            },
            "vector frozen initial upload",
        )?;
    }
    o.stream.drain()
}
fn vector_native_run() -> Result<(), String> {
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        return Err("Use built ./dev".into());
    }
    let input = std::env::var("KRASIS_KDA_OWNER_PLAN").map_err(|_| "vector frozen plan")?;
    let output = std::env::var("KRASIS_KDA_OWNER_OUTPUT").map_err(|_| "vector fresh output")?;
    let p: OwnerPlan = serde_json::from_slice(&fs::read(input).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if !p.diagnostic_only || p.qualifies_candidate || p.inputs.len() != 5 || p.rows < 17 {
        return Err("vector diagnostic scope".into());
    }
    for x in &p.admission {
        owner_checked(x)?;
    }
    owner_checked(&p.executable)?;
    if fs::canonicalize(std::env::current_exe().map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?
        != fs::canonicalize(&p.executable.path).map_err(|e| e.to_string())?
    {
        return Err("vector actual executable identity".into());
    }
    owner_checked(&p.driver)?;
    let cfg = String::from_utf8(owner_checked(&p.runtime_config)?).map_err(|e| e.to_string())?;
    let reserve = cfg
        .lines()
        .find_map(|x| x.strip_prefix("CFG_VRAM_SAFETY_MARGIN="))
        .ok_or("vector configured reserve")?
        .trim_matches('"')
        .parse::<usize>()
        .map_err(|e| e.to_string())?
        .checked_mul(1024 * 1024)
        .ok_or("vector reserve overflow")?;
    let raw = p
        .inputs
        .iter()
        .map(owner_checked)
        .collect::<Result<Vec<_>, _>>()?;
    let refs = p
        .references
        .iter()
        .map(|(n, x)| Ok((n.clone(), owner_checked(x)?)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let reference = |n: &str| refs.get(n).ok_or(format!("vector reference missing {n}"));
    let initial = reference("initial")?;
    let hd = p.heads.checked_mul(p.dim).ok_or("vector HD overflow")?;
    let st = hd.checked_mul(p.dim).ok_or("vector state overflow")?;
    let out = reference("output")?;
    let states = reference("states")?;
    if initial.len() != st * 4 || out.len() != p.rows * hd * 4 || states.len() != p.rows * st * 4 {
        return Err("vector complete reference extents".into());
    }
    let root = Path::new(&output);
    fs::create_dir(root).map_err(|e| e.to_string())?;
    let device = CudaDevice::new_with_stream(p.cuda_index).map_err(|e| e.to_string())?;
    device.bind_to_thread().map_err(|e| e.to_string())?;
    let mut ordinal = 0;
    let mut uuid: sys::CUuuid = unsafe { std::mem::zeroed() };
    check(
        unsafe { sys::lib().cuCtxGetDevice(&mut ordinal) },
        "vector actual device",
    )?;
    check(
        unsafe { sys::lib().cuDeviceGetUuid_v2(&mut uuid, ordinal) },
        "vector actual UUID",
    )?;
    let u = uuid
        .bytes
        .iter()
        .map(|x| format!("{:02x}", *x as u8))
        .collect::<String>();
    let actual_uuid = format!(
        "GPU-{}-{}-{}-{}-{}",
        &u[..8],
        &u[8..12],
        &u[12..16],
        &u[16..20],
        &u[20..]
    );
    if actual_uuid != p.uuid
        || device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| e.to_string())?
            != p.compute[0]
        || device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| e.to_string())?
            != p.compute[1]
    {
        return Err("vector actual architecture".into());
    }
    let stream = NativeStream::create(device.clone())?;
    let before = memory()?;
    let transport =
        String::from_utf8(owner_checked(&p.transport_ptx)?).map_err(|e| e.to_string())?;
    let math = String::from_utf8(owner_checked(&p.math_ptx)?).map_err(|e| e.to_string())?;
    let a = PreparedOwner::new_on_stream(
        device.clone(),
        stream.clone(),
        p.heads,
        p.dim,
        p.rows,
        reserve,
        &transport,
        &math,
    )?;
    let b = PreparedOwner::from_shared(&a, reserve)?;
    let mut ra = crate::session_cache::SequenceStateRegistry::default();
    let mut rb = crate::session_cache::SequenceStateRegistry::default();
    a.register(&mut ra, 0)?;
    b.register(&mut rb, 0)?;
    let one = NativeRawOwner::allocate(
        stream.clone(),
        Schema::checked(
            RawContract::KdaPreparedF32VectorGate,
            1,
            p.heads,
            p.dim,
            p.dim,
        )?,
        reserve,
    )?;
    let captured = native_capture(&a, &one)?;
    let graph_inventory = vector_graph_inventory(&a, &captured.graph)?;
    let mut retained = Vec::new();
    let mut prefixes = Vec::new();
    let mut partitions = Vec::new();
    for len in 1..=p.rows {
        vector_seed(&a, initial)?;
        let live = NativeRawOwner::allocate(
            stream.clone(),
            Schema::checked(
                RawContract::KdaPreparedF32VectorGate,
                len,
                p.heads,
                p.dim,
                p.dim,
            )?,
            reserve,
        )?;
        {
            let lease = a.arena.lease()?;
            native_upload(&a, &lease, &live, &raw_slice(&raw, 0, len, p.heads, p.dim))?;
            a.prefill_in(&lease, &live.view()?, 0, len)?;
        }
        a.stream.drain()?;
        owner_compare(
            &owner_array(&a, "emitted", len * hd)?,
            &out[..len * hd * 4],
            "vector complete prefix output",
        )?;
        owner_compare(
            &owner_array(&a, "latest", st)?,
            &states[(len - 1) * st * 4..len * st * 4],
            "vector complete prefix final",
        )?;
        let commit = len / 16 * 16;
        let expected_commit = if commit == 0 {
            initial.as_slice()
        } else {
            &states[(commit - 1) * st * 4..commit * st * 4]
        };
        for role in ["incoming", "complete"] {
            owner_compare(
                &owner_array(&a, role, st)?,
                expected_commit,
                "vector committed carry",
            )?;
        }
        if owner_metadata(&a)? != (len as i32 % 16, 0, len as u64) {
            return Err("vector prefix cursor".into());
        }
        let pending = len % 16;
        for (i, (desc, ptr)) in a.pending_raw.inventory().into_iter().enumerate() {
            let mut actual = vec![0u8; desc.storage_bytes];
            check(
                unsafe {
                    sys::lib().cuMemcpyDtoH_v2(actual.as_mut_ptr().cast(), ptr, actual.len())
                },
                "vector full partial raw read",
            )?;
            let width = if i == 4 { p.heads * 4 } else { hd * 4 };
            let mut expected = vec![0u8; 16 * width];
            for v in expected.chunks_exact_mut(4) {
                v.copy_from_slice(&0x7fc1a123u32.to_le_bytes());
            }
            expected[..pending * width]
                .copy_from_slice(&raw[i][(len - pending) * width..len * width]);
            owner_compare(&actual, &expected, "vector original-F32 pending raw")?;
        }
        let emitted = owner_array(&a, "emitted", len * hd)?;
        let bf16 = vector_bf16_array(&a, len * hd)?;
        owner_compare(
            &bf16,
            &vector_rne_view(&emitted)?,
            "vector complete BF16 RNE view",
        )?;
        retained.push(owner_save(
            root,
            &format!("prefix{len}-output.bf16"),
            &bf16,
        )?);
        let latest = owner_array(&a, "latest", st)?;
        let committed = owner_array(&a, "complete", st)?;
        retained.push(owner_save(
            root,
            &format!("prefix{len}-output.f32"),
            &emitted,
        )?);
        retained.push(owner_save(
            root,
            &format!("prefix{len}-latest.f32"),
            &latest,
        )?);
        retained.push(owner_save(
            root,
            &format!("prefix{len}-committed.f32"),
            &committed,
        )?);
        prefixes
            .push(json!({"rows":len,"pending":pending,"complete_output_state_and_raw_exact":true}));
        if len < p.rows {
            let snapshot = capture_owner(&a, &ra)?;
            restore_owner(&b, &rb, &snapshot)?;
            for row in len..p.rows {
                {
                    let lease = a.arena.lease()?;
                    native_upload(&a, &lease, &one, &raw_slice(&raw, row, 1, p.heads, p.dim))?;
                    a.launch_graph_in(&lease, &captured.graph)?;
                }
                a.stream.drain()?;
                owner_compare(
                    &vector_bf16_array(&a, hd)?,
                    &vector_rne_view(&out[row * hd * 4..(row + 1) * hd * 4])?,
                    "vector singleton BF16 RNE view",
                )?;
                owner_compare(
                    &owner_array(&a, "emitted", hd)?,
                    &out[row * hd * 4..(row + 1) * hd * 4],
                    "vector captured singleton output",
                )?;
                owner_compare(
                    &owner_array(&a, "latest", st)?,
                    &states[row * st * 4..(row + 1) * st * 4],
                    "vector captured singleton state",
                )?;
            }
            let tail = NativeRawOwner::allocate(
                stream.clone(),
                Schema::checked(
                    RawContract::KdaPreparedF32VectorGate,
                    p.rows - len,
                    p.heads,
                    p.dim,
                    p.dim,
                )?,
                reserve,
            )?;
            {
                let lease = b.arena.lease()?;
                native_upload(
                    &b,
                    &lease,
                    &tail,
                    &raw_slice(&raw, len, p.rows - len, p.heads, p.dim),
                )?;
                b.prefill_in(&lease, &tail.view()?, len as u64, p.rows - len)?;
            }
            b.stream.drain()?;
            owner_compare(
                &vector_bf16_array(&b, (p.rows - len) * hd)?,
                &vector_rne_view(&out[len * hd * 4..])?,
                "vector restored batch BF16 RNE view",
            )?;
            owner_compare(
                &owner_array(&b, "emitted", (p.rows - len) * hd)?,
                &out[len * hd * 4..],
                "vector restored batch tail",
            )?;
            owner_compare(
                &owner_array(&b, "latest", st)?,
                &states[(p.rows - 1) * st * 4..],
                "vector restored full final",
            )?;
            native_snapshot_equal(
                &a,
                &ra,
                &capture_owner(&b, &rb)?,
                "vector shared final state equality",
            )?;
            partitions.push(
                json!({"cut":len,"graph_steps":p.rows-len,"restored_batch_and_graph_exact":true}),
            );
        }
    }
    let good = capture_owner(&a, &ra)?;
    let b_good = capture_owner(&b, &rb)?;
    let output_before = owner_array(&a, "emitted", hd)?;
    let bf16_before = vector_bf16_array(&a, hd)?;
    let mut faults = Vec::new();
    for (label, plane, bits, expected_error) in [
        ("positive-subnormal-gate", 3, 1u32, 3),
        ("negative-subnormal-beta", 4, 0x80000001, 3),
        ("nan-q", 0, 0x7fc00001, 3),
    ] {
        restore_owner(&a, &ra, &good)?;
        let mut bad = raw_slice(&raw, 0, 1, p.heads, p.dim);
        bad[plane][..4].copy_from_slice(&bits.to_le_bytes());
        {
            let lease = a.arena.lease()?;
            native_upload(&a, &lease, &one, &bad)?;
            a.launch_graph_in(&lease, &captured.graph)?;
        }
        a.stream.drain()?;
        let meta = owner_metadata(&a)?;
        if meta != (p.rows as i32 % 16, expected_error, p.rows as u64) {
            return Err(format!("vector fault metadata {label}"));
        }
        let after = capture_owner(&a, &ra)?;
        for (want, got) in good.blobs.iter().zip(&after.blobs) {
            if want.kind.ends_with("_error") {
                if got.bytes != expected_error.to_le_bytes() {
                    return Err("vector exact error code".into());
                }
            } else {
                owner_compare(&got.bytes, &want.bytes, "vector invalid input no-write")?;
            }
        }
        owner_compare(
            &owner_array(&a, "emitted", hd)?,
            &output_before,
            "vector invalid output no-write",
        )?;
        owner_compare(
            &vector_bf16_array(&a, hd)?,
            &bf16_before,
            "vector invalid BF16 output no-write",
        )?;
        native_snapshot_equal(&b, &rb, &b_good, "vector other owner isolation")?;
        faults.push(
            json!({"label":label,"error":expected_error,"persistent_and_output_no_write":true}),
        );
    }
    restore_owner(&a, &ra, &good)?;
    let after = memory()?;
    if after[0] < reserve {
        return Err("vector observed reserve".into());
    }
    fs::write(
        root.join("loaded-maps.txt"),
        fs::read("/proc/self/maps").map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    for x in &p.admission {
        owner_checked(x)?;
    }
    owner_save(root,"REPORT.json",&serde_json::to_vec_pretty(&json!({"actual_uuid":actual_uuid,"compute":p.compute,"arithmetic":a.arithmetic_sha256,"prefixes":prefixes,"partitions":partitions,"faults":faults,"retained":retained,"graph_inventory":graph_inventory,"arena_inventory":native_pointer_inventory(&a),"arena_pointer_admissions":a.arena.native_admissions,"A_pointer_admissions":a.native_admissions,"B_pointer_admissions":b.native_admissions,"arena_raw_admissions":a.arena.raw.admissions,"A_raw_admissions":a.pending_raw.admissions,"B_raw_admissions":b.pending_raw.admissions,"graph_input_admissions":one.admissions,"A_inventory":ra.inventory(p.rows),"B_inventory":rb.inventory(p.rows),"persistent_bytes_each":a.persistent_bytes,"arena_bytes":a.arena.requested_bytes,"requested_total_bytes":a.arena.requested_bytes+a.persistent_bytes+b.persistent_bytes,"reserve_bytes":reserve,"memory_before":before,"memory_after":after,"adopted":false,"ordinary_gate":"FAILED_UNCHANGED","native_windows":"OUTSTANDING","performance":"NOT_MEASURED","scope":"actual-core all-F32 per-key native owner fixture; complete prefix states, shared arena, captured continuation and full-schema restore; live producer/session serving/calibration/AMP/performance unqualified"})).map_err(|e|e.to_string())?)?;
    Ok(())
}
#[test]
fn actual_native_vector_owner() {
    vector_native_run().unwrap();
}

fn vector_graph_inventory(o: &PreparedOwner, g: &OwnerGraph<'_>) -> Result<Value, String> {
    let mut n = 0usize;
    check(
        unsafe { sys::lib().cuGraphGetNodes(g.graph, std::ptr::null_mut(), &mut n) },
        "vector graph node count",
    )?;
    let mut nodes = vec![std::ptr::null_mut(); n];
    check(
        unsafe { sys::lib().cuGraphGetNodes(g.graph, nodes.as_mut_ptr(), &mut n) },
        "vector complete graph nodes",
    )?;
    if n != 15 || nodes.len() != n {
        return Err("vector graph complete node count".into());
    }
    let mut inventory = Vec::new();
    for &node in &nodes {
        let mut p: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        check(
            unsafe { sys::lib().cuGraphKernelNodeGetParams_v2(node, &mut p) },
            "vector actual kernel node",
        )?;
        let name = o
            .arena
            .functions
            .iter()
            .find_map(|(n, f)| {
                if f.cu_function() == p.func {
                    Some(*n)
                } else {
                    None
                }
            })
            .ok_or("vector graph unknown actual function")?;
        inventory.push(json!({"node":node as usize,"name":name,"function":p.func as usize,"grid":[p.gridDimX,p.gridDimY,p.gridDimZ],"block":[p.blockDimX,p.blockDimY,p.blockDimZ],"shared_bytes":p.sharedMemBytes}));
    }
    let mut e = 0usize;
    check(
        unsafe {
            sys::lib().cuGraphGetEdges(g.graph, std::ptr::null_mut(), std::ptr::null_mut(), &mut e)
        },
        "vector graph edge count",
    )?;
    let mut from = vec![std::ptr::null_mut(); e];
    let mut to = vec![std::ptr::null_mut(); e];
    check(
        unsafe { sys::lib().cuGraphGetEdges(g.graph, from.as_mut_ptr(), to.as_mut_ptr(), &mut e) },
        "vector complete graph dependencies",
    )?;
    if e != 14 || from.len() != e || to.len() != e {
        return Err("vector graph single-stream dependencies".into());
    }
    for (&f, &t) in from.iter().zip(&to) {
        if !nodes.contains(&f) || !nodes.contains(&t) {
            return Err("vector foreign graph dependency".into());
        }
    }
    Ok(
        json!({"nodes":inventory,"edges":from.iter().zip(&to).map(|(f,t)|json!([*f as usize,*t as usize])).collect::<Vec<_>>(),"complete":true}),
    )
}
