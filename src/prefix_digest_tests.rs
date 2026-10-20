#[test]
fn prefix_digest_nist_and_partition_contract() {
    let expected = [
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22,
        0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00,
        0x15, 0xad,
    ];
    assert_eq!(prefix_row_sha256(b"abc", 3).unwrap(), expected);
    let data = b"abcdefghijkl";
    let full = prefix_row_sha256(data, 3).unwrap();
    let partition = [
        prefix_row_sha256(&data[..3], 3).unwrap(),
        prefix_row_sha256(&data[3..9], 3).unwrap(),
        prefix_row_sha256(&data[9..], 3).unwrap(),
    ]
    .concat();
    assert_eq!(full, partition);
    assert_ne!(&full[..32], &full[32..64]);
}
#[test]
fn prefix_digest_rejects_invalid_geometry_and_implicit_modes() {
    assert!(prefix_row_sha256(b"abc", 0).is_err());
    assert!(prefix_row_sha256(b"abc", 2).is_err());
    assert!(prefix_row_sha256(b"", 1).is_err());
    let raw = serde_json::json!({"directory":"/bounded","targets":[{"operation":"glm5_window_attn_norm_sha256","layer":0,"row":0,"heights":[5,9],"window_rows":9,"window_max_bytes":72}]});
    assert!(parse(&raw.to_string()).is_ok());
    let mut bad = raw.clone();
    bad["targets"][0]["operation"] = serde_json::json!("glm5_window_attn_norm");
    assert!(parse(&bad.to_string()).is_err());
    bad = raw.clone();
    bad["targets"][0]["operation"] = serde_json::json!("decode_moe_input_sha256");
    assert!(parse(&bad.to_string()).is_err());
    let state = serde_json::json!({"directory":"/bounded","targets":[{"operation":"selected_prepared_window_sha256","layer":16,"row":0,"heights":[9],"maximum_bytes":64}]});
    assert!(parse(&state.to_string()).is_ok());
    let mut bad = state.clone();
    bad["targets"][0]
        .as_object_mut()
        .unwrap()
        .remove("maximum_bytes");
    assert!(parse(&bad.to_string()).is_err());
    bad = state.clone();
    bad["targets"][0]["maximum_bytes"] = serde_json::json!(0);
    assert!(parse(&bad.to_string()).is_err());
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PrefixDigestIdentity {
    path: std::path::PathBuf,
    bytes: usize,
    sha256: String,
}
impl PrefixDigestIdentity {
    fn read(&self) -> Vec<u8> {
        use sha2::{Digest, Sha256};
        let z = std::fs::read(&self.path).unwrap();
        assert_eq!(z.len(), self.bytes);
        assert_eq!(format!("{:x}", Sha256::digest(&z)), self.sha256);
        z
    }
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PrefixDigestCase {
    name: String,
    input: PrefixDigestIdentity,
    reference: PrefixDigestIdentity,
    rows: usize,
    row_bytes: usize,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PrefixDigestPlan {
    executable: PrefixDigestIdentity,
    source_admission: PrefixDigestIdentity,
    gpu_uuid: String,
    safety_margin_bytes: usize,
    output: std::path::PathBuf,
    cases: Vec<PrefixDigestCase>,
}
struct PrefixDigestGpuOwner {
    dev: std::sync::Arc<cudarc::driver::CudaDevice>,
    image: Option<cudarc::driver::CudaSlice<u8>>,
}
impl Drop for PrefixDigestGpuOwner {
    fn drop(&mut self) {
        if let Err(e) = self.dev.synchronize() {
            eprintln!("FATAL: digest test completion unknown; retaining owners: {e}");
            std::process::abort();
        }
    }
}
#[test]
#[ignore = "explicit supported ./dev prefix-digest-gpu with frozen actual-operand plan required"]
fn prefix_digest_gpu_actual_rows() {
    use cudarc::driver::{sys, CudaDevice, DevicePtr};
    use sha2::{Digest, Sha256};
    assert_eq!(
        std::env::var("KRASIS_DEV_SCRIPT").as_deref(),
        Ok("1"),
        "use ./dev prefix-digest-gpu"
    );
    let raw = std::fs::read(
        std::env::var("KRASIS_PREFIX_DIGEST_TEST_PLAN").expect("frozen digest plan required"),
    )
    .unwrap();
    let plan: PrefixDigestPlan = serde_json::from_slice(&raw).unwrap();
    assert!(plan.output.is_absolute() && !plan.output.exists());
    assert!(!plan.cases.is_empty());
    assert!(plan.safety_margin_bytes > 0);
    assert_eq!(
        std::fs::canonicalize(std::env::current_exe().unwrap()).unwrap(),
        std::fs::canonicalize(&plan.executable.path).unwrap()
    );
    plan.executable.read();
    plan.source_admission.read();
    std::fs::create_dir(&plan.output).unwrap();
    let dev = CudaDevice::new_with_stream(0).unwrap();
    let mut uuid = unsafe { std::mem::zeroed::<sys::CUuuid>() };
    unsafe { sys::lib().cuDeviceGetUuid_v2(&mut uuid, *dev.cu_device()) }
        .result()
        .unwrap();
    let h = uuid
        .bytes
        .iter()
        .map(|b| format!("{:02x}", *b as u8))
        .collect::<String>();
    let actual = format!(
        "GPU-{}-{}-{}-{}-{}",
        &h[..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..]
    );
    assert_eq!(actual, plan.gpu_uuid);
    let mut owner = PrefixDigestGpuOwner {
        dev: dev.clone(),
        image: None,
    };
    let mut reports = Vec::new();
    const GUARD: usize = 32; // Test canary, not a memory budget.
    for case in &plan.cases {
        assert!(case
            .name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-'));
        assert!(case.rows > 0 && case.row_bytes > 0);
        let input = case.input.read();
        assert_eq!(input.len(), case.rows.checked_mul(case.row_bytes).unwrap());
        let reference = case.reference.read();
        assert_eq!(reference.len(), case.rows * 32);
        let mut image = vec![0xa5u8; GUARD];
        image.extend_from_slice(&input);
        image.extend_from_slice(&[0x5a; GUARD]);
        let (free_before, total) = cudarc::driver::result::mem_get_info().unwrap();
        assert!(free_before >= image.len().checked_add(plan.safety_margin_bytes).unwrap());
        owner.image = Some(dev.htod_copy(image.clone()).unwrap());
        let ptr = owner
            .image
            .as_ref()
            .unwrap()
            .device_ptr()
            .checked_add(GUARD as u64)
            .unwrap();
        let target = Target {
            row: 0,
            directory: plan.output.join(&case.name),
        };
        let saved = target
            .device_row_sha256(
                "full.sha256",
                ptr,
                input.len(),
                case.row_bytes,
                input.len(),
                *dev.cu_stream(),
            )
            .unwrap();
        assert_eq!(saved, reference.len());
        assert_eq!(
            std::fs::read(target.directory.join("full.sha256")).unwrap(),
            reference
        );
        let mut single = Vec::new();
        for row in 0..case.rows {
            let file = format!("row-{row}.sha256");
            target
                .device_row_sha256(
                    &file,
                    ptr + (row * case.row_bytes) as u64,
                    case.row_bytes,
                    case.row_bytes,
                    input.len(),
                    *dev.cu_stream(),
                )
                .unwrap();
            single.extend_from_slice(&std::fs::read(target.directory.join(file)).unwrap());
        }
        assert_eq!(single, reference);
        let split = case.rows / 2;
        if split > 0 {
            let mut joined = Vec::new();
            for (start, rows, file) in [
                (0, split, "part-a.sha256"),
                (split, case.rows - split, "part-b.sha256"),
            ] {
                target
                    .device_row_sha256(
                        file,
                        ptr + (start * case.row_bytes) as u64,
                        rows * case.row_bytes,
                        case.row_bytes,
                        input.len(),
                        *dev.cu_stream(),
                    )
                    .unwrap();
                joined.extend_from_slice(&std::fs::read(target.directory.join(file)).unwrap());
            }
            assert_eq!(joined, reference);
        }
        assert!(target
            .device_row_sha256(
                "bad-rows",
                ptr,
                input.len(),
                0,
                input.len(),
                *dev.cu_stream()
            )
            .is_err());
        assert!(target
            .read_digest_source(
                "bad-cap",
                ptr,
                input.len(),
                input.len() - 1,
                *dev.cu_stream()
            )
            .is_err());
        assert!(target
            .read_digest_source("overflow", u64::MAX, 2, 2, *dev.cu_stream())
            .is_err());
        assert!(!target.directory.join("bad-rows").exists());
        assert!(!target.directory.join("bad-cap").exists());
        assert_eq!(
            dev.dtoh_sync_copy(owner.image.as_ref().unwrap()).unwrap(),
            image
        );
        let (free_after, _) = cudarc::driver::result::mem_get_info().unwrap();
        assert!(free_after >= plan.safety_margin_bytes);
        reports.push(serde_json::json!({"free_before_allocation":free_before,"free_after_completed":free_after,"total_bytes":total,"safety_margin_bytes":plan.safety_margin_bytes,"memory_reading_scope":"completed boundaries, not transient low-water","name":case.name,"rows":case.rows,"row_bytes":case.row_bytes,"source_bytes_read_per_full_copy":input.len(),"retained_bytes_per_full_copy":reference.len(),"reference_sha256":format!("{:x}",Sha256::digest(&reference)),"full_single_partition_exact":true,"readonly_and_guards_exact":true,"negative_guards_rejected_before_copy":true}));
    }
    dev.synchronize().unwrap();
    let result = serde_json::json!({"passed":true,"gpu_uuid":actual,"cases":reports,"plan_sha256":format!("{:x}",Sha256::digest(&raw)),"scope":"actual captured BF16/F32 row and complete-state readback, independent Node crypto reference; no numerical kernel/timing/model acceptance"});
    let target = Target {
        row: 0,
        directory: plan.output,
    };
    target
        .write("REPORT.json", &serde_json::to_vec_pretty(&result).unwrap())
        .unwrap();
    println!("{result}");
}
