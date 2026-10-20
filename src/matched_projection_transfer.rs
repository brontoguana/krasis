//! Private, admitted conditional projection transfer. No ordinary path substitution.
use crate::matched_probe::Target;
use cudarc::driver::sys as cuda;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashSet},
    fs,
    path::PathBuf,
    sync::{Arc, OnceLock},
};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    path: PathBuf,
    sha256: String,
    bytes: usize,
}
impl File {
    fn read(&self) -> Result<Vec<u8>, String> {
        if !self.path.is_absolute()
            || self.sha256.len() != 64
            || !self.sha256.bytes().all(|v| v.is_ascii_hexdigit())
        {
            return Err("invalid projection file identity".into());
        }
        let z = fs::read(&self.path).map_err(|e| format!("{}: {e}", self.path.display()))?;
        if z.len() != self.bytes || digest(&z) != self.sha256 {
            return Err(format!(
                "projection identity changed {}",
                self.path.display()
            ));
        }
        Ok(z)
    }
}
fn digest(z: &[u8]) -> String {
    format!("{:x}", Sha256::digest(z))
}
#[derive(Clone, Copy, Deserialize, Serialize, PartialEq, Debug)]
#[serde(rename_all = "snake_case")]
enum Mode {
    NoWrite,
    QOnly,
    BetaOnly,
    Both,
}
impl Mode {
    fn writes(self, projection: &str) -> bool {
        matches!(
            (self, projection),
            (Self::QOnly, "q_proj")
                | (Self::BetaOnly, "b_proj")
                | (Self::Both, "q_proj" | "b_proj")
        )
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    projection: String,
    live_files: BTreeMap<String, File>,
    descriptor_fields: Value,
    replacement: File,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: u32,
    mode: Mode,
    teacher_label: String,
    layer: usize,
    height: usize,
    consumed: u32,
    directory: PathBuf,
    native: File,
    centered_module: File,
    baseline: File,
    no_write_control: Option<File>,
    identities: Vec<File>,
    cases: Vec<Case>,
}
fn validate(p: &Plan) -> Result<(), String> {
    if p.schema != 1
        || !p.directory.is_absolute()
        || p.teacher_label.is_empty()
        || p.height == 0
        || p.identities.is_empty()
    {
        return Err("invalid conditional projection scope".into());
    }
    let names: HashSet<_> = p.cases.iter().map(|c| c.projection.as_str()).collect();
    if p.cases.len() != 2 || names != HashSet::from(["q_proj", "b_proj"]) {
        return Err("complete distinct Q and beta controls required".into());
    }
    for c in &p.cases {
        let common = [
            "normalized-input.bf16",
            "conv_incoming.f32",
            "recurrent_incoming.f32",
        ];
        let own = [
            format!("{}-input.bf16", c.projection),
            format!("{}-packed.bin", c.projection),
            format!("{}-scales.f32", c.projection),
            format!("{}-zeros.f32", c.projection),
            format!("{}-output.bf16", c.projection),
        ];
        if c.live_files.len() != common.len() + own.len()
            || common.iter().any(|n| !c.live_files.contains_key(*n))
            || own.iter().any(|n| !c.live_files.contains_key(n))
        {
            return Err("complete actual operand/state/output admission required".into());
        }
        let fields = c
            .descriptor_fields
            .as_object()
            .ok_or("projection descriptor must be an object")?;
        let required = [
            "projection",
            "rows",
            "cols",
            "group_size",
            "axis",
            "layout",
            "packed_dtype",
            "scales_dtype",
            "zeros_dtype",
            "packed_row_stride_bytes",
            "scales_row_stride_bytes",
            "zeros_row_stride_bytes",
            "hqq6_selected_vec4_warps",
            "nbits",
            "input_dtype",
            "output_dtype",
        ];
        if fields.len() != required.len()
            || required.iter().any(|k| !fields.contains_key(*k))
            || fields["projection"] != c.projection
            || fields["input_dtype"] != "bfloat16"
            || fields["output_dtype"] != "bfloat16"
            || fields["scales_dtype"] != "float32"
            || fields["zeros_dtype"] != "float32"
        {
            return Err("complete exact projection descriptor admission required".into());
        }
        let rows = fields["rows"].as_u64().ok_or("projection rows required")?;
        if rows == 0
            || rows.checked_mul(2) != Some(c.replacement.bytes as u64)
            || c.live_files[&format!("{}-output.bf16", c.projection)].bytes != c.replacement.bytes
        {
            return Err("replacement output geometry mismatch".into());
        }
    }
    if (p.mode == Mode::NoWrite) != p.no_write_control.is_none() {
        return Err("only no-write admission may omit its completed control receipt".into());
    }
    Ok(())
}
fn parse(raw: &[u8]) -> Result<Plan, String> {
    let p: Plan = serde_json::from_slice(raw).map_err(|e| e.to_string())?;
    validate(&p)?;
    Ok(p)
}
fn receipt(f: &File, forced: bool) -> Result<(), String> {
    let r: Value = serde_json::from_slice(&f.read()?).map_err(|e| e.to_string())?;
    validate_receipt(&r, forced)
}
fn validate_receipt(r: &Value, forced: bool) -> Result<(), String> {
    if r["passed"] != true
        || r["forcing"] != forced
        || r["complete_C6_C8_vectors_hidden_112_states_exact"] != true
    {
        return Err("complete exact baseline/control receipt required".into());
    }
    Ok(())
}
fn config() -> Result<Option<Arc<Plan>>, String> {
    static PLAN: OnceLock<Result<Option<Arc<Plan>>, String>> = OnceLock::new();
    PLAN.get_or_init(|| {
        let path = match std::env::var("KRASIS_MATCHED_PROJECTION_TRANSFER_CONFIG") {
            Ok(p) => p,
            Err(std::env::VarError::NotPresent) => return Ok(None),
            Err(e) => return Err(e.to_string()),
        };
        let p = parse(&fs::read(path).map_err(|e| e.to_string())?)?;
        p.native.read()?;
        let maps = fs::read_to_string("/proc/self/maps").map_err(|e| e.to_string())?;
        if !maps
            .lines()
            .any(|l| l.split_whitespace().last() == p.native.path.to_str())
        {
            return Err("conditional native is not mapped".into());
        }
        let center = std::env::var_os("KRASIS_MATCHED_KDA_INTRA_CONFIG")
            .ok_or("explicit original center module required")?;
        if PathBuf::from(center) != p.centered_module.path {
            return Err("conditional center identity mismatch".into());
        }
        p.centered_module.read()?;
        receipt(&p.baseline, false)?;
        if let Some(f) = &p.no_write_control {
            receipt(f, true)?;
        }
        for f in &p.identities {
            f.read()?;
        }
        for c in &p.cases {
            for f in c.live_files.values() {
                f.read()?;
            }
            finite_bf16(&c.replacement.read()?)?;
        }
        Ok(Some(Arc::new(p)))
    })
    .as_ref()
    .map(Clone::clone)
    .map_err(Clone::clone)
}
fn finite_bf16(z: &[u8]) -> Result<(), String> {
    if z.is_empty()
        || z.len() % 2 != 0
        || z.chunks_exact(2)
            .any(|v| !f32::from_bits((u16::from_le_bytes([v[0], v[1]]) as u32) << 16).is_finite())
    {
        Err("replacement BF16 is empty/odd/nonfinite".into())
    } else {
        Ok(())
    }
}
struct Active {
    label: String,
    plan: Option<Arc<Plan>>,
    entered: bool,
    frame: bool,
    applied: HashSet<String>,
}
thread_local! { static ACTIVE:RefCell<Option<Active>>=const{RefCell::new(None)}; }
pub(crate) struct Scope {
    finished: bool,
}
impl Scope {
    pub(crate) fn enter(label: Option<&str>) -> Result<Option<Self>, String> {
        // Calibration and ordinary requests never configure or enter a transfer.
        let Some(label) = label else { return Ok(None) };
        let plan = config()?;
        ACTIVE.with(|cell| {
            let mut v = cell.borrow_mut();
            if v.is_some() {
                return Err("nested projection request lease".into());
            }
            *v = Some(Active {
                label: label.into(),
                plan,
                entered: false,
                frame: false,
                applied: HashSet::new(),
            });
            Ok(Some(Self { finished: false }))
        })
    }
    pub(crate) fn finish(&mut self) -> Result<(), String> {
        if self.finished {
            return Err("projection request receipt already completed".into());
        }
        ACTIVE.with(|cell|{let mut v=cell.borrow_mut();let a=v.as_mut().ok_or("missing projection request lease")?;if let Some(p)=&a.plan{
            let mut applied:Vec<_>=a.applied.iter().cloned().collect();applied.sort();
            let complete=a.label!=p.teacher_label||(a.entered&&applied==vec!["b_proj".to_string(),"q_proj".to_string()]);
            let output=Target{row:0,directory:p.directory.join(&a.label)};output.write("REQUEST_COMPLETE.json",&serde_json::to_vec_pretty(&json!({"mode":p.mode,"teacher_label":a.label,"entered":a.entered,"applied":applied,"complete":complete,"diagnostic_only":true,"qualifies_candidate":false,"ordinary_cache_equality_evidence":false})).map_err(|e|e.to_string())?)?;
            if !complete{return Err("missing/incomplete conditional projection applications".into())}
        }self.finished=true;Ok(())})
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|c| {
            c.borrow_mut().take();
        });
    }
}
pub(crate) fn forcing_label() -> Option<String> {
    ACTIVE.with(|c| c.borrow().as_ref().map(|v| v.label.clone()))
}
pub(crate) fn boundary_intervention() -> Option<Value> {
    ACTIVE.with(|cell|cell.borrow().as_ref().and_then(|a|a.plan.as_ref().map(|p|json!({"mode":p.mode,"target_teacher":p.teacher_label,"target_frame":a.frame,"applied":a.applied,"projection_operand_changed":a.frame&&a.applied.iter().any(|n|p.mode.writes(n)),"diagnostic_only":true,"qualifies_candidate":false}))))
}
pub(crate) struct Frame;
impl Frame {
    pub(crate) fn enter(
        layer: usize,
        height: usize,
        consumed: usize,
        observed: bool,
    ) -> Result<Self, String> {
        ACTIVE.with(|cell| {
            let mut v = cell.borrow_mut();
            if let Some(a) = v.as_mut() {
                a.frame = false;
                if let Some(p) = &a.plan {
                    if a.label == p.teacher_label && layer == p.layer && height == p.height {
                        if !observed || consumed != p.consumed as usize || a.entered {
                            return Err(
                                "conditional frame missing observation/wrong input/duplicate"
                                    .into(),
                            );
                        }
                        a.entered = true;
                        a.frame = true;
                    }
                }
            }
            Ok(Self)
        })
    }
}
impl Drop for Frame {
    fn drop(&mut self) {
        ACTIVE.with(|c| {
            if let Some(a) = c.borrow_mut().as_mut() {
                a.frame = false;
            }
        });
    }
}
pub(crate) fn apply(
    target: &Target,
    projection: &str,
    output: u64,
    stream: cuda::CUstream,
) -> Result<(), String> {
    ACTIVE.with(|cell|{
        let mut active=cell.borrow_mut();let Some(a)=active.as_mut()else{return Ok(())};if !a.frame{return Ok(())}let p=a.plan.as_ref().ok_or("conditional frame missing plan")?;
        let c=p.cases.iter().find(|c|c.projection==projection).ok_or("unadmitted conditional projection")?;
        if a.applied.contains(projection){return Err("duplicate conditional projection".into())}
        for (name,f) in &c.live_files {let bytes=fs::read(target.directory.join(name)).map_err(|e|format!("live {name}: {e}"))?;if bytes.len()!=f.bytes || digest(&bytes)!=f.sha256{return Err(format!("unmatched live conditional operand/state {name}"))}}
        let live:Value=serde_json::from_slice(&fs::read(target.directory.join(format!("{projection}-descriptor.json"))).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        for(k,v)in c.descriptor_fields.as_object().ok_or("missing conditional fields")?{if live.get(k)!=Some(v){return Err(format!("conditional descriptor mismatch {k}"))}}
        let replacement=c.replacement.read()?;finite_bf16(&replacement)?;
        let writes=p.mode.writes(projection);
        if writes {
            let mut status=cuda::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
            let rc=unsafe{cuda::lib().cuStreamIsCapturing(stream,&mut status)};
            if rc!=cuda::CUresult::CUDA_SUCCESS||status!=cuda::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE{return Err("conditional transfer during capture/unknown stream".into())}
            let rc=unsafe{cuda::lib().cuMemcpyHtoDAsync_v2(output,replacement.as_ptr().cast(),replacement.len(),stream)};
            // Always complete even a failed enqueue before releasing host backing.
            let completion=unsafe{cuda::lib().cuStreamSynchronize(stream)};
            if completion!=cuda::CUresult::CUDA_SUCCESS {
                // Unknown DMA completion cannot release host or model backing.
                // This explicitly admitted diagnostic stops the whole process
                // without Rust unwinding; CUDA context teardown ends DMA.
                let record=json!({"projection":projection,"enqueue":format!("{rc:?}"),"completion":format!("{completion:?}"),"diagnostic_only":true,"fatal_teardown":true,"backing_released_before_completion":false});
                let saved=target.write(&format!("{projection}-conditional-fatal.json"),record.to_string().as_bytes());
                eprintln!("FATAL conditional projection DMA completion {completion:?}; enqueue {rc:?}; preserve={saved:?}; stopping diagnostic without releasing backing");
                std::process::abort();
            }
            if rc!=cuda::CUresult::CUDA_SUCCESS{return Err(format!("conditional transfer enqueue failure after completed drain {rc:?}"))}
        }
        target.device(&format!("{projection}-conditional-after.bf16"),output,replacement.len(),stream)?;
        if writes && fs::read(target.directory.join(format!("{projection}-conditional-after.bf16"))).map_err(|e|e.to_string())?!=replacement{return Err("conditional copy did not restore exact output".into())}
        target.write(&format!("{projection}-conditional.json"),&serde_json::to_vec_pretty(&json!({"mode":p.mode,"teacher_label":a.label,"layer":p.layer,"height":p.height,"consumed":p.consumed,"projection":projection,"wrote":writes,"replacement_sha256":c.replacement.sha256,"diagnostic_only":true,"ordinary_cache_equality_evidence":false,"qualifies_candidate":false})).map_err(|e|e.to_string())?)?;
        a.applied.insert(projection.into());Ok(())
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    fn file() -> Value {
        json!({"path":"/tmp/admitted","sha256":"a".repeat(64),"bytes":8})
    }
    fn fixture() -> Value {
        let cases=["q_proj","b_proj"].into_iter().map(|projection|{
            let mut files=serde_json::Map::new();for n in ["normalized-input.bf16","conv_incoming.f32","recurrent_incoming.f32"]{files.insert(n.into(),file());}for n in ["input.bf16","packed.bin","scales.f32","zeros.f32","output.bf16"]{files.insert(format!("{projection}-{n}"),file());}
            json!({"projection":projection,"live_files":files,"replacement":file(),"descriptor_fields":{"projection":projection,"rows":4,"cols":8,"group_size":4,"axis":1,"layout":"test_complete_row_layout","packed_dtype":"uint8","scales_dtype":"float32","zeros_dtype":"float32","packed_row_stride_bytes":6,"scales_row_stride_bytes":8,"zeros_row_stride_bytes":8,"hqq6_selected_vec4_warps":2,"nbits":6,"input_dtype":"bfloat16","output_dtype":"bfloat16"}})
        }).collect::<Vec<_>>();
        json!({"schema":1,"mode":"no_write","teacher_label":"teacher_seed_0","layer":0,"height":53,"consumed":13,"directory":"/tmp/projection-evidence","native":file(),"centered_module":file(),"baseline":file(),"no_write_control":null,"identities":[file()],"cases":cases})
    }
    fn parsed(v: &Value) -> Result<Plan, String> {
        parse(&serde_json::to_vec(v).unwrap())
    }
    #[test]
    fn generic_empty_source_identity_is_exact_and_not_an_empty_replacement() {
        let root = std::env::temp_dir().join(format!(
            "projection-empty-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir(&root).unwrap();
        let path = root.join("empty-source");
        fs::write(&path, []).unwrap();
        let empty = File {
            path: path.clone(),
            bytes: 0,
            sha256: digest(&[]),
        };
        assert_eq!(empty.read().unwrap(), Vec::<u8>::new());
        assert!(finite_bf16(&empty.read().unwrap()).is_err());
        let mut bad = empty.clone();
        bad.sha256 = digest(&[0]);
        assert!(bad.read().is_err());
        let mut bad = empty.clone();
        bad.bytes = 1;
        assert!(bad.read().is_err());
        fs::write(&path, [0]).unwrap();
        assert!(empty.read().is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn centered_projection_admission_requires_complete_operand_state_and_control_contract() {
        let good = fixture();
        assert!(parsed(&good).is_ok());
        for mode in ["q_only", "beta_only", "both"] {
            let mut p = good.clone();
            p["mode"] = json!(mode);
            assert!(parsed(&p).is_err());
            p["no_write_control"] = file();
            assert!(parsed(&p).is_ok());
        }
        for key in [
            "normalized-input.bf16",
            "conv_incoming.f32",
            "recurrent_incoming.f32",
            "q_proj-input.bf16",
            "q_proj-packed.bin",
            "q_proj-scales.f32",
            "q_proj-zeros.f32",
            "q_proj-output.bf16",
        ] {
            let mut p = good.clone();
            p["cases"][0]["live_files"]
                .as_object_mut()
                .unwrap()
                .remove(key);
            assert!(parsed(&p).is_err(), "{key}");
        }
        let mut p = good.clone();
        p["cases"][1] = p["cases"][0].clone();
        assert!(parsed(&p).is_err());
        let mut p = good.clone();
        p["cases"][0]["replacement"]["bytes"] = json!(7);
        assert!(parsed(&p).is_err());
        let mut p = good.clone();
        p["cases"][0]["descriptor_fields"]["rows"] = json!(0);
        assert!(parsed(&p).is_err());
        let mut p = good.clone();
        p["cases"][0]["descriptor_fields"]
            .as_object_mut()
            .unwrap()
            .remove("hqq6_selected_vec4_warps");
        assert!(parsed(&p).is_err());
        let mut p = good.clone();
        p["cases"][0]["descriptor_fields"]["extra_guess"] = json!(1);
        assert!(parsed(&p).is_err());
        let mut p = good.clone();
        p["directory"] = json!("relative");
        assert!(parsed(&p).is_err());
        let mut p = good.clone();
        p["no_write_control"] = file();
        assert!(parsed(&p).is_err());
        let mut p = good;
        p["fallback"] = json!(true);
        assert!(parsed(&p).is_err());
    }
    #[test]
    fn receipts_require_complete_exact_vectors_states_and_actual_forcing_scope() {
        let good = json!({"passed":true,"forcing":false,"complete_C6_C8_vectors_hidden_112_states_exact":true});
        assert!(validate_receipt(&good, false).is_ok());
        assert!(validate_receipt(&good, true).is_err());
        for key in [
            "passed",
            "forcing",
            "complete_C6_C8_vectors_hidden_112_states_exact",
        ] {
            let mut r = good.clone();
            r.as_object_mut().unwrap().remove(key);
            assert!(validate_receipt(&r, false).is_err());
        }
        let mut bad = good.clone();
        bad["complete_C6_C8_vectors_hidden_112_states_exact"] = json!(false);
        assert!(validate_receipt(&bad, false).is_err());
        let mut forced = good;
        forced["forcing"] = json!(true);
        assert!(validate_receipt(&forced, true).is_ok());
    }
    #[test]
    fn ordinary_request_does_not_enter_or_configure_a_lease() {
        assert!(Scope::enter(None).unwrap().is_none());
        assert!(forcing_label().is_none());
    }
    #[test]
    fn finite_replacement_does_not_admit_empty_odd_nan_or_infinity() {
        for b in [
            vec![],
            vec![0],
            vec![0x80, 0x7f],
            vec![0xc1, 0x7f],
            vec![0x80, 0xff],
        ] {
            assert!(finite_bf16(&b).is_err());
        }
        assert!(finite_bf16(&[0, 0, 0x80, 0x3f, 0x80, 0xbf]).is_ok());
    }
    #[test]
    fn projection_modes_preserve_unselected_projections() {
        for p in ["q_proj", "b_proj"] {
            assert!(!Mode::NoWrite.writes(p));
            assert!(Mode::Both.writes(p));
        }
        assert!(Mode::QOnly.writes("q_proj"));
        assert!(!Mode::QOnly.writes("b_proj"));
        assert!(Mode::BetaOnly.writes("b_proj"));
        assert!(!Mode::BetaOnly.writes("q_proj"));
        assert!(!Mode::Both.writes("k_proj"));
    }
    #[test]
    fn projection_frame_rejects_wrong_history_duplicates_and_clears_on_all_exits() {
        let p = Arc::new(parsed(&fixture()).unwrap());
        ACTIVE.with(|cell| {
            *cell.borrow_mut() = Some(Active {
                label: p.teacher_label.clone(),
                plan: Some(p.clone()),
                entered: false,
                frame: false,
                applied: HashSet::new(),
            })
        });
        let lease = Scope { finished: false };
        assert!(Frame::enter(0, 53, 12, true).is_err());
        assert!(Frame::enter(0, 53, 13, false).is_err());
        {
            let _frame = Frame::enter(0, 53, 13, true).unwrap();
            assert!(ACTIVE.with(|c| c.borrow().as_ref().unwrap().frame));
        }
        assert!(!ACTIVE.with(|c| c.borrow().as_ref().unwrap().frame));
        assert!(Frame::enter(0, 53, 13, true).is_err());
        drop(lease);
        assert!(forcing_label().is_none());
        let _frame = Frame::enter(0, 53, 13, true).unwrap();
        assert!(forcing_label().is_none());
    }
    #[test]
    fn non_target_teacher_frame_cannot_apply_an_experiment() {
        let p = Arc::new(parsed(&fixture()).unwrap());
        ACTIVE.with(|c| {
            *c.borrow_mut() = Some(Active {
                label: "teacher_full_A4".into(),
                plan: Some(p),
                entered: false,
                frame: false,
                applied: HashSet::new(),
            })
        });
        let lease = Scope { finished: false };
        let _frame = Frame::enter(0, 53, 13, true).unwrap();
        assert!(!ACTIVE.with(|c| c.borrow().as_ref().unwrap().frame));
        drop(lease);
        assert!(forcing_label().is_none());
    }
}
