// Explicit reference-only GGML operator execution; never linked into serving.
#[cfg(unix)]
mod probe {
    use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
    use serde::Deserialize;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::{ffi::c_void, fs, path::Path, time::Instant};
    type Ptr = *mut c_void;
    #[repr(C)]
    struct Init {
        mem_size: usize,
        mem_buffer: Ptr,
        no_alloc: bool,
    }
    #[derive(Deserialize)]
    struct Artifact {
        path: String,
        sha256: String,
    }
    #[derive(Deserialize)]
    struct Case {
        name: String,
        heads: usize,
        dim: usize,
        tokens: usize,
        prefix_tokens: usize,
        inputs: Vec<Artifact>,
    }
    #[derive(Deserialize)]
    struct Plan {
        libraries: Vec<Artifact>,
        backend: String,
        cases: Vec<Case>,
    }
    fn checked(a: &Artifact) -> Result<Vec<u8>, String> {
        let b = fs::read(&a.path).map_err(|e| format!("{}: {e}", a.path))?;
        if format!("{:x}", Sha256::digest(&b)) != a.sha256 {
            return Err(format!("identity changed: {}", a.path));
        }
        Ok(b)
    }
    fn f32s(a: &Artifact, count: usize) -> Result<Vec<f32>, String> {
        let b = checked(a)?;
        if b.len() != count.checked_mul(4).ok_or("input overflow")? {
            return Err(format!("wrong size: {}", a.path));
        }
        let v: Vec<_> = b
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        if !v.iter().all(|v| v.is_finite()) {
            return Err("nonfinite input".into());
        }
        Ok(v)
    }
    struct Api {
        _libs: Vec<Library>,
        init: unsafe extern "C" fn(Init) -> Ptr,
        free: unsafe extern "C" fn(Ptr),
        tensor_overhead: unsafe extern "C" fn() -> usize,
        graph_overhead: unsafe extern "C" fn() -> usize,
        new_tensor: unsafe extern "C" fn(Ptr, i32, i64, i64, i64, i64) -> Ptr,
        gdn: unsafe extern "C" fn(Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, i64) -> Ptr,
        new_graph: unsafe extern "C" fn(Ptr) -> Ptr,
        expand: unsafe extern "C" fn(Ptr, Ptr),
        cpu_init: unsafe extern "C" fn() -> Ptr,
        cuda_init: unsafe extern "C" fn(i32) -> Ptr,
        backend_free: unsafe extern "C" fn(Ptr),
        supports: unsafe extern "C" fn(Ptr, Ptr) -> bool,
        alloc: unsafe extern "C" fn(Ptr, Ptr) -> Ptr,
        buffer_free: unsafe extern "C" fn(Ptr),
        buffer_size: unsafe extern "C" fn(Ptr) -> usize,
        nbytes: unsafe extern "C" fn(Ptr) -> usize,
        set: unsafe extern "C" fn(Ptr, *const c_void, usize, usize),
        get: unsafe extern "C" fn(Ptr, Ptr, usize, usize),
        compute: unsafe extern "C" fn(Ptr, Ptr) -> i32,
        synchronize: unsafe extern "C" fn(Ptr),
    }
    impl Api {
        unsafe fn load(p: &Plan) -> Result<Self, String> {
            let mut libs = Vec::new();
            for a in &p.libraries {
                checked(a)?;
                libs.push(
                    Library::open(Some(Path::new(&a.path)), RTLD_NOW | RTLD_GLOBAL)
                        .map_err(|e| e.to_string())?,
                );
            }
            unsafe fn symbol<T: Copy>(libs: &[Library], name: &[u8]) -> Result<T, String> {
                for l in libs {
                    if let Ok(v) = l.get::<T>(name) {
                        return Ok(*v);
                    }
                }
                Err(format!(
                    "required reference symbol missing: {}",
                    String::from_utf8_lossy(name)
                ))
            }
            macro_rules! sym {
                ($n:literal) => {
                    symbol(&libs, concat!($n, "\0").as_bytes())?
                };
            }
            Ok(Self {
                init: sym!("ggml_init"),
                free: sym!("ggml_free"),
                tensor_overhead: sym!("ggml_tensor_overhead"),
                graph_overhead: sym!("ggml_graph_overhead"),
                new_tensor: sym!("ggml_new_tensor_4d"),
                gdn: sym!("ggml_gated_delta_net"),
                new_graph: sym!("ggml_new_graph"),
                expand: sym!("ggml_build_forward_expand"),
                cpu_init: sym!("ggml_backend_cpu_init"),
                cuda_init: sym!("ggml_backend_cuda_init"),
                backend_free: sym!("ggml_backend_free"),
                supports: sym!("ggml_backend_supports_op"),
                alloc: sym!("ggml_backend_alloc_ctx_tensors"),
                buffer_free: sym!("ggml_backend_buffer_free"),
                buffer_size: sym!("ggml_backend_buffer_get_size"),
                nbytes: sym!("ggml_nbytes"),
                set: sym!("ggml_backend_tensor_set"),
                get: sym!("ggml_backend_tensor_get"),
                compute: sym!("ggml_backend_graph_compute"),
                synchronize: sym!("ggml_backend_synchronize"),
                _libs: libs,
            })
        }
    }
    struct Session<'a> {
        api: &'a Api,
        ctx: Ptr,
        backend: Ptr,
        buffer: Ptr,
    }
    impl Drop for Session<'_> {
        fn drop(&mut self) {
            unsafe {
                if !self.buffer.is_null() {
                    (self.api.buffer_free)(self.buffer)
                }
                if !self.backend.is_null() {
                    (self.api.backend_free)(self.backend)
                }
                if !self.ctx.is_null() {
                    (self.api.free)(self.ctx)
                }
            }
        }
    }
    fn save(out: &Path, name: &str, v: &[f32]) -> Result<(), String> {
        use std::io::Write;
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join(name))
            .map_err(|e| e.to_string())?;
        for x in v {
            f.write_all(&x.to_le_bytes()).map_err(|e| e.to_string())?
        }
        Ok(())
    }
    fn metric(actual: &[f32], reference: &[f64]) -> Result<Value, String> {
        if actual.len() != reference.len() {
            return Err("reference length mismatch".into());
        }
        let (mut e, mut n, mut max) = (0f64, 0f64, 0f64);
        for (&a, &b) in actual.iter().zip(reference) {
            if !a.is_finite() || !b.is_finite() {
                return Err("nonfinite reference/result".into());
            }
            e += (a as f64 - b).powi(2);
            n += b * b;
            max = max.max((a as f64 - b).abs());
        }
        Ok(
            json!({"elements":actual.len(),"relative_l2":if n>0.{Some((e/n).sqrt())}else{None},"max_abs":max,"reference_zero":n==0.}),
        )
    }
    // F64 mathematical recurrence using exactly the explicit F32 operator inputs.
    fn recurrence(v: &[Vec<f32>], h: usize, k: usize, t: usize) -> (Vec<f64>, Vec<f64>) {
        let mut state: Vec<_> = v[5].iter().map(|&x| x as f64).collect();
        let mut out = vec![0.; t * h * k];
        let scale = (1f32 / (k as f32).sqrt()) as f64;
        for step in 0..t {
            for head in 0..h {
                let row = (step * h + head) * k;
                let base = head * k * k;
                for col in 0..k {
                    let mut predicted = 0.;
                    for key in 0..k {
                        predicted += (v[3][row + key] as f64).exp()
                            * state[base + col * k + key]
                            * v[1][row + key] as f64;
                    }
                    let delta = (v[2][row + col] as f64 - predicted) * v[4][step * h + head] as f64;
                    let mut attention = 0.;
                    for key in 0..k {
                        let idx = base + col * k + key;
                        state[idx] = (v[3][row + key] as f64).exp() * state[idx]
                            + v[1][row + key] as f64 * delta;
                        attention += state[idx] * v[0][row + key] as f64;
                    }
                    out[row + col] = attention * scale;
                }
            }
        }
        (out, state)
    }
    unsafe fn run_case(api: &Api, p: &Plan, c: &Case, out: &Path) -> Result<Value, String> {
        if c.heads == 0
            || c.dim == 0
            || c.tokens == 0
            || c.tokens > i32::MAX as usize
            || c.prefix_tokens == 0
            || c.prefix_tokens > c.tokens
            || c.inputs.len() != 6
        {
            return Err("invalid operator dimensions/inputs".into());
        }
        let mul = |a: usize, b: usize| a.checked_mul(b).ok_or("shape overflow".to_string());
        let hk = mul(c.heads, c.dim)?;
        let rows = mul(c.tokens, hk)?;
        let state_size = mul(hk, c.dim)?;
        let counts = [rows, rows, rows, rows, mul(c.tokens, c.heads)?, state_size];
        let mut inputs = Vec::new();
        for (a, count) in c.inputs.iter().zip(counts) {
            inputs.push(f32s(a, count)?)
        }
        let meta = (api.tensor_overhead)()
            .checked_mul(c.inputs.len() + 1)
            .and_then(|n| n.checked_add((api.graph_overhead)()))
            .ok_or("context size overflow")?;
        let ctx = (api.init)(Init {
            mem_size: meta,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: true,
        });
        if ctx.is_null() {
            return Err("reference context allocation failed".into());
        }
        let mut owner = Session {
            api,
            ctx,
            backend: std::ptr::null_mut(),
            buffer: std::ptr::null_mut(),
        };
        owner.backend=match p.backend.as_str(){"cpu"=>(api.cpu_init)(),"cuda:0"=>(api.cuda_init)(0),_=>return Err("explicit cpu or cuda:0 backend required; select CUDA_VISIBLE_DEVICES outside the process".into())};
        if owner.backend.is_null() {
            return Err("requested backend unavailable; no fallback".into());
        }
        let mut tensors = Vec::new();
        for i in 0..6 {
            let shape = if i == 5 {
                [c.dim, c.dim, c.heads, 1]
            } else {
                [if i == 4 { 1 } else { c.dim }, c.heads, c.tokens, 1]
            };
            let dims: Vec<i64> = shape
                .into_iter()
                .map(i64::try_from)
                .collect::<Result<_, _>>()
                .map_err(|_| "dimension outside GGML ABI")?;
            let tensor = (api.new_tensor)(ctx, 0, dims[0], dims[1], dims[2], dims[3]);
            if tensor.is_null() {
                return Err("tensor header allocation failed".into());
            }
            tensors.push(tensor);
        }
        let snapshots = i64::try_from(c.tokens).map_err(|_| "snapshot count overflow")?;
        let result = (api.gdn)(
            ctx, tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], snapshots,
        );
        if result.is_null() || !(api.supports)(owner.backend, result) {
            return Err("requested backend does not support actual GDN operation".into());
        }
        let expected = rows
            .checked_add(mul(c.tokens, state_size)?)
            .and_then(|n| n.checked_mul(4))
            .ok_or("result overflow")?;
        if (api.nbytes)(result) != expected {
            return Err("Witness result layout differs from explicit ABI contract".into());
        }
        let graph = (api.new_graph)(ctx);
        if graph.is_null() {
            return Err("graph header allocation failed".into());
        }
        (api.expand)(graph, result);
        owner.buffer = (api.alloc)(ctx, owner.backend);
        if owner.buffer.is_null() {
            return Err("requested backend tensor allocation failed".into());
        }
        for (tensor, data) in tensors.iter().zip(&inputs) {
            (api.set)(*tensor, data.as_ptr().cast(), 0, data.len() * 4);
        }
        (api.synchronize)(owner.backend);
        let now = Instant::now();
        let status = (api.compute)(owner.backend, graph);
        (api.synchronize)(owner.backend);
        let elapsed = now.elapsed().as_secs_f64();
        if status != 0 {
            return Err(format!("actual GDN operation failed: {status}"));
        }
        let mut output = vec![0f32; rows];
        let mut final_state = vec![0f32; state_size];
        (api.get)(result, output.as_mut_ptr().cast(), 0, rows * 4);
        (api.get)(
            result,
            final_state.as_mut_ptr().cast(),
            rows * 4,
            state_size * 4,
        );
        let prefix = c.prefix_tokens;
        let mut prefix_state = vec![0f32; state_size];
        let prefix_offset = (rows + (c.tokens - prefix) * state_size) * 4;
        (api.get)(
            result,
            prefix_state.as_mut_ptr().cast(),
            prefix_offset,
            state_size * 4,
        );
        let (reference, ref_state) = recurrence(&inputs, c.heads, c.dim, c.tokens);
        let (_, ref_prefix_state) = recurrence(&inputs, c.heads, c.dim, prefix);
        save(out, &format!("{}-output.f32", c.name), &output)?;
        save(
            out,
            &format!("{}-final-state-transposed.f32", c.name),
            &final_state,
        )?;
        save(
            out,
            &format!("{}-prefix-state-transposed.f32", c.name),
            &prefix_state,
        )?;
        Ok(
            json!({"name":c.name,"heads":c.heads,"dim":c.dim,"tokens":c.tokens,"actual_backend":p.backend,"snapshot_slots":c.tokens,"context_bytes":meta,"actual_buffer_bytes":(api.buffer_size)(owner.buffer),"synchronized_operation_seconds":elapsed,"output_vs_f64":metric(&output,&reference)?,"prefix_output_vs_f64":metric(&output[..prefix*hk],&reference[..prefix*hk])?,"prefix_state_vs_f64":metric(&prefix_state,&ref_prefix_state)?,"final_state_vs_f64":metric(&final_state,&ref_state)?,"qualifies_candidate":false}),
        )
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn independent_recurrence_matches_explicit_two_coordinate_example() {
            let inputs = vec![
                vec![1., 0.],
                vec![0.5, 0.5],
                vec![2., 4.],
                vec![0., 0.],
                vec![0.5],
                vec![0., 0., 0., 0.],
            ];
            let (output, state) = recurrence(&inputs, 1, 2, 1);
            assert_eq!(state, vec![0.5, 0.5, 1., 1.]);
            let scale = (1f32 / 2f32.sqrt()) as f64;
            assert_eq!(output, vec![0.5 * scale, scale]);
        }
        #[test]
        fn independent_recurrence_prefix_ignores_future_inputs() {
            let inputs = vec![
                vec![1., 0., 999., -999.],
                vec![0.5, 0.5, -0.2, 0.3],
                vec![2., 4., 7., -5.],
                vec![0., 0., -1., -2.],
                vec![0.5, 0.8],
                vec![0., 0., 0., 0.],
            ];
            let (short, _) = recurrence(&inputs, 1, 2, 1);
            let (long, _) = recurrence(&inputs, 1, 2, 2);
            assert_eq!(short, &long[..2]);
        }
        #[test]
        fn metric_rejects_nonfinite_reference_instead_of_fabricating_statistics() {
            assert!(metric(&[f32::NAN], &[0.]).is_err());
            assert!(metric(&[0.], &[f64::INFINITY]).is_err());
            assert!(metric(&[0.], &[]).is_err());
        }
    }
    pub fn run() -> Result<(), String> {
        if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
            return Err("Run through ./dev witness-kda-operator".into());
        }
        let a: Vec<_> = std::env::args().skip(1).collect();
        if a.len() != 2 {
            return Err("usage: ./dev witness-kda-operator <plan.json> <fresh-output>".into());
        }
        let p: Plan = serde_json::from_slice(&fs::read(&a[0]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        if p.libraries.is_empty() || p.cases.is_empty() {
            return Err("empty reference plan".into());
        }
        let mut names = std::collections::HashSet::new();
        for c in &p.cases {
            if c.name.is_empty()
                || !c
                    .name
                    .bytes()
                    .all(|x| x.is_ascii_alphanumeric() || x == b'-')
                || !names.insert(c.name.clone())
            {
                return Err("invalid/duplicate case name".into());
            }
        }
        let out = Path::new(&a[1]);
        fs::create_dir(out).map_err(|e| e.to_string())?;
        let api = unsafe { Api::load(&p)? };
        let mut results = Vec::new();
        for c in &p.cases {
            results.push(unsafe { run_case(&api, &p, c, out)? })
        }
        fs::write(out.join("REPORT.json"),serde_json::to_vec_pretty(&json!({"diagnostic_only":true,"qualifies_candidate":false,"actual_witness_operator":true,"contract":"Identical explicit F32 prepared inputs, natural-log gates and transposed F32 state. F64 comparison does not emulate expf/FMA/warp order.","results":results})).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        Ok(())
    }
}
fn main() {
    #[cfg(unix)]
    if let Err(e) = probe::run() {
        eprintln!("{e}");
        std::process::exit(1)
    }
    #[cfg(not(unix))]
    {
        eprintln!("This explicit reference loader requires Unix shared libraries");
        std::process::exit(1)
    }
}
