// Explicit diagnostic: uploads and dispatch are unchanged. Incoming state
// reads are blocking observations before replay and can affect timing; scalar
// reads follow the normal final sync. No observational invariance is assumed.
struct ReplayScalarAudit {
    directory: std::path::PathBuf,
    request: String,
    position: usize,
    token: usize,
    step: usize,
    execution_path: &'static str,
    uploads: Vec<(String, u64, i32, cuda_sys::CUresult)>,
}
impl ReplayScalarAudit {
    fn begin(
        request: &str,
        position: usize,
        token: usize,
        step: usize,
        execution_path: &'static str,
    ) -> Result<Option<Self>, String> {
        let dir = std::env::var("KRASIS_REPLAY_SCALAR_AUDIT_DIR").ok();
        let target = std::env::var("KRASIS_REPLAY_SCALAR_AUDIT_POSITION").ok();
        match (dir, target) {
            (None, None) => Ok(None),
            (Some(dir), Some(target)) => {
                let target = target
                    .parse::<usize>()
                    .map_err(|_| "scalar audit position must be unsigned")?;
                if position != target {
                    return Ok(None);
                }
                let directory = std::path::PathBuf::from(dir);
                if !directory.is_absolute() || !directory.is_dir() {
                    return Err("scalar audit needs existing absolute output directory".into());
                }
                Ok(Some(Self {
                    directory,
                    request: request.to_owned(),
                    position,
                    token,
                    step,
                    execution_path,
                    uploads: Vec::new(),
                }))
            }
            _ => Err("scalar audit requires both output directory and position".into()),
        }
    }
    fn capture_incoming(
        &self,
        selected: &crate::kda_prefix_owner::KdaEngineRequestStorage,
        layer: usize,
    ) -> Result<(), String> {
        use sha2::Digest;
        use std::io::Write;
        let key = format!("{:x}", sha2::Sha256::digest(self.request.as_bytes()));
        let directory = self.directory.join(format!(
            "{key}-position{}-token{}-step{}-incoming",
            self.position, self.token, self.step
        ));
        std::fs::create_dir(&directory).map_err(|e| e.to_string())?;
        let mut rows = Vec::new();
        for allocation in selected.diagnostic_serving_layer(layer)? {
            let mut bytes = vec![0u8; allocation.storage_bytes];
            let result = unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    bytes.as_mut_ptr().cast(),
                    allocation.ptr,
                    bytes.len(),
                )
            };
            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("scalar observer actual incoming read: {result:?}"));
            }
            let path = directory.join(format!("{}.bin", allocation.name));
            let mut out = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|e| e.to_string())?;
            out.write_all(&bytes)
                .and_then(|_| out.sync_all())
                .map_err(|e| e.to_string())?;
            rows.push(serde_json::json!({"allocation":allocation,"path":path,"bytes":bytes.len(),"sha256":format!("{:x}",sha2::Sha256::digest(&bytes))}));
        }
        std::fs::write(directory.join("REPORT.json"),serde_json::to_vec_pretty(&serde_json::json!({"request":self.request,"position":self.position,"token":self.token,"layer":layer,"before_graph_replay":self.execution_path=="graph_replay","actual_execution_path":self.execution_path,"replaced_inputs":false,"rows":rows})).map_err(|e|e.to_string())?).map_err(|e|e.to_string())
    }
    fn record(&mut self, name: &str, ptr: u64, expected: i32, rc: cuda_sys::CUresult) {
        self.uploads.push((name.into(), ptr, expected, rc));
    }
    fn complete(self) -> Result<(), String> {
        use sha2::Digest;
        use std::io::Write;
        let mut rows = Vec::new();
        for (name, ptr, expected, upload) in self.uploads {
            let mut actual = 0i32;
            let read = unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    (&mut actual as *mut i32).cast(),
                    ptr,
                    std::mem::size_of::<i32>(),
                )
            };
            rows.push(serde_json::json!({"name":name,"ptr":ptr,"expected":expected,"actual":if read==cuda_sys::CUresult::CUDA_SUCCESS{Some(actual)}else{None},"upload_result":format!("{upload:?}"),"read_result":format!("{read:?}")}));
        }
        let key = format!("{:x}", sha2::Sha256::digest(self.request.as_bytes()));
        let path = self.directory.join(format!(
            "{key}-position{}-token{}-step{}.json",
            self.position, self.token, self.step
        ));
        let data = serde_json::json!({"request":self.request,"position":self.position,"token":self.token,"step":self.step,"scalar_uploads":rows,"source":"Actual incoming state at selected execution path; scalar uploads only recorded for graph replay","actual_execution_path":self.execution_path,"replaced_inputs":false,"forced_dispatch":false,"explicit_gpu_barrier_added":false,"blocking_incoming_d2h_before_execution":true,"blocking_incoming_d2h_before_replay":self.execution_path=="graph_replay","observational_invariance_claimed":false});
        let mut out = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        out.write_all(&serde_json::to_vec_pretty(&data).map_err(|e| e.to_string())?)
            .and_then(|_| out.sync_all())
            .map_err(|e| e.to_string())
    }
}
