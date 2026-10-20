// Explicit diagnostics only: source tensors stay borrowed through synchronous
// readback. Hashes reduce retained disk data, never the observed source extent.
pub(crate) fn prefix_row_sha256(data: &[u8], row_bytes: usize) -> Result<Vec<u8>, String> {
    use sha2::{Digest, Sha256};
    if row_bytes == 0 || data.is_empty() || data.len() % row_bytes != 0 {
        return Err("SHA256 rows require a nonempty exact row extent".into());
    }
    let bytes = (data.len() / row_bytes)
        .checked_mul(32)
        .ok_or("row digest extent overflow")?;
    let mut out = Vec::new();
    out.try_reserve_exact(bytes)
        .map_err(|e| format!("row digest allocation: {e}"))?;
    for row in data.chunks_exact(row_bytes) {
        out.extend_from_slice(&Sha256::digest(row));
    }
    Ok(out)
}
impl Target {
    pub(crate) fn digest_child(&self, name: &str) -> Result<Self, String> {
        Self {
            row: self.row,
            directory: self.directory.clone(),
        }
        .child(name)
    }
    pub(crate) fn read_digest_source(
        &self,
        name: &str,
        ptr: u64,
        bytes: usize,
        maximum: usize,
        stream: cuda::CUstream,
    ) -> Result<Vec<u8>, String> {
        if ptr == 0 || bytes == 0 || maximum == 0 || bytes > maximum {
            return Err("digest source null/empty/exceeds admitted byte bound".into());
        }
        ptr.checked_add(u64::try_from(bytes).map_err(|_| "digest source extent exceeds u64")?)
            .ok_or("digest source pointer overflow")?;
        let rc = unsafe { cuda::lib().cuStreamSynchronize(stream) };
        if rc != cuda::CUresult::CUDA_SUCCESS {
            return Err(format!("digest source completion: {rc:?}"));
        }
        let mut host = Vec::new();
        host.try_reserve_exact(bytes)
            .map_err(|e| format!("digest source host allocation: {e}"))?;
        host.resize(bytes, 0u8);
        let rc = unsafe { cuda::lib().cuMemcpyDtoH_v2(host.as_mut_ptr().cast(), ptr, bytes) };
        if rc != cuda::CUresult::CUDA_SUCCESS {
            return Err(format!("digest source download {name}: {rc:?}"));
        }
        Ok(host)
    }
    pub(crate) fn device_row_sha256(
        &self,
        name: &str,
        ptr: u64,
        bytes: usize,
        row_bytes: usize,
        maximum: usize,
        stream: cuda::CUstream,
    ) -> Result<usize, String> {
        if row_bytes == 0 || bytes == 0 || bytes % row_bytes != 0 {
            return Err("invalid digest row geometry".into());
        }
        let host = self.read_digest_source(name, ptr, bytes, maximum, stream)?;
        let digest = prefix_row_sha256(&host, row_bytes)?;
        self.write(name, &digest)?;
        Ok(digest.len())
    }
}
