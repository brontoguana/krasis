// Private diagnostic artifact storage only; GPU execution and arithmetic are unchanged.
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::{self, BufWriter, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

pub fn write_sparse_new(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = BufWriter::new(file);
    let threshold = writer.capacity();
    let mut pending = 0;
    let mut cursor = 0;
    while cursor < bytes.len() {
        if bytes[cursor] != 0 {
            cursor += 1;
            continue;
        }
        let start = cursor;
        while cursor < bytes.len() && bytes[cursor] == 0 {
            cursor += 1;
        }
        if cursor - start >= threshold {
            writer.write_all(&bytes[pending..start])?;
            writer.seek(SeekFrom::Start(
                u64::try_from(cursor).map_err(|_| io::Error::other("artifact offset overflow"))?,
            ))?;
            pending = cursor;
        }
    }
    writer.write_all(&bytes[pending..])?;
    writer.flush()?;
    writer.get_ref().set_len(
        u64::try_from(bytes.len()).map_err(|_| io::Error::other("artifact length overflow"))?,
    )?;
    Ok(())
}

#[derive(Serialize)]
pub struct PrefixSpan {
    kind: &'static str,
    storage_path: PathBuf,
    storage_bytes: usize,
    storage_sha256: String,
    offset: usize,
    bytes: usize,
    sha256: String,
}

// Caller has compared every measured prefix word against this full allocation.
// The auditor independently reads the backing bytes and checks both hashes.
pub fn prefix_span(
    path: &Path,
    storage_bytes: usize,
    storage_sha256: &str,
    measured: &[u8],
) -> Result<PrefixSpan, String> {
    if measured.len() > storage_bytes
        || fs::metadata(path).map_err(|e| e.to_string())?.len()
            != u64::try_from(storage_bytes).map_err(|e| e.to_string())?
    {
        return Err("invalid measured artifact span".into());
    }
    Ok(PrefixSpan {
        kind: "measured-prefix-span-v1",
        storage_path: path.to_owned(),
        storage_bytes,
        storage_sha256: storage_sha256.to_owned(),
        offset: 0,
        bytes: measured.len(),
        sha256: format!("{:x}", Sha256::digest(measured)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_raw_roundtrip_and_explicit_span_contract() {
        let root = std::env::temp_dir().join(format!(
            "krasis-artifact-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let capacity = BufWriter::new(Vec::<u8>::new()).capacity();
        let mut mixed = vec![0; capacity * 5 + 1];
        for (pos, val) in [
            (capacity - 1, 1),
            (capacity * 2, 128),
            (capacity * 3 + 1, 255),
            (capacity * 4, 17),
        ] {
            mixed[pos] = val;
        }
        let cases = [
            Vec::new(),
            vec![0],
            vec![0; capacity - 1],
            vec![0; capacity],
            vec![0; capacity * 3 + 1],
            vec![255; capacity + 1],
            mixed,
        ];
        for (index, bytes) in cases.iter().enumerate() {
            let path = root.join(index.to_string());
            write_sparse_new(&path, bytes).unwrap();
            let actual = fs::read(&path).unwrap();
            assert_eq!(actual, *bytes);
            let full_sha = format!("{:x}", Sha256::digest(&actual));
            for len in [0, bytes.len() / 2, bytes.len()] {
                let span = prefix_span(&path, bytes.len(), &full_sha, &bytes[..len]).unwrap();
                let v = serde_json::to_value(span).unwrap();
                assert_eq!(v["kind"], "measured-prefix-span-v1");
                assert_eq!(v["storage_sha256"], full_sha);
                assert_eq!(v["sha256"], format!("{:x}", Sha256::digest(&actual[..len])));
                assert_eq!(v["bytes"].as_u64(), Some(len as u64));
                assert_eq!(v["offset"], 0);
            }
            assert_eq!(
                write_sparse_new(&path, b"overwrite").unwrap_err().kind(),
                io::ErrorKind::AlreadyExists
            );
            assert_eq!(fs::read(&path).unwrap(), *bytes);
            assert!(prefix_span(&path, bytes.len() + 1, &full_sha, bytes).is_err());
            assert!(prefix_span(&path, bytes.len(), &full_sha, &vec![1; bytes.len() + 1]).is_err());
        }
        assert!(write_sparse_new(&root.join("missing/child"), b"x").is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
