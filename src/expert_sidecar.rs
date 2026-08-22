//! Versioned, expert-contiguous storage for bit-exact compressed Marlin experts.

use crate::expert_codec::{CodecResult, CodecTables};
use memmap2::{MmapMut, MmapOptions};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::fs::File;
#[cfg(unix)]
use std::os::unix::fs::FileExt as UnixFileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt as WindowsFileExt;
use std::path::{Path, PathBuf};

pub const SIDECAR_MAGIC: [u8; 8] = *b"KRASRANS";
pub const SIDECAR_VERSION: u32 = 1;
pub const SIDECAR_HEADER_BYTES: usize = 160;

#[derive(Clone, Debug)]
pub struct ExpertSidecarHeader {
    pub source_cache_bytes: u64,
    pub source_header_sha256: [u8; 32],
    pub routed_expert_sha256: [u8; 32],
    pub expert_bytes: usize,
    pub expert_count: usize,
    pub lane_bytes: usize,
    pub tables_offset: usize,
    pub index_offset: usize,
    pub payload_offset: usize,
    pub sidecar_bytes: usize,
}

pub struct ExpertSidecar {
    path: PathBuf,
    // CUDA host registration rejects read-only file mappings. Keep the
    // on-disk artifact immutable while exposing bounded writable VMAs through
    // MAP_PRIVATE. Some NVIDIA drivers associate registration with the whole
    // VMA, so a sidecar-sized mapping is unsafe even if registration names a
    // small subrange.
    payload_mappings: Vec<MmapMut>,
    blob_locations: Vec<BlobLocation>,
    header: ExpertSidecarHeader,
    tables: CodecTables,
    offsets: Vec<u64>,
}

#[derive(Clone, Copy, Debug)]
struct BlobLocation {
    mapping_idx: usize,
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PayloadMappingGroup {
    ordinal_start: usize,
    ordinal_end: usize,
    file_start: usize,
    file_end: usize,
}

impl ExpertSidecar {
    pub fn open(
        path: &Path,
        source_header: &[u8],
        source_cache_bytes: u64,
        routed_expert_sha256: [u8; 32],
        expected_expert_bytes: usize,
        expected_expert_count: usize,
        max_mapping_bytes: usize,
    ) -> CodecResult<Self> {
        let file = File::open(path).map_err(|error| {
            format!("failed to open expert sidecar {}: {error}", path.display())
        })?;
        let file_bytes = as_usize(
            file.metadata()
                .map_err(|error| {
                    format!("failed to stat expert sidecar {}: {error}", path.display())
                })?
                .len(),
            "file length",
        )?;
        let mut encoded_header = [0_u8; SIDECAR_HEADER_BYTES];
        read_file_exact_at(&file, &mut encoded_header, 0).map_err(|error| {
            format!(
                "failed to read expert sidecar header {}: {error}",
                path.display()
            )
        })?;
        let header = parse_header_for_file(&encoded_header, file_bytes)?;
        if header.source_cache_bytes != source_cache_bytes {
            return Err(format!(
                "expert sidecar source length {} != loaded cache {}",
                header.source_cache_bytes, source_cache_bytes,
            ));
        }
        let actual_header_hash = source_header_sha256(source_header);
        if header.source_header_sha256 != actual_header_hash {
            return Err("expert sidecar source-cache header hash mismatch".to_string());
        }
        if header.routed_expert_sha256 != routed_expert_sha256 {
            return Err("expert sidecar routed-expert payload hash mismatch".to_string());
        }
        if header.expert_bytes != expected_expert_bytes
            || header.expert_count != expected_expert_count
        {
            return Err(format!(
                "expert sidecar geometry experts={} bytes/expert={} != loaded cache experts={} bytes/expert={}",
                header.expert_count,
                header.expert_bytes,
                expected_expert_count,
                expected_expert_bytes,
            ));
        }
        let tables_end = header
            .tables_offset
            .checked_add(CodecTables::SERIALIZED_FREQUENCIES_BYTES)
            .ok_or_else(|| "expert sidecar table range overflow".to_string())?;
        if tables_end > file_bytes {
            return Err("expert sidecar tables are truncated".to_string());
        }
        let mut encoded_tables = vec![0_u8; CodecTables::SERIALIZED_FREQUENCIES_BYTES];
        read_file_exact_at(&file, &mut encoded_tables, header.tables_offset as u64)
            .map_err(|error| format!("failed to read expert sidecar tables: {error}"))?;
        let tables = CodecTables::from_serialized_frequencies(&encoded_tables)?;
        let index_entries = header
            .expert_count
            .checked_add(1)
            .ok_or_else(|| "expert sidecar index count overflow".to_string())?;
        let index_bytes = index_entries
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| "expert sidecar index size overflow".to_string())?;
        let index_end = header
            .index_offset
            .checked_add(index_bytes)
            .ok_or_else(|| "expert sidecar index range overflow".to_string())?;
        if index_end > file_bytes {
            return Err("expert sidecar index is truncated".to_string());
        }
        let mut encoded_index = vec![0_u8; index_bytes];
        read_file_exact_at(&file, &mut encoded_index, header.index_offset as u64)
            .map_err(|error| format!("failed to read expert sidecar index: {error}"))?;
        let offsets = encoded_index
            .chunks_exact(8)
            .map(|raw| u64::from_le_bytes(raw.try_into().unwrap()))
            .collect::<Vec<_>>();
        if offsets.first().copied() != Some(header.payload_offset as u64)
            || offsets.last().copied() != Some(header.sidecar_bytes as u64)
            || offsets.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(
                "expert sidecar index is not a complete strictly increasing range".to_string(),
            );
        }
        let mapping_granularity = system_mapping_granularity()?;
        let groups = payload_mapping_groups(&offsets, max_mapping_bytes, mapping_granularity)?;
        let mut payload_mappings = Vec::with_capacity(groups.len());
        let mut blob_locations = Vec::with_capacity(header.expert_count);
        for group in groups {
            let mapping_bytes = group
                .file_end
                .checked_sub(group.file_start)
                .ok_or_else(|| "expert sidecar mapping range underflow".to_string())?;
            let mmap = unsafe {
                MmapOptions::new()
                    .offset(group.file_start as u64)
                    .len(mapping_bytes)
                    .map_copy(&file)
            }
            .map_err(|error| {
                format!(
                    "failed to privately map expert sidecar {} bytes {}..{}: {error}",
                    path.display(),
                    group.file_start,
                    group.file_end,
                )
            })?;
            let mapping_idx = payload_mappings.len();
            for ordinal in group.ordinal_start..group.ordinal_end {
                let start = as_usize(offsets[ordinal], "expert offset")?
                    .checked_sub(group.file_start)
                    .ok_or_else(|| "expert sidecar local offset underflow".to_string())?;
                let end = as_usize(offsets[ordinal + 1], "expert end offset")?
                    .checked_sub(group.file_start)
                    .ok_or_else(|| "expert sidecar local end underflow".to_string())?;
                if end > mmap.len() || start >= end {
                    return Err(format!(
                        "expert sidecar blob {ordinal} is outside its bounded mapping",
                    ));
                }
                blob_locations.push(BlobLocation {
                    mapping_idx,
                    start,
                    end,
                });
            }
            payload_mappings.push(mmap);
        }
        if blob_locations.len() != header.expert_count {
            return Err(format!(
                "expert sidecar mapped {} blobs, expected {}",
                blob_locations.len(),
                header.expert_count,
            ));
        }
        Ok(Self {
            path: path.to_path_buf(),
            payload_mappings,
            blob_locations,
            header,
            tables,
            offsets,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn header(&self) -> &ExpertSidecarHeader {
        &self.header
    }

    pub fn tables(&self) -> &CodecTables {
        &self.tables
    }

    pub fn mapped_ranges(&self) -> impl Iterator<Item = (*const u8, usize)> + '_ {
        self.payload_mappings
            .iter()
            .map(|mmap| (mmap.as_ptr(), mmap.len()))
    }

    pub fn mapped_bytes(&self) -> usize {
        self.payload_mappings.iter().map(|mmap| mmap.len()).sum()
    }

    pub fn blob(&self, ordinal: usize) -> CodecResult<&[u8]> {
        let location = self
            .blob_locations
            .get(ordinal)
            .ok_or_else(|| format!("expert sidecar has no ordinal {ordinal}"))?;
        self.payload_mappings[location.mapping_idx]
            .get(location.start..location.end)
            .ok_or_else(|| format!("expert sidecar blob {ordinal} range is invalid"))
    }

    pub fn max_blob_bytes(&self) -> usize {
        self.offsets
            .windows(2)
            .map(|pair| (pair[1] - pair[0]) as usize)
            .max()
            .unwrap_or(0)
    }
}

/// Map a bounded private writable prefix of the real sidecar payload. This is
/// used by the standalone GPU gate to prove the exact VMA type and
/// runtime-derived registration size without loading the model or exposing
/// the complete sidecar as one driver-visible mapping.
pub fn private_payload_mapping(path: &Path, maximum_bytes: usize) -> CodecResult<MmapMut> {
    let file = File::open(path)
        .map_err(|error| format!("failed to open expert sidecar {}: {error}", path.display()))?;
    let file_bytes = as_usize(
        file.metadata()
            .map_err(|error| format!("failed to stat expert sidecar {}: {error}", path.display()))?
            .len(),
        "file length",
    )?;
    let mut encoded_header = [0_u8; SIDECAR_HEADER_BYTES];
    read_file_exact_at(&file, &mut encoded_header, 0).map_err(|error| {
        format!(
            "failed to read expert sidecar header {}: {error}",
            path.display()
        )
    })?;
    let header = parse_header_for_file(&encoded_header, file_bytes)?;
    let mapping_granularity = system_mapping_granularity()?;
    let file_start = align_down(header.payload_offset, mapping_granularity);
    let available = file_bytes
        .checked_sub(file_start)
        .ok_or_else(|| "expert sidecar payload mapping underflow".to_string())?;
    let requested = maximum_bytes.min(available);
    let mapping_bytes = if requested == available {
        requested
    } else {
        align_down(requested, mapping_granularity)
    };
    let payload_prefix = header.payload_offset - file_start;
    if mapping_bytes <= payload_prefix {
        return Err(format!(
            "expert sidecar registration mapping {} does not reach payload offset {}",
            mapping_bytes, payload_prefix,
        ));
    }
    unsafe {
        MmapOptions::new()
            .offset(file_start as u64)
            .len(mapping_bytes)
            .map_copy(&file)
    }
    .map_err(|error| {
        format!(
            "failed to privately map expert sidecar {} payload prefix: {error}",
            path.display(),
        )
    })
}

fn payload_mapping_groups(
    offsets: &[u64],
    max_mapping_bytes: usize,
    mapping_granularity: usize,
) -> CodecResult<Vec<PayloadMappingGroup>> {
    if offsets.len() < 2 || max_mapping_bytes == 0 || mapping_granularity == 0 {
        return Err(format!(
            "invalid expert sidecar mapping geometry offsets={} max_mapping={} granularity={}",
            offsets.len(),
            max_mapping_bytes,
            mapping_granularity,
        ));
    }
    let mut groups = Vec::new();
    let expert_count = offsets.len() - 1;
    let mut ordinal_start = 0usize;
    while ordinal_start < expert_count {
        let first_start = as_usize(offsets[ordinal_start], "expert offset")?;
        let file_start = align_down(first_start, mapping_granularity);
        let first_end = as_usize(offsets[ordinal_start + 1], "expert end offset")?;
        if first_end.saturating_sub(file_start) > max_mapping_bytes {
            return Err(format!(
                "expert sidecar blob {ordinal_start} requires {} mapped bytes, exceeding runtime-derived maximum {}",
                first_end.saturating_sub(file_start),
                max_mapping_bytes,
            ));
        }
        let mut ordinal_end = ordinal_start + 1;
        while ordinal_end < expert_count {
            let candidate_end = as_usize(offsets[ordinal_end + 1], "expert end offset")?;
            if candidate_end.saturating_sub(file_start) > max_mapping_bytes {
                break;
            }
            ordinal_end += 1;
        }
        let file_end = as_usize(offsets[ordinal_end], "mapping end offset")?;
        groups.push(PayloadMappingGroup {
            ordinal_start,
            ordinal_end,
            file_start,
            file_end,
        });
        ordinal_start = ordinal_end;
    }
    Ok(groups)
}

#[cfg(unix)]
pub fn system_mapping_granularity() -> CodecResult<usize> {
    let page_bytes = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_bytes <= 0 {
        Err("expert sidecar could not determine the host mapping granularity".to_string())
    } else {
        Ok(page_bytes as usize)
    }
}

#[cfg(windows)]
pub fn system_mapping_granularity() -> CodecResult<usize> {
    use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};

    let mut info = SYSTEM_INFO::default();
    unsafe { GetSystemInfo(&mut info) };
    usize::try_from(info.dwAllocationGranularity)
        .ok()
        .filter(|&value| value > 0)
        .ok_or_else(|| {
            "expert sidecar could not determine the host mapping granularity".to_string()
        })
}

#[cfg(unix)]
pub fn read_file_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> std::io::Result<()> {
    UnixFileExt::read_exact_at(file, buffer, offset)
}

#[cfg(windows)]
pub fn read_file_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> std::io::Result<()> {
    let mut filled = 0usize;
    while filled < buffer.len() {
        let read = match WindowsFileExt::seek_read(
            file,
            &mut buffer[filled..],
            offset
                .checked_add(filled as u64)
                .ok_or_else(|| std::io::Error::other("file offset overflow"))?,
        ) {
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if read == 0 {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
        }
        filled = filled
            .checked_add(read)
            .ok_or_else(|| std::io::Error::other("positional file-read length overflow"))?;
    }
    Ok(())
}

fn align_down(value: usize, alignment: usize) -> usize {
    value - value % alignment
}

pub fn source_header_sha256(source_header: &[u8]) -> [u8; 32] {
    Sha256::digest(source_header).into()
}

/// Stable, parallel identity for the exact routed-expert payload. Expert
/// boundaries are part of the cache format, so the digest is independent of
/// host thread count while allowing all experts to be hashed concurrently.
pub fn routed_expert_sha256(bytes: &[u8], expert_bytes: usize) -> CodecResult<[u8; 32]> {
    if expert_bytes == 0 || bytes.is_empty() || bytes.len() % expert_bytes != 0 {
        return Err(format!(
            "routed expert identity range {} is not a positive multiple of expert bytes {}",
            bytes.len(),
            expert_bytes,
        ));
    }
    let expert_digests = bytes
        .par_chunks(expert_bytes)
        .map(source_header_sha256)
        .collect::<Vec<_>>();
    routed_expert_sha256_from_digests(expert_bytes, &expert_digests)
}

pub fn routed_expert_sha256_from_digests(
    expert_bytes: usize,
    expert_digests: &[[u8; 32]],
) -> CodecResult<[u8; 32]> {
    if expert_bytes == 0 || expert_digests.is_empty() {
        return Err("routed expert identity requires positive geometry".to_string());
    }
    let mut hasher = Sha256::new();
    hasher.update(b"KRASIS_ROUTED_EXPERT_SHA256_V1");
    hasher.update((expert_bytes as u64).to_le_bytes());
    hasher.update((expert_digests.len() as u64).to_le_bytes());
    for digest in expert_digests {
        hasher.update(digest);
    }
    Ok(hasher.finalize().into())
}

pub fn encode_header(header: &ExpertSidecarHeader) -> CodecResult<[u8; SIDECAR_HEADER_BYTES]> {
    let mut encoded = [0_u8; SIDECAR_HEADER_BYTES];
    encoded[0..8].copy_from_slice(&SIDECAR_MAGIC);
    write_u32(&mut encoded, 8, SIDECAR_VERSION);
    write_u32(&mut encoded, 12, SIDECAR_HEADER_BYTES as u32);
    write_u64(&mut encoded, 16, header.source_cache_bytes);
    encoded[24..56].copy_from_slice(&header.source_header_sha256);
    encoded[56..88].copy_from_slice(&header.routed_expert_sha256);
    write_u64(
        &mut encoded,
        88,
        as_u64(header.expert_bytes, "expert_bytes")?,
    );
    write_u64(
        &mut encoded,
        96,
        as_u64(header.expert_count, "expert_count")?,
    );
    write_u64(&mut encoded, 104, as_u64(header.lane_bytes, "lane_bytes")?);
    write_u64(
        &mut encoded,
        112,
        as_u64(header.tables_offset, "tables_offset")?,
    );
    write_u64(
        &mut encoded,
        120,
        as_u64(header.index_offset, "index_offset")?,
    );
    write_u64(
        &mut encoded,
        128,
        as_u64(header.payload_offset, "payload_offset")?,
    );
    write_u64(
        &mut encoded,
        136,
        as_u64(header.sidecar_bytes, "sidecar_bytes")?,
    );
    Ok(encoded)
}

pub fn parse_header(bytes: &[u8]) -> CodecResult<ExpertSidecarHeader> {
    parse_header_for_file(bytes, bytes.len())
}

pub fn parse_header_for_file(bytes: &[u8], file_bytes: usize) -> CodecResult<ExpertSidecarHeader> {
    if bytes.len() < SIDECAR_HEADER_BYTES || bytes[0..8] != SIDECAR_MAGIC {
        return Err("expert sidecar magic/header mismatch".to_string());
    }
    let version = read_u32(bytes, 8)?;
    let header_bytes = read_u32(bytes, 12)? as usize;
    if version != SIDECAR_VERSION || header_bytes != SIDECAR_HEADER_BYTES {
        return Err(format!(
            "unsupported expert sidecar version/header {version}/{header_bytes}"
        ));
    }
    let mut source_header_sha256 = [0_u8; 32];
    source_header_sha256.copy_from_slice(&bytes[24..56]);
    let mut routed_expert_sha256 = [0_u8; 32];
    routed_expert_sha256.copy_from_slice(&bytes[56..88]);
    let header = ExpertSidecarHeader {
        source_cache_bytes: read_u64(bytes, 16)?,
        source_header_sha256,
        routed_expert_sha256,
        expert_bytes: as_usize(read_u64(bytes, 88)?, "expert_bytes")?,
        expert_count: as_usize(read_u64(bytes, 96)?, "expert_count")?,
        lane_bytes: as_usize(read_u64(bytes, 104)?, "lane_bytes")?,
        tables_offset: as_usize(read_u64(bytes, 112)?, "tables_offset")?,
        index_offset: as_usize(read_u64(bytes, 120)?, "index_offset")?,
        payload_offset: as_usize(read_u64(bytes, 128)?, "payload_offset")?,
        sidecar_bytes: as_usize(read_u64(bytes, 136)?, "sidecar_bytes")?,
    };
    if header.lane_bytes == 0
        || header.expert_bytes == 0
        || header.expert_count == 0
        || header.tables_offset < SIDECAR_HEADER_BYTES
        || header.index_offset < header.tables_offset + CodecTables::SERIALIZED_FREQUENCIES_BYTES
        || header.payload_offset < header.index_offset
        || header.sidecar_bytes != file_bytes
    {
        return Err("expert sidecar header ranges are inconsistent".to_string());
    }
    Ok(header)
}

fn read_u32(bytes: &[u8], offset: usize) -> CodecResult<u32> {
    bytes
        .get(offset..offset + 4)
        .map(|raw| u32::from_le_bytes(raw.try_into().unwrap()))
        .ok_or_else(|| format!("missing sidecar u32 at {offset}"))
}

fn read_u64(bytes: &[u8], offset: usize) -> CodecResult<u64> {
    bytes
        .get(offset..offset + 8)
        .map(|raw| u64::from_le_bytes(raw.try_into().unwrap()))
        .ok_or_else(|| format!("missing sidecar u64 at {offset}"))
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn as_usize(value: u64, label: &str) -> CodecResult<usize> {
    usize::try_from(value).map_err(|_| format!("expert sidecar {label} exceeds usize"))
}

fn as_u64(value: usize, label: &str) -> CodecResult<u64> {
    u64::try_from(value).map_err(|_| format!("expert sidecar {label} exceeds u64"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_header_round_trip_is_versioned() {
        let header = ExpertSidecarHeader {
            source_cache_bytes: 91,
            source_header_sha256: source_header_sha256(b"real-cache-header"),
            routed_expert_sha256: routed_expert_sha256(b"abcdef", 2).unwrap(),
            expert_bytes: 17,
            expert_count: 3,
            lane_bytes: 1024,
            tables_offset: SIDECAR_HEADER_BYTES,
            index_offset: SIDECAR_HEADER_BYTES + CodecTables::SERIALIZED_FREQUENCIES_BYTES,
            payload_offset: SIDECAR_HEADER_BYTES + CodecTables::SERIALIZED_FREQUENCIES_BYTES + 32,
            sidecar_bytes: SIDECAR_HEADER_BYTES
                + CodecTables::SERIALIZED_FREQUENCIES_BYTES
                + 32
                + 51,
        };
        let encoded = encode_header(&header).unwrap();
        let mut whole = encoded.to_vec();
        whole.resize(header.sidecar_bytes, 0);
        let decoded = parse_header(&whole).unwrap();
        assert_eq!(decoded.source_header_sha256, header.source_header_sha256);
        assert_eq!(decoded.routed_expert_sha256, header.routed_expert_sha256);
        assert_eq!(decoded.expert_count, 3);
        assert_eq!(decoded.lane_bytes, 1024);
    }

    #[test]
    fn routed_identity_is_geometry_stable_and_payload_sensitive() {
        let source = b"abcdefgh";
        let first = routed_expert_sha256(source, 2).unwrap();
        assert_eq!(first, routed_expert_sha256(source, 2).unwrap());
        assert_ne!(first, routed_expert_sha256(b"abcdefgi", 2).unwrap());
        assert_ne!(first, routed_expert_sha256(source, 4).unwrap());
    }

    #[test]
    fn payload_groups_keep_whole_experts_in_runtime_sized_mappings() {
        let offsets = vec![160, 1_160, 2_460, 3_260, 4_660];
        let groups = payload_mapping_groups(&offsets, 2_500, 64).unwrap();
        assert_eq!(
            groups,
            vec![
                PayloadMappingGroup {
                    ordinal_start: 0,
                    ordinal_end: 2,
                    file_start: 128,
                    file_end: 2_460,
                },
                PayloadMappingGroup {
                    ordinal_start: 2,
                    ordinal_end: 4,
                    file_start: 2_432,
                    file_end: 4_660,
                },
            ]
        );
        assert!(payload_mapping_groups(&offsets, 1_000, 64).is_err());
    }
}
