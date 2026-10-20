//! Exact source-record ablation assembly and independent verification via ./dev.
#[path = "../weights/mixed_precision.rs"]
mod mixed_precision;
#[path = "../weights/scales_ablation.rs"]
mod scales_ablation;

use mixed_precision::*;
use scales_ablation::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
type Result<T> = std::result::Result<T, String>;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    source_reserve_path: String,
    source_reserve_sha256: String,
    original_amax_path: String,
    original_amax: MixedCacheSourceIdentity,
    coverage_path: String,
    coverage_sha256: String,
    protocol_path: String,
    protocol_sha256: String,
    expected_replacements: usize,
}

fn read_bound(path: &Path, expected: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if sha256_hex(&bytes) != expected {
        return Err(format!("identity changed: {}", path.display()));
    }
    Ok(bytes)
}
fn range(file: &mut File, offset: u64, count: u64) -> Result<String> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| e.to_string())?;
    let mut buffer = vec![0; 8 * 1024 * 1024];
    let mut remaining = count;
    let mut hash = Sha256::new();
    while remaining > 0 {
        let n = remaining.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..n])
            .map_err(|e| e.to_string())?;
        hash.update(&buffer[..n]);
        remaining -= n as u64;
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn open_bound(path: &Path, bytes: u64, hash: &str) -> Result<File> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    file.try_lock_shared().map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len() != bytes
        || range(&mut file, 0, bytes)? != hash
    {
        return Err(format!("source size/hash mismatch: {}", path.display()));
    }
    println!("SOURCE_FULL_HASH_PASS {} {hash}", path.display());
    Ok(file)
}
fn record(file: &mut File, offset: u64, bytes: u64) -> Result<Vec<u8>> {
    let mut data = vec![0; usize::try_from(bytes).map_err(|e| e.to_string())?];
    file.seek(SeekFrom::Start(offset))
        .and_then(|_| file.read_exact(&mut data))
        .map_err(|e| e.to_string())?;
    Ok(data)
}
fn unhex(s: &str) -> Result<Vec<u8>> {
    if s.len() != 64 {
        return Err("invalid hash length".into());
    }
    (0..32)
        .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).map_err(|e| e.to_string()))
        .collect()
}
fn new_write(path: &Path, data: &[u8]) -> Result<()> {
    let mut f = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    f.write_all(data)
        .and_then(|_| f.sync_all())
        .map_err(|e| e.to_string())
}
fn compact_header(m: &MixedPrecisionManifest, payload_sha: &str) -> Result<Vec<u8>> {
    let mut h = Vec::from(*MIXED_COMPACT_CACHE_MAGIC);
    h.extend(MIXED_COMPACT_CACHE_VERSION.to_le_bytes());
    h.extend(128u64.to_le_bytes());
    h.extend(m.budget.baseline_int4_routed_bytes.to_le_bytes());
    h.extend(unhex(payload_sha)?);
    h.extend(unhex(&region_map_sha256(&m.regions)?)?);
    h.extend(unhex(&m.checkpoint_sha256)?);
    h.resize(128, 0);
    Ok(h)
}
fn validate_original_header(
    h: &[u8],
    s: &MixedPrecisionManifest,
    id: &MixedCacheSourceIdentity,
) -> Result<()> {
    let u = |start| u64::from_le_bytes(h[start..start + 8].try_into().unwrap());
    if h.len() != 64
        || &h[..4] != b"KRAS"
        || u32::from_le_bytes(h[4..8].try_into().unwrap()) != 7
        || u(8) != s.hidden_size as u64
        || u(16) != s.intermediate_size as u64
        || u(24) != s.routed_experts_per_layer as u64
        || u(32) != s.routed_layers as u64
        || u(40) != s.group_size as u64
        || u(48) != id.header_config_fnv1a
        || u(56) >> 32 != 0
        || sha256_hex(h) != id.header_sha256
        || id.bits != 4
        || id.expert_int4_calibration_mode != "amax"
    {
        return Err("original amax header/geometry identity mismatch".into());
    }
    Ok(())
}
fn run(mode: &str, plan_path: &Path, out: &Path) -> Result<()> {
    let plan_raw = fs::read(plan_path).map_err(|e| e.to_string())?;
    let p: Plan = serde_json::from_slice(&plan_raw).map_err(|e| e.to_string())?;
    let source_raw = read_bound(Path::new(&p.source_reserve_path), &p.source_reserve_sha256)?;
    let source = parse_canonical_manifest(&source_raw)?;
    let g = geometry(&source)?;
    validate_manifest(
        &source,
        &g,
        &source.checkpoint_sha256,
        &source.cache_namespace,
        &source.config_sha256,
    )?;
    let coverage = read_bound(Path::new(&p.coverage_path), &p.coverage_sha256)?;
    validate_coverage(&source, &coverage)?;
    read_bound(Path::new(&p.protocol_path), &p.protocol_sha256)?;
    let indices = replacement_indices(&source)?;
    if indices.len() != p.expected_replacements || indices.is_empty() {
        return Err("frozen replacement cardinality mismatch".into());
    }
    let source_dir = Path::new(&p.source_reserve_path)
        .parent()
        .ok_or("source has no parent")?;
    let ranking =
        fs::read(source_dir.join("mixed-expert-ranking.json")).map_err(|e| e.to_string())?;
    validate_ranking_evidence_bytes(&source, &ranking)?;
    let id = source
        .compact_cache
        .as_ref()
        .ok_or("source has no compact cache")?;
    let mut bank = open_bound(&source_dir.join(&id.basename), id.bytes, &id.sha256)?;
    let source_header = record(&mut bank, 0, 128)?;
    // Whole-file identity is bound above; independently check header fields and payload digest.
    let mut source_for_header = source.clone();
    source_for_header.budget.baseline_int4_routed_bytes = id.bytes - 128;
    let expected_header = compact_header(&source_for_header, &id.payload_sha256)?;
    if source_header != expected_header
        || sha256_hex(&source_header) != id.header_sha256
        || id.region_map_sha256 != region_map_sha256(&source.regions)?
        || range(&mut bank, 128, id.bytes - 128)? != id.payload_sha256
    {
        return Err("source compact header/payload mismatch".into());
    }
    let mut original = open_bound(
        Path::new(&p.original_amax_path),
        p.original_amax.bytes,
        &p.original_amax.sha256,
    )?;
    validate_original_header(&record(&mut original, 0, 64)?, &source, &p.original_amax)?;
    let mut m = source.clone();
    m.format = FORMAT.into();
    m.schema_version = SCHEMA_VERSION;
    m.regions = project_regions(&source)?;
    m.budget.maximum_added_bytes = 0;
    m.budget.achieved_added_bytes = 0;
    m.budget.achieved_percent_millionths = 0;
    let cache_path = out.join("scales-only.krmx");
    let manifest_path = out.join("scales-only.manifest.json");
    let mut cache;
    let verified;
    if mode == "assemble" {
        fs::create_dir(out).map_err(|e| format!("new output directory required: {e}"))?;
        new_write(&out.join("assembly-plan.json"), &plan_raw)?;
        cache = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&cache_path)
            .map_err(|e| e.to_string())?;
        cache.write_all(&[0; 128]).map_err(|e| e.to_string())?;
        verified = None;
    } else if mode == "verify" {
        let raw = fs::read(&manifest_path).map_err(|e| e.to_string())?;
        let existing = parse_canonical_manifest(&raw)?;
        validate_manifest(
            &existing,
            &g,
            &source.checkpoint_sha256,
            &source.cache_namespace,
            &source.config_sha256,
        )?;
        let prov = existing.scales_only.as_ref().ok_or("missing provenance")?;
        if prov.source_reserve_sha256 != p.source_reserve_sha256
            || prov.original_amax != p.original_amax
            || prov.coverage_sha256 != p.coverage_sha256
            || prov.protocol_sha256 != p.protocol_sha256
        {
            return Err("verification plan/provenance mismatch".into());
        }
        let identity = existing.compact_cache.as_ref().unwrap();
        cache = open_bound(&cache_path, identity.bytes, &identity.sha256)?;
        let output_header = record(&mut cache, 0, 128)?;
        if output_header != compact_header(&existing, &identity.payload_sha256)?
            || sha256_hex(&output_header) != identity.header_sha256
            || range(&mut cache, 128, identity.bytes - 128)? != identity.payload_sha256
            || read_bound(&out.join("source-coverage.json"), &p.coverage_sha256)? != coverage
            || fs::read(out.join("mixed-expert-ranking.json")).map_err(|e| e.to_string())?
                != ranking
        {
            return Err("output header/payload/evidence mismatch".into());
        }
        verified = Some(existing);
    } else {
        return Err("expected assemble or verify".into());
    }
    let mut records = Vec::new();
    let mut offset = 128;
    let mut dest = 128;
    let mut payload = Sha256::new();
    for (i, r) in source.regions.iter().enumerate() {
        let rare = indices.binary_search(&i).is_ok();
        let source_offset = if rare {
            64 + i as u64 * r.int4_bytes
        } else {
            offset
        };
        let data = record(
            if rare { &mut original } else { &mut bank },
            source_offset,
            r.int4_bytes,
        )?;
        let hash = sha256_hex(&data);
        let provenance = SourceRecord {
            index: i,
            source: if rare {
                "original_amax"
            } else {
                "source_reserve"
            }
            .into(),
            source_offset,
            output_offset: dest,
            bytes: r.int4_bytes,
            sha256: hash,
        };
        if let Some(v) = &verified {
            if record(&mut cache, dest, r.int4_bytes)? != data
                || v.scales_only.as_ref().unwrap().records[i] != provenance
            {
                return Err(format!(
                    "source/destination byte or provenance mismatch {i}"
                ));
            }
        } else {
            cache.write_all(&data).map_err(|e| e.to_string())?;
        }
        payload.update(&data);
        records.push(provenance);
        offset += if r.bits == 4 {
            r.int4_bytes
        } else {
            r.int8_bytes
        };
        dest += r.int4_bytes;
        if i % source.routed_experts_per_layer as usize == 0 {
            println!("RECORDS_{mode} layer={} index={i}", r.model_layer);
        }
    }
    if mode == "assemble" {
        let payload_sha = format!("{:x}", payload.finalize());
        let h = compact_header(&m, &payload_sha)?;
        cache
            .seek(SeekFrom::Start(0))
            .and_then(|_| cache.write_all(&h))
            .and_then(|_| cache.sync_all())
            .map_err(|e| e.to_string())?;
        m.compact_cache = Some(MixedCompactCacheIdentity {
            basename: "scales-only.krmx".into(),
            bytes: dest,
            sha256: range(&mut cache, 0, dest)?,
            header_sha256: sha256_hex(&h),
            payload_sha256: payload_sha,
            region_map_sha256: region_map_sha256(&m.regions)?,
        });
        m.scales_only = Some(Box::new(ScalesOnlyProvenance {
            method: "copy_searched_int4_replace_coverage_reserve_with_original_amax_v1".into(),
            protocol_sha256: p.protocol_sha256.clone(),
            coverage_sha256: p.coverage_sha256.clone(),
            source_reserve_sha256: p.source_reserve_sha256.clone(),
            source_reserve: source.clone(),
            original_amax: p.original_amax.clone(),
            replacement_indices: indices.clone(),
            records,
        }));
        validate_manifest(
            &m,
            &g,
            &source.checkpoint_sha256,
            &source.cache_namespace,
            &source.config_sha256,
        )?;
        new_write(&out.join("source-coverage.json"), &coverage)?;
        new_write(&out.join("mixed-expert-ranking.json"), &ranking)?;
        // Publish manifest last; incomplete outputs can never be loaded as a treatment.
        new_write(
            &manifest_path,
            &serde_json::to_vec(&m).map_err(|e| e.to_string())?,
        )?;
        File::open(out)
            .and_then(|f| f.sync_all())
            .map_err(|e| e.to_string())?;
    }
    // Re-read small frozen inputs after the copy; whole-source verification is repeated
    // by a separate verify invocation before admission to any runtime.
    read_bound(Path::new(&p.source_reserve_path), &p.source_reserve_sha256)?;
    read_bound(Path::new(&p.coverage_path), &p.coverage_sha256)?;
    read_bound(Path::new(&p.protocol_path), &p.protocol_sha256)?;
    println!(
        "SCALES_ONLY_{}_PASS records={} replacements={}",
        mode.to_uppercase(),
        source.regions.len(),
        indices.len()
    );
    Ok(())
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    if std::env::var("KRASIS_DEV_SCRIPT").as_deref() != Ok("1") {
        eprintln!("use ./dev scales-only");
        std::process::exit(2);
    }
    if args.len() != 4 {
        eprintln!("usage: ./dev scales-only <assemble|verify> <plan.json> <output-dir>");
        std::process::exit(2);
    }
    if let Err(e) = run(&args[1], Path::new(&args[2]), Path::new(&args[3])) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn exact_copy_rejects_corruption_overwrite_and_changed_frozen_inputs() {
        let root = std::env::temp_dir().join(format!(
            "krasis-scales-only-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let mut source = scales_ablation::tests::fixture()
            .scales_only
            .unwrap()
            .source_reserve;
        let candidates:Vec<_>=source.regions.iter().map(|r| {
            let cov=r.coverage.as_ref().unwrap();
            let mut row=json!({"model_layer":r.model_layer,"moe_layer":r.moe_layer,"expert":r.expert,
                "int4_bytes":r.int4_bytes,"int8_bytes":r.int8_bytes,"route_count":cov.route_count,
                "corpus_route_counts":cov.corpus_route_counts,"coverage_required_int8":r.bits==8});
            if let Some(g)=r.quality_gain_units {row["expected_gain_units"]=json!(g);row["expected_rank"]=json!(r.quality_rank.unwrap());}
            row
        }).collect();
        let ranking =
            serde_json::to_vec(&json!({"tail_fraction":0.01,"minimum_tail_observations":5,
            "coverage_policy":"underobserved_int8_v1","minimum_routes_per_expert":500,
            "search_rmse_calibration_sha256":source.calibration.search_rmse_calibration_sha256,
            "candidates":candidates}))
            .unwrap();
        source.calibration.ranking_evidence_sha256 = sha256_hex(&ranking);
        let counts: Vec<_> = source
            .regions
            .iter()
            .map(|r| r.coverage.as_ref().unwrap().route_count)
            .collect();
        let coverage=serde_json::to_vec(&json!({"minimum_routes_per_expert":500,"layers":[{"model_layer":3,"route_counts":counts,"corpora":[{"route_counts":counts}]}]})).unwrap();
        let payload: Vec<_> = source
            .regions
            .iter()
            .enumerate()
            .flat_map(|(i, r)| {
                vec![
                    i as u8 + 10;
                    if r.bits == 4 {
                        r.int4_bytes as usize
                    } else {
                        r.int8_bytes as usize
                    }
                ]
            })
            .collect();
        let payload_sha = sha256_hex(&payload);
        let mut header_source = source.clone();
        header_source.budget.baseline_int4_routed_bytes = payload.len() as u64;
        let h = compact_header(&header_source, &payload_sha).unwrap();
        let bank = [h.clone(), payload].concat();
        source.compact_cache = Some(MixedCompactCacheIdentity {
            basename: "source.krmx".into(),
            bytes: bank.len() as u64,
            sha256: sha256_hex(&bank),
            header_sha256: sha256_hex(&h),
            payload_sha256: payload_sha,
            region_map_sha256: region_map_sha256(&source.regions).unwrap(),
        });
        let mut original_header = Vec::from(*b"KRAS");
        original_header.extend(7u32.to_le_bytes());
        for n in [
            source.hidden_size as u64,
            source.intermediate_size as u64,
            source.routed_experts_per_layer as u64,
            source.routed_layers as u64,
            source.group_size as u64,
            0,
            0,
        ] {
            original_header.extend(n.to_le_bytes());
        }
        let original = [original_header.clone(), vec![99; 64]].concat();
        let original_id = MixedCacheSourceIdentity {
            bits: 4,
            basename: "original.bin".into(),
            bytes: original.len() as u64,
            sha256: sha256_hex(&original),
            header_sha256: sha256_hex(&original_header),
            header_config_fnv1a: 0,
            expert_int4_calibration_mode: "amax".into(),
        };
        let raw = serde_json::to_vec(&source).unwrap();
        new_write(&root.join("source.manifest.json"), &raw).unwrap();
        new_write(&root.join("source.krmx"), &bank).unwrap();
        new_write(&root.join("mixed-expert-ranking.json"), &ranking).unwrap();
        new_write(&root.join("original.bin"), &original).unwrap();
        new_write(&root.join("coverage.json"), &coverage).unwrap();
        new_write(&root.join("protocol.md"), b"frozen protocol").unwrap();
        let path_string = |name| root.join(name).to_str().unwrap().to_string();
        let plan = Plan {
            source_reserve_path: path_string("source.manifest.json"),
            source_reserve_sha256: sha256_hex(&raw),
            original_amax_path: path_string("original.bin"),
            original_amax: original_id,
            coverage_path: path_string("coverage.json"),
            coverage_sha256: sha256_hex(&coverage),
            protocol_path: path_string("protocol.md"),
            protocol_sha256: sha256_hex(b"frozen protocol"),
            expected_replacements: 1,
        };
        new_write(&root.join("plan.json"), &serde_json::to_vec(&plan).unwrap()).unwrap();
        let out = root.join("output");
        run("assemble", &root.join("plan.json"), &out).unwrap();
        run("verify", &root.join("plan.json"), &out).unwrap();
        let bytes = fs::read(out.join("scales-only.krmx")).unwrap();
        assert_eq!(&bytes[128..144], &[10; 16]);
        assert_eq!(&bytes[144..160], &[99; 16]);
        assert_eq!(&bytes[160..176], &[12; 16]);
        assert_eq!(&bytes[176..192], &[13; 16]);
        let manifest_raw = fs::read(out.join("scales-only.manifest.json")).unwrap();
        let mut forged = parse_canonical_manifest(&manifest_raw).unwrap();
        forged.compact_cache.as_mut().unwrap().header_sha256 = "00".repeat(32);
        fs::write(
            out.join("scales-only.manifest.json"),
            serde_json::to_vec(&forged).unwrap(),
        )
        .unwrap();
        assert!(
            run("verify", &root.join("plan.json"), &out).is_err(),
            "corrupted output header SHA must fail built verification"
        );
        fs::write(out.join("scales-only.manifest.json"), manifest_raw).unwrap();
        assert!(run("assemble", &root.join("plan.json"), &out).is_err());
        let mut damaged = bytes.clone();
        damaged[144] ^= 1;
        fs::write(out.join("scales-only.krmx"), damaged).unwrap();
        assert!(run("verify", &root.join("plan.json"), &out).is_err());
        fs::write(out.join("scales-only.krmx"), bytes).unwrap();
        fs::write(root.join("coverage.json"), b"{}").unwrap();
        assert!(run("verify", &root.join("plan.json"), &out).is_err());
        fs::write(root.join("coverage.json"), coverage).unwrap();
        let mut changed = original;
        changed[64] ^= 1;
        fs::write(root.join("original.bin"), changed).unwrap();
        assert!(run("verify", &root.join("plan.json"), &out).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
