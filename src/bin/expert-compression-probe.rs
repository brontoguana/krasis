//! Phase-0 lossless-compression and peer-capacity measurement on real Marlin
//! expert cache bytes.  This command is intentionally offline: it mmaps the
//! canonical cache, samples real experts, and performs no model execution.

use libloading::Library;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::fs::File;
use std::path::{Path, PathBuf};

const CACHE_HEADER_BYTES: usize = 64;
const MARLIN_CACHE_VERSION: u32 = 7;

type ProbeResult<T> = Result<T, String>;

#[derive(Debug)]
struct Args {
    cache: PathBuf,
    model_config: PathBuf,
    heatmap: PathBuf,
    sample_count: usize,
    primary_residents: usize,
    peer_free_bytes: u64,
    safety_bytes: u64,
    recorded_cold_routes_per_token: f64,
    measured_pcie_bytes_per_second: f64,
    zstd_level: i32,
    output: Option<PathBuf>,
}

#[derive(Deserialize)]
struct ModelConfig {
    #[serde(default = "default_experts_gated")]
    experts_gated: bool,
}

fn default_experts_gated() -> bool {
    true
}

#[derive(Debug, Serialize)]
struct Header {
    version: u32,
    hidden_size: usize,
    intermediate_size: usize,
    experts_per_layer: usize,
    moe_layers: usize,
    group_size: usize,
    config_hash: u64,
    shared_experts: usize,
    calibration_tag: u32,
}

#[derive(Debug)]
struct Layout {
    w13_packed: usize,
    w13_scales: usize,
    w2_packed: usize,
    w2_scales: usize,
    expert_bytes: usize,
}

#[derive(Clone, Debug)]
struct Route {
    relative_layer: usize,
    absolute_layer: usize,
    expert: usize,
    count: u64,
}

#[derive(Default)]
struct Accumulator {
    original: u64,
    compressed: u64,
    minimum_ratio: f64,
    maximum_ratio: f64,
    ratios: Vec<f64>,
}

impl Accumulator {
    fn add(&mut self, original: usize, compressed: usize) {
        let ratio = compressed as f64 / original as f64;
        if self.ratios.is_empty() {
            self.minimum_ratio = ratio;
            self.maximum_ratio = ratio;
        } else {
            self.minimum_ratio = self.minimum_ratio.min(ratio);
            self.maximum_ratio = self.maximum_ratio.max(ratio);
        }
        self.original += original as u64;
        self.compressed += compressed as u64;
        self.ratios.push(ratio);
    }

    fn report(&self) -> CodecReport {
        let aggregate_ratio = self.compressed as f64 / self.original as f64;
        let mean_ratio = self.ratios.iter().sum::<f64>() / self.ratios.len() as f64;
        CodecReport {
            original_bytes: self.original,
            compressed_bytes: self.compressed,
            aggregate_ratio,
            aggregate_saving_pct: (1.0 - aggregate_ratio) * 100.0,
            mean_expert_ratio: mean_ratio,
            minimum_expert_ratio: self.minimum_ratio,
            maximum_expert_ratio: self.maximum_ratio,
        }
    }
}

#[derive(Serialize)]
struct CodecReport {
    original_bytes: u64,
    compressed_bytes: u64,
    aggregate_ratio: f64,
    aggregate_saving_pct: f64,
    mean_expert_ratio: f64,
    minimum_expert_ratio: f64,
    maximum_expert_ratio: f64,
}

#[derive(Serialize)]
struct EntropyReport {
    aggregate_nibble_bits: f64,
    aggregate_nibble_theoretical_packed_ratio: f64,
    nibble_lane_bits: [f64; 8],
    scale_low_byte_bits: f64,
    scale_high_byte_bits: f64,
}

#[derive(Serialize)]
struct PeerReport {
    primary_resident_experts: usize,
    peer_free_bytes: u64,
    safety_bytes: u64,
    peer_usable_bytes: u64,
    peer_expert_capacity: usize,
    total_heatmap_routes_per_token: f64,
    primary_heatmap_hits_per_token: f64,
    heatmap_cold_routes_before_peer_per_token: f64,
    peer_heatmap_hits_per_token: f64,
    heatmap_cold_routes_after_peer_per_token: f64,
    peer_fraction_of_primary_cold_routes: f64,
    recorded_cold_routes_per_token: f64,
    implied_recorded_routes_caught_per_token: f64,
    implied_bytes_avoided_per_token: f64,
    implied_transfer_ms_avoided_per_token: f64,
}

#[derive(Serialize)]
struct SampleRange {
    sample_count: usize,
    nonzero_cold_population: usize,
    hottest_sample_count: u64,
    coldest_sample_count: u64,
    every_decompression_bit_exact: bool,
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    source_cache: String,
    source_cache_bytes: u64,
    model_config: String,
    heatmap: String,
    heatmap_decode_tokens: u64,
    zstd_version: String,
    zstd_level: i32,
    header: Header,
    experts_gated: bool,
    expert_bytes: usize,
    stream_layout_bytes: [usize; 4],
    sample: SampleRange,
    generic_zstd: CodecReport,
    four_native_stream_zstd: CodecReport,
    format_aware_twenty_stream_zstd: CodecReport,
    entropy: EntropyReport,
    peer: PeerReport,
}

struct Zstd {
    _library: Library,
    compress_bound: unsafe extern "C" fn(usize) -> usize,
    compress: unsafe extern "C" fn(*mut c_void, usize, *const c_void, usize, c_int) -> usize,
    decompress: unsafe extern "C" fn(*mut c_void, usize, *const c_void, usize) -> usize,
    is_error: unsafe extern "C" fn(usize) -> u32,
    error_name: unsafe extern "C" fn(usize) -> *const c_char,
    version_string: unsafe extern "C" fn() -> *const c_char,
}

impl Zstd {
    fn load() -> ProbeResult<Self> {
        let library = unsafe { Library::new("libzstd.so.1") }
            .map_err(|error| format!("failed to load libzstd.so.1: {error}"))?;
        unsafe {
            let compress_bound = *library
                .get::<unsafe extern "C" fn(usize) -> usize>(b"ZSTD_compressBound\0")
                .map_err(|error| error.to_string())?;
            let compress = *library
                .get::<unsafe extern "C" fn(
                    *mut c_void,
                    usize,
                    *const c_void,
                    usize,
                    c_int,
                ) -> usize>(b"ZSTD_compress\0")
                .map_err(|error| error.to_string())?;
            let decompress = *library
                .get::<unsafe extern "C" fn(*mut c_void, usize, *const c_void, usize) -> usize>(
                    b"ZSTD_decompress\0",
                )
                .map_err(|error| error.to_string())?;
            let is_error = *library
                .get::<unsafe extern "C" fn(usize) -> u32>(b"ZSTD_isError\0")
                .map_err(|error| error.to_string())?;
            let error_name = *library
                .get::<unsafe extern "C" fn(usize) -> *const c_char>(b"ZSTD_getErrorName\0")
                .map_err(|error| error.to_string())?;
            let version_string = *library
                .get::<unsafe extern "C" fn() -> *const c_char>(b"ZSTD_versionString\0")
                .map_err(|error| error.to_string())?;
            Ok(Self {
                _library: library,
                compress_bound,
                compress,
                decompress,
                is_error,
                error_name,
                version_string,
            })
        }
    }

    fn version(&self) -> String {
        unsafe { CStr::from_ptr((self.version_string)()) }
            .to_string_lossy()
            .into_owned()
    }

    fn round_trip(&self, source: &[u8], level: i32) -> ProbeResult<usize> {
        let bound = unsafe { (self.compress_bound)(source.len()) };
        let mut compressed = vec![0_u8; bound];
        let compressed_bytes = unsafe {
            (self.compress)(
                compressed.as_mut_ptr().cast(),
                compressed.len(),
                source.as_ptr().cast(),
                source.len(),
                level,
            )
        };
        self.check(compressed_bytes, "ZSTD_compress")?;
        compressed.truncate(compressed_bytes);
        let mut restored = vec![0_u8; source.len()];
        let restored_bytes = unsafe {
            (self.decompress)(
                restored.as_mut_ptr().cast(),
                restored.len(),
                compressed.as_ptr().cast(),
                compressed.len(),
            )
        };
        self.check(restored_bytes, "ZSTD_decompress")?;
        if restored_bytes != source.len() || restored != source {
            return Err("Zstd round trip was not bit-exact".to_string());
        }
        Ok(compressed_bytes)
    }

    fn check(&self, code: usize, operation: &str) -> ProbeResult<()> {
        if unsafe { (self.is_error)(code) } == 0 {
            return Ok(());
        }
        let reason = unsafe { CStr::from_ptr((self.error_name)(code)) }.to_string_lossy();
        Err(format!("{operation} failed: {reason}"))
    }
}

struct Histograms {
    all_nibbles: [u64; 16],
    nibble_lanes: [[u64; 16]; 8],
    scale_low: [u64; 256],
    scale_high: [u64; 256],
}

impl Default for Histograms {
    fn default() -> Self {
        Self {
            all_nibbles: [0; 16],
            nibble_lanes: [[0; 16]; 8],
            scale_low: [0; 256],
            scale_high: [0; 256],
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ERROR: {error}");
        std::process::exit(1);
    }
}

fn run() -> ProbeResult<()> {
    let args = parse_args()?;
    let config: ModelConfig = serde_json::from_reader(
        File::open(&args.model_config)
            .map_err(|error| format!("failed to open {}: {error}", args.model_config.display()))?,
    )
    .map_err(|error| format!("failed to parse model config: {error}"))?;
    let cache_file = File::open(&args.cache)
        .map_err(|error| format!("failed to open {}: {error}", args.cache.display()))?;
    let source_cache_bytes = cache_file
        .metadata()
        .map_err(|error| error.to_string())?
        .len();
    let mmap = unsafe { Mmap::map(&cache_file) }.map_err(|error| error.to_string())?;
    let header = read_header(&mmap)?;
    let layout = marlin_layout(&header, config.experts_gated)?;
    let routed_bytes = header
        .moe_layers
        .checked_mul(header.experts_per_layer)
        .and_then(|count| count.checked_mul(layout.expert_bytes))
        .and_then(|bytes| CACHE_HEADER_BYTES.checked_add(bytes))
        .ok_or_else(|| "routed cache byte range overflow".to_string())?;
    if routed_bytes > mmap.len() {
        return Err(format!(
            "routed expert range {routed_bytes} exceeds cache size {}",
            mmap.len()
        ));
    }

    let (mut routes, decode_tokens) = load_routes(&args.heatmap, &header)?;
    routes.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.relative_layer.cmp(&right.relative_layer))
            .then_with(|| left.expert.cmp(&right.expert))
    });
    if args.primary_residents >= routes.len() {
        return Err("--primary-residents leaves no cold expert sample population".to_string());
    }
    let nonzero_end = routes.partition_point(|route| route.count > 0);
    if nonzero_end <= args.primary_residents {
        return Err("heatmap has no observed cold-route population".to_string());
    }
    let sample_indices = stratified_indices(
        args.primary_residents,
        nonzero_end,
        args.sample_count.min(nonzero_end - args.primary_residents),
    );
    let zstd = Zstd::load()?;
    eprintln!(
        "Sampling {} real experts ({} bytes each) with Zstd {} level {}",
        sample_indices.len(),
        layout.expert_bytes,
        zstd.version(),
        args.zstd_level
    );

    let mut generic = Accumulator::default();
    let mut native_streams = Accumulator::default();
    let mut format_aware = Accumulator::default();
    let mut histograms = Histograms::default();
    for (sample_number, rank) in sample_indices.iter().copied().enumerate() {
        let route = &routes[rank];
        let expert = expert_bytes(&mmap, &header, &layout, route)?;
        let streams = native_streams_for_expert(expert, &layout);

        let generic_bytes = zstd.round_trip(expert, args.zstd_level)?;
        generic.add(expert.len(), generic_bytes);

        let mut split_bytes = streams.len() * std::mem::size_of::<u32>();
        for stream in &streams {
            split_bytes += zstd.round_trip(stream, args.zstd_level)?;
        }
        native_streams.add(expert.len(), split_bytes);

        let transformed = format_aware_streams(&streams, &mut histograms)?;
        let mut transformed_bytes = transformed.len() * std::mem::size_of::<u32>();
        for stream in &transformed {
            transformed_bytes += zstd.round_trip(stream, args.zstd_level)?;
        }
        let restored = reverse_format_aware(&transformed, &layout)?;
        if restored != expert {
            return Err(format!(
                "format-aware transform was not bit-exact for layer {} expert {}",
                route.absolute_layer, route.expert
            ));
        }
        format_aware.add(expert.len(), transformed_bytes);

        if (sample_number + 1) % 25 == 0 || sample_number + 1 == sample_indices.len() {
            eprintln!(
                "  verified {}/{} experts",
                sample_number + 1,
                sample_indices.len()
            );
        }
    }

    let peer = peer_report(&args, &routes, decode_tokens, layout.expert_bytes)?;
    let entropy = EntropyReport {
        aggregate_nibble_bits: entropy(&histograms.all_nibbles),
        aggregate_nibble_theoretical_packed_ratio: entropy(&histograms.all_nibbles) / 4.0,
        nibble_lane_bits: std::array::from_fn(|lane| entropy(&histograms.nibble_lanes[lane])),
        scale_low_byte_bits: entropy(&histograms.scale_low),
        scale_high_byte_bits: entropy(&histograms.scale_high),
    };
    let hottest_sample_count = routes[*sample_indices.first().unwrap()].count;
    let coldest_sample_count = routes[*sample_indices.last().unwrap()].count;
    let report = Report {
        schema_version: 1,
        source_cache: args.cache.display().to_string(),
        source_cache_bytes,
        model_config: args.model_config.display().to_string(),
        heatmap: args.heatmap.display().to_string(),
        heatmap_decode_tokens: decode_tokens,
        zstd_version: zstd.version(),
        zstd_level: args.zstd_level,
        stream_layout_bytes: [
            layout.w13_packed,
            layout.w13_scales,
            layout.w2_packed,
            layout.w2_scales,
        ],
        expert_bytes: layout.expert_bytes,
        experts_gated: config.experts_gated,
        header,
        sample: SampleRange {
            sample_count: sample_indices.len(),
            nonzero_cold_population: nonzero_end - args.primary_residents,
            hottest_sample_count,
            coldest_sample_count,
            every_decompression_bit_exact: true,
        },
        generic_zstd: generic.report(),
        four_native_stream_zstd: native_streams.report(),
        format_aware_twenty_stream_zstd: format_aware.report(),
        entropy,
        peer,
    };
    let json = serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?;
    println!("{json}");
    if let Some(path) = &args.output {
        std::fs::write(path, format!("{json}\n"))
            .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    }
    Ok(())
}

fn parse_args() -> ProbeResult<Args> {
    let mut raw = std::env::args().skip(1);
    let cache = required_path(raw.next(), "cache path")?;
    let model_config = required_path(raw.next(), "model config path")?;
    let heatmap = required_path(raw.next(), "heatmap path")?;
    let mut args = Args {
        cache,
        model_config,
        heatmap,
        sample_count: 300,
        primary_residents: 0,
        peer_free_bytes: 0,
        safety_bytes: 0,
        recorded_cold_routes_per_token: 0.0,
        measured_pcie_bytes_per_second: 0.0,
        zstd_level: 1,
        output: None,
    };
    while let Some(flag) = raw.next() {
        match flag.as_str() {
            "--sample-count" => args.sample_count = parse_value(raw.next(), &flag)?,
            "--primary-residents" => args.primary_residents = parse_value(raw.next(), &flag)?,
            "--peer-free-bytes" => args.peer_free_bytes = parse_value(raw.next(), &flag)?,
            "--safety-bytes" => args.safety_bytes = parse_value(raw.next(), &flag)?,
            "--recorded-cold-routes" => {
                args.recorded_cold_routes_per_token = parse_value(raw.next(), &flag)?
            }
            "--pcie-bytes-per-second" => {
                args.measured_pcie_bytes_per_second = parse_value(raw.next(), &flag)?
            }
            "--zstd-level" => args.zstd_level = parse_value(raw.next(), &flag)?,
            "--output" => args.output = Some(required_path(raw.next(), &flag)?),
            "--help" | "-h" => return Err(usage()),
            _ => return Err(format!("unknown argument {flag}\n{}", usage())),
        }
    }
    if args.sample_count == 0
        || args.primary_residents == 0
        || args.peer_free_bytes == 0
        || args.safety_bytes == 0
        || args.recorded_cold_routes_per_token <= 0.0
        || args.measured_pcie_bytes_per_second <= 0.0
    {
        return Err(format!(
            "all measurement inputs must be explicit and non-zero\n{}",
            usage()
        ));
    }
    Ok(args)
}

fn usage() -> String {
    "Usage: ./dev expert-compression-probe <cache.bin> <config.json> <heatmap.json> --primary-residents N --peer-free-bytes N --safety-bytes N --recorded-cold-routes F --pcie-bytes-per-second F [--sample-count N] [--zstd-level N] [--output PATH]".to_string()
}

fn required_path(raw: Option<String>, name: &str) -> ProbeResult<PathBuf> {
    Ok(PathBuf::from(
        raw.ok_or_else(|| format!("missing {name}\n{}", usage()))?,
    ))
}

fn parse_value<T: std::str::FromStr>(raw: Option<String>, name: &str) -> ProbeResult<T>
where
    T::Err: std::fmt::Display,
{
    raw.ok_or_else(|| format!("{name} requires a value"))?
        .parse::<T>()
        .map_err(|error| format!("invalid {name}: {error}"))
}

fn read_header(bytes: &[u8]) -> ProbeResult<Header> {
    if bytes.len() < CACHE_HEADER_BYTES || &bytes[0..4] != b"KRAS" {
        return Err("invalid Marlin cache header".to_string());
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != MARLIN_CACHE_VERSION {
        return Err(format!(
            "Marlin cache version {version}, expected {MARLIN_CACHE_VERSION}"
        ));
    }
    let tail = u64::from_le_bytes(bytes[56..64].try_into().unwrap());
    Ok(Header {
        version,
        hidden_size: read_u64(bytes, 8)?,
        intermediate_size: read_u64(bytes, 16)?,
        experts_per_layer: read_u64(bytes, 24)?,
        moe_layers: read_u64(bytes, 32)?,
        group_size: read_u64(bytes, 40)?,
        config_hash: u64::from_le_bytes(bytes[48..56].try_into().unwrap()),
        shared_experts: (tail & 0xffff_ffff) as usize,
        calibration_tag: (tail >> 32) as u32,
    })
}

fn read_u64(bytes: &[u8], offset: usize) -> ProbeResult<usize> {
    usize::try_from(u64::from_le_bytes(
        bytes[offset..offset + 8].try_into().unwrap(),
    ))
    .map_err(|_| "cache dimension does not fit usize".to_string())
}

fn marlin_layout(header: &Header, gated: bool) -> ProbeResult<Layout> {
    let hidden = header.hidden_size;
    let intermediate = header.intermediate_size;
    if hidden % 8 != 0 || hidden % header.group_size != 0 || intermediate % 8 != 0 {
        return Err("cache dimensions do not satisfy Marlin INT4 packing".to_string());
    }
    let padded_hidden = if hidden == intermediate && hidden % 256 != 0 {
        hidden + 64
    } else {
        hidden
    };
    let w13_width = if gated {
        intermediate * 2
    } else {
        intermediate
    };
    let w13_packed = (hidden / 8) * w13_width * 4;
    let w13_scales = (hidden / header.group_size) * w13_width * 2;
    let w2_packed = (intermediate / 8) * padded_hidden * 4;
    let w2_scales = intermediate.div_ceil(header.group_size) * padded_hidden * 2;
    Ok(Layout {
        w13_packed,
        w13_scales,
        w2_packed,
        w2_scales,
        expert_bytes: w13_packed + w13_scales + w2_packed + w2_scales,
    })
}

fn load_routes(path: &Path, header: &Header) -> ProbeResult<(Vec<Route>, u64)> {
    let root: Value = serde_json::from_reader(
        File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("failed to parse heatmap: {error}"))?;
    let object = root
        .as_object()
        .ok_or_else(|| "heatmap root is not an object".to_string())?;
    let decode_tokens = root
        .pointer("/_metadata/heatmap_build/total_decode_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| "heatmap has no total_decode_tokens".to_string())?;
    let mut counts = HashMap::<(usize, usize), u64>::new();
    let mut min_layer = usize::MAX;
    let mut max_layer = 0;
    for (key, value) in object {
        if key.starts_with('_') {
            continue;
        }
        let (layer, expert) = key
            .split_once(',')
            .ok_or_else(|| format!("invalid heatmap route key {key}"))?;
        let layer = layer
            .parse::<usize>()
            .map_err(|error| format!("invalid heatmap layer {layer}: {error}"))?;
        let expert = expert
            .parse::<usize>()
            .map_err(|error| format!("invalid heatmap expert {expert}: {error}"))?;
        if expert >= header.experts_per_layer {
            return Err(format!("heatmap expert {expert} exceeds cache geometry"));
        }
        let count = value
            .as_u64()
            .ok_or_else(|| format!("heatmap count for {key} is not a u64"))?;
        counts.insert((layer, expert), count);
        min_layer = min_layer.min(layer);
        max_layer = max_layer.max(layer);
    }
    if min_layer == usize::MAX || max_layer - min_layer + 1 != header.moe_layers {
        return Err(format!(
            "heatmap layer range {min_layer}..={max_layer} does not match {} cache layers",
            header.moe_layers
        ));
    }
    let mut routes = Vec::with_capacity(header.moe_layers * header.experts_per_layer);
    for relative_layer in 0..header.moe_layers {
        let absolute_layer = min_layer + relative_layer;
        for expert in 0..header.experts_per_layer {
            routes.push(Route {
                relative_layer,
                absolute_layer,
                expert,
                count: counts.get(&(absolute_layer, expert)).copied().unwrap_or(0),
            });
        }
    }
    Ok((routes, decode_tokens))
}

fn stratified_indices(start: usize, end: usize, samples: usize) -> Vec<usize> {
    (0..samples)
        .map(|index| start + ((2 * index + 1) * (end - start)) / (2 * samples))
        .collect()
}

fn expert_bytes<'a>(
    mmap: &'a [u8],
    header: &Header,
    layout: &Layout,
    route: &Route,
) -> ProbeResult<&'a [u8]> {
    let ordinal = route
        .relative_layer
        .checked_mul(header.experts_per_layer)
        .and_then(|value| value.checked_add(route.expert))
        .ok_or_else(|| "expert ordinal overflow".to_string())?;
    let start = CACHE_HEADER_BYTES
        .checked_add(ordinal.saturating_mul(layout.expert_bytes))
        .ok_or_else(|| "expert offset overflow".to_string())?;
    let end = start + layout.expert_bytes;
    mmap.get(start..end).ok_or_else(|| {
        format!(
            "expert byte range missing for layer {} expert {}",
            route.absolute_layer, route.expert
        )
    })
}

fn native_streams_for_expert<'a>(expert: &'a [u8], layout: &Layout) -> Vec<&'a [u8]> {
    let w13s = layout.w13_packed;
    let w2p = w13s + layout.w13_scales;
    let w2s = w2p + layout.w2_packed;
    vec![
        &expert[..w13s],
        &expert[w13s..w2p],
        &expert[w2p..w2s],
        &expert[w2s..],
    ]
}

fn format_aware_streams(
    streams: &[&[u8]],
    histograms: &mut Histograms,
) -> ProbeResult<Vec<Vec<u8>>> {
    if streams.len() != 4 {
        return Err("expected four native Marlin streams".to_string());
    }
    let mut transformed = Vec::with_capacity(20);
    for matrix in 0..2 {
        let packed = streams[matrix * 2];
        let scales = streams[matrix * 2 + 1];
        let lanes = deinterleave_nibbles(packed, histograms)?;
        transformed.extend(lanes);
        let (low, high) = split_bf16_bytes(scales, histograms)?;
        transformed.push(low);
        transformed.push(high);
    }
    Ok(transformed)
}

fn deinterleave_nibbles(packed: &[u8], histograms: &mut Histograms) -> ProbeResult<Vec<Vec<u8>>> {
    if packed.len() % 4 != 0 || (packed.len() / 4) % 2 != 0 {
        return Err("packed Marlin stream cannot be evenly de-interleaved".to_string());
    }
    let words = packed.len() / 4;
    let mut lanes: Vec<Vec<u8>> = (0..8).map(|_| Vec::with_capacity(words / 2)).collect();
    for lane in 0..8 {
        let mut pending = None;
        for word_index in 0..words {
            let word_offset = word_index * 4;
            let word = u32::from_le_bytes(packed[word_offset..word_offset + 4].try_into().unwrap());
            let nibble = ((word >> (lane * 4)) & 0x0f) as u8;
            histograms.all_nibbles[nibble as usize] += 1;
            histograms.nibble_lanes[lane][nibble as usize] += 1;
            if let Some(low) = pending.take() {
                lanes[lane].push(low | (nibble << 4));
            } else {
                pending = Some(nibble);
            }
        }
        debug_assert!(pending.is_none());
    }
    Ok(lanes)
}

fn split_bf16_bytes(scales: &[u8], histograms: &mut Histograms) -> ProbeResult<(Vec<u8>, Vec<u8>)> {
    if scales.len() % 2 != 0 {
        return Err("BF16 scale stream has an odd byte count".to_string());
    }
    let mut low = Vec::with_capacity(scales.len() / 2);
    let mut high = Vec::with_capacity(scales.len() / 2);
    for pair in scales.chunks_exact(2) {
        low.push(pair[0]);
        high.push(pair[1]);
        histograms.scale_low[pair[0] as usize] += 1;
        histograms.scale_high[pair[1] as usize] += 1;
    }
    Ok((low, high))
}

fn reverse_format_aware(streams: &[Vec<u8>], layout: &Layout) -> ProbeResult<Vec<u8>> {
    if streams.len() != 20 {
        return Err(format!(
            "expected 20 format-aware streams, got {}",
            streams.len()
        ));
    }
    let mut restored = Vec::with_capacity(layout.expert_bytes);
    for matrix in 0..2 {
        let base = matrix * 10;
        let packed_bytes = if matrix == 0 {
            layout.w13_packed
        } else {
            layout.w2_packed
        };
        restored.extend(reinterleave_nibbles(
            &streams[base..base + 8],
            packed_bytes,
        )?);
        let low = &streams[base + 8];
        let high = &streams[base + 9];
        if low.len() != high.len() {
            return Err("split BF16 streams have unequal sizes".to_string());
        }
        for index in 0..low.len() {
            restored.push(low[index]);
            restored.push(high[index]);
        }
    }
    Ok(restored)
}

fn reinterleave_nibbles(lanes: &[Vec<u8>], packed_bytes: usize) -> ProbeResult<Vec<u8>> {
    if lanes.len() != 8 || lanes.iter().any(|lane| lane.len() * 8 != packed_bytes) {
        return Err("nibble lane geometry does not match packed stream".to_string());
    }
    let words = packed_bytes / 4;
    let mut packed = vec![0_u8; packed_bytes];
    for word_index in 0..words {
        let mut word = 0_u32;
        for (lane, values) in lanes.iter().enumerate() {
            let byte = values[word_index / 2];
            let nibble = if word_index % 2 == 0 {
                byte & 0x0f
            } else {
                byte >> 4
            };
            word |= (nibble as u32) << (lane * 4);
        }
        packed[word_index * 4..word_index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    Ok(packed)
}

fn entropy<const N: usize>(counts: &[u64; N]) -> f64 {
    let total = counts.iter().sum::<u64>() as f64;
    counts
        .iter()
        .filter(|&&count| count != 0)
        .map(|&count| {
            let probability = count as f64 / total;
            -probability * probability.log2()
        })
        .sum()
}

fn peer_report(
    args: &Args,
    routes: &[Route],
    decode_tokens: u64,
    expert_bytes: usize,
) -> ProbeResult<PeerReport> {
    let peer_usable_bytes = args
        .peer_free_bytes
        .checked_sub(args.safety_bytes)
        .ok_or_else(|| "peer free bytes are below the safety reserve".to_string())?;
    let peer_expert_capacity = usize::try_from(peer_usable_bytes / expert_bytes as u64)
        .map_err(|_| "peer capacity does not fit usize".to_string())?
        .min(routes.len() - args.primary_residents);
    let total = routes.iter().map(|route| route.count).sum::<u64>();
    let primary = routes[..args.primary_residents]
        .iter()
        .map(|route| route.count)
        .sum::<u64>();
    let peer = routes[args.primary_residents..args.primary_residents + peer_expert_capacity]
        .iter()
        .map(|route| route.count)
        .sum::<u64>();
    let cold_before = total - primary;
    let token_scale = decode_tokens as f64;
    let peer_fraction = peer as f64 / cold_before as f64;
    let implied_routes = args.recorded_cold_routes_per_token * peer_fraction;
    let implied_bytes = implied_routes * expert_bytes as f64;
    Ok(PeerReport {
        primary_resident_experts: args.primary_residents,
        peer_free_bytes: args.peer_free_bytes,
        safety_bytes: args.safety_bytes,
        peer_usable_bytes,
        peer_expert_capacity,
        total_heatmap_routes_per_token: total as f64 / token_scale,
        primary_heatmap_hits_per_token: primary as f64 / token_scale,
        heatmap_cold_routes_before_peer_per_token: cold_before as f64 / token_scale,
        peer_heatmap_hits_per_token: peer as f64 / token_scale,
        heatmap_cold_routes_after_peer_per_token: (cold_before - peer) as f64 / token_scale,
        peer_fraction_of_primary_cold_routes: peer_fraction,
        recorded_cold_routes_per_token: args.recorded_cold_routes_per_token,
        implied_recorded_routes_caught_per_token: implied_routes,
        implied_bytes_avoided_per_token: implied_bytes,
        implied_transfer_ms_avoided_per_token: implied_bytes / args.measured_pcie_bytes_per_second
            * 1_000.0,
    })
}
