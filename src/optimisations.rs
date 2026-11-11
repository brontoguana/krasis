//! Compile-time feature registry. Adopted entries default on; rebuild to bisect.
//! Names, defaults, arithmetic class, descriptions and dependencies live here.
macro_rules! registry {
    ($($name:ident, $default:literal, $kind:literal, $description:literal, $requires:expr;)*) => {
        $(#[doc = $description]
        pub const $name: bool = match option_env!(concat!("KRASIS_OPT_", stringify!($name))) {
            Some(value) => value.as_bytes()[0] == b'1', None => $default,
        };)*
        /// Per-engine selection. Non-test builds retain constant dispatch and zero storage.
        #[derive(Clone, Copy, Debug)]
        #[allow(non_snake_case)]
        pub struct Selection { $(#[cfg(feature = "live-switches")] pub $name: bool,)* }
        impl Selection {
            pub const fn compiled() -> Self { Self { $(#[cfg(feature = "live-switches")] $name: $name,)* } }
            $(#[inline(always)] #[allow(non_snake_case)]
            pub fn $name(&self) -> bool {
                #[cfg(feature = "live-switches")] { self.$name }
                #[cfg(not(feature = "live-switches"))] { $name }
            })*
            #[cfg(feature = "live-switches")]
            pub fn set(&mut self, name: &str, enabled: bool) -> Result<(), String> {
                match name { $(stringify!($name) => self.$name = enabled,)*
                    _ => return Err(format!("unknown optimisation: {name}")), }
                Ok(())
            }
            #[cfg(feature = "live-switches")]
            pub fn values(&self) -> Vec<(&'static str, bool)> { vec![$((stringify!($name), self.$name),)*] }
        }
        pub const ENTRIES: &[(&str, bool, &str, &str, &[&str])] = &[
            $((stringify!($name), $default, $kind, $description, $requires),)*
        ];
    }
}
registry! {
    SHARED_EXPERT_COPY_OVERLAP, false, "exact", "Overlap shared expert projections with demand transfers", &[];
    COLD_EXPERT_PRUNING_FIVE, true, "approximation", "Protect five of eight routes with a 25 percent cold-mass ceiling", &["COLD_EXPERT_PRUNING_20"];
    STREAMED_COLD_NF3, true, "approximation", "Layer-fitted three-bit codebooks for streamed cold experts", &[];
    STREAMED_COLD_INT3, false, "approximation", "Three-bit streamed cold weights with unchanged resident and CPU weights", &[];
    CPU_FULL_QUEUE_PLANNER, true, "approximation", "Calibrate complete CPU and GPU cold queues for partition selection", &["CPU_COLD_LANE"];
    CPU_VECTOR_SCALES, true, "exact", "Vector scale FMA for 1024-output CPU reader bands", &["CPU_MARLIN_READER"];
    CPU_K_MAJOR_READER, true, "exact", "Contiguous packed K traversal for 1024-output AVX2 bands", &["CPU_MARLIN_READER"];
    COLD_EXPERT_PRUNING_20, true, "approximation", "Protect six of eight routes with a 20 percent cold-mass ceiling", &["COLD_EXPERT_PRUNING_15"];
    CPU_CACHE_LOCAL_TEAMS, true, "approximation", "Group expert workers within discovered last-level cache domains", &["CPU_COLD_LANE"];
    COLD_EXPERT_PRUNING_15, true, "approximation", "Protect six of eight routes with a 15 percent cold-mass ceiling", &["COLD_EXPERT_PRUNING"];
    CPU_MORE_WORKERS, true, "approximation", "Calibrated four-expert CPU partition with two-worker fallback", &[];
    CPU_MARLIN_READER, true, "exact", "Group-banded paired-row AVX2 Marlin reader", &[];
    CPU_COLD_LANE, true, "approximation", "Planned direct-host CPU cold expert lane", &[];
    COLD_EXPERT_PRUNING, true, "approximation", "GLM-5.3 host-route 75/10 cold-expert pruning", &[];
    EXACT_ATTENTION, true, "exact", "K6 exact scores and shared unpacked values", &[];
    ORDERED_ABSORPTION, true, "exact", "Ordered sparse MLA query absorption", &[];
    PREFILL_F32, true, "exact", "Prepared FP32 prefill attention", &[];
    INDEXER_QUERY_TILE, true, "exact", "Fixed row indexer query projection tiles", &[];
    INDEX_SCORE_ROW_BANDS, true, "exact", "Causal index score row bands", &[];
    INDEX_RADIX_TOPK, true, "exact", "Radix index selection", &[];
    VALUE_REASSOCIATION, false, "approximation", "Reassociate prefill value contraction", &["PREFILL_F32"];
    TENSOR_DECODE_ATTENTION, false, "approximation", "Tensor sparse decode attention", &[];
    TENSOR_ABSORPTION, false, "approximation", "Tensor sparse query absorption", &[];
    KDA_RELAY, true, "exact", "Pipelined two-warp prepared GLM-5.3 KDA recurrence", &[];
}
/// Stable manifest copied into receipts and printed at process startup.
pub fn manifest() -> &'static str {
    option_env!("KRASIS_OPT_MANIFEST").unwrap_or("unavailable during build-script compilation")
}
pub fn print_startup() {
    eprintln!("KRASIS_OPTIMISATIONS={}", manifest());
    if let Some(directory) = std::env::var_os("KRASIS_RUN_DIR") {
        let path = std::path::PathBuf::from(directory).join("optimisations.json");
        if let Err(error) = std::fs::write(&path, manifest()) {
            eprintln!("Cannot record optimisation set at {}: {error}", path.display());
        }
    }
}
