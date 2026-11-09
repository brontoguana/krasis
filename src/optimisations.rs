//! Compile-time feature registry. Adopted entries default on; rebuild to bisect.
//! Names, defaults, arithmetic class, descriptions and dependencies live here.
macro_rules! registry {
    ($($name:ident, $default:literal, $kind:literal, $description:literal, $requires:expr;)*) => {
        $(#[doc = $description]
        pub const $name: bool = match option_env!(concat!("KRASIS_OPT_", stringify!($name))) {
            Some(value) => value.as_bytes()[0] == b'1', None => $default,
        };)*
        pub const ENTRIES: &[(&str, bool, &str, &str, &[&str])] = &[
            $((stringify!($name), $default, $kind, $description, $requires),)*
        ];
    }
}
registry! {
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
