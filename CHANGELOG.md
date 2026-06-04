# Changelog

## Unreleased

- Improved measured-safe Q122B prefill chunk sizing. After startup calibration
  measures the post-scratch runtime low-water delta, prefill scratch planning
  now uses that measured reserve directly instead of adding a second full
  cold-staging reserve on top of it. Before calibration has measured runtime
  overhead, Krasis still falls back to the full cold-staging reserve. This keeps
  the `600 MB` safety margin and automatic HCS pressure behavior intact while
  allowing larger measured-safe chunks.
- Validation: `./dev build` passed. Instrumented Q122B HQQ6/k4v4 diagnostic
  produced `3401.7 tok/s` internal prefill, `28.31 tok/s` internal decode, and
  `48.47 tok/s` HTTP with `776 MB` minimum free VRAM and a clean health scan.
  Clean timing-off `./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf`
  produced `3608.7 tok/s` internal prefill, `27.76 tok/s` internal decode, and
  `47.99 tok/s` HTTP with `4050/12288` HCS, `806 MB` minimum free VRAM, and no
  CUDA errors, VRAM monitor warnings, hard-floor exits, or HCS copy failures.
- Improved Q122B HQQ6+k4v4 prefill planning without weakening the VRAM safety
  margin. Runtime prefill now builds an explicit chunk plan, preserves the
  fewest measured-safe passes, and smooths pathological tiny tails across the
  same pass count. Gate pre-scan and dense pointer-table prefetch now consume
  actual chunk boundaries.
- Added dense-active optional-pinning policy for prefill. When pre-scan shows a
  chunk is near-all-experts, Krasis skips the temporary optional pinning pool
  and uses the dense pointer-table/cold path instead of paying pinning overhead
  for experts that are effectively all active anyway.
- Tightened post-scratch prefill allocation acceptance: scratch is accepted
  only when actual post-allocation free VRAM still covers the measured runtime
  transient floor plus cold-staging reserve, not just the raw safety margin.
- Fixed HCS soft-reload safety after prefill. Reload now sizes chunks by the
  physical soft-tier chunk boundary and leaves one measured soft-HCS chunk of
  headroom above the idle floor, preventing reload from grazing below the
  `600 MB` safety margin before pressure eviction reacts.
- Fixed a test-only Polar4 kernel smoke compile error exposed by
  `./dev test-kernels`, where the append-kernel parameter list referenced an
  undeclared `norm_correction_i32`.
- Validation: `./dev build` passed. Clean timing-off Q122B HQQ6/k4v4 benchmark
  produced `3238.3 tok/s` internal prefill, `27.21 tok/s` internal decode, and
  `47.04 tok/s` HTTP round trip with `4050/12288` HCS, `804 MB` minimum free
  VRAM, and clean CUDA/VRAM/HCS health scan. `PATH="$HOME/.cargo/bin:$PATH"
  ./dev test-kernels polar4_kv_roundtrip_smoke` passed.
- Restored the prefill-pass planner fixes from the shelved Q122B investigation:
  Krasis now evicts additional soft HCS when measured VRAM says that can avoid
  an extra prefill pass, uses post-scratch HCS eviction at chunk boundaries,
  chunks as max-safe plus tail instead of equalizing chunks, and removes a
  double-counted HCS reserve from optional prefill pinning budget calculation.
- Validation: `./dev build` passed, and `./dev speed-test` on QCN HQQ4/k4v4
  produced `5054.2 tok/s` internal prefill, `85.34 tok/s` internal decode, and
  `117.52 tok/s` HTTP round trip with `16443/24576` HCS, `672 MB` minimum free
  VRAM, and clean CUDA/VRAM/HCS health scan.

## 1.0.14 - 2026-06-04

- Fixed VRAM pressure handling after a Typhon Qwen3.6-35B dynamic-HCS failure
  where cleanup-time lows reached `322 MB` free against a `600 MB` safety
  margin before CUDA later reported `ILLEGAL_ADDRESS`. Pending pressure now
  retains the worst observed low until HCS drain reacts, chat requests drain
  HCS pressure at cleanup end, and startup force-drains again immediately
  before publishing server-ready.
- Fixed `./dev release-test` model alias resolution so documented aliases such
  as `QCN` resolve to the real model directory before invoking the guarded
  release-test runner.

- Scoped `./dev benchmark` GPU cleanup to `CFG_SELECTED_GPUS` so benchmarks on one GPU do not stop unrelated Krasis services on other GPUs.

- Deprecated and disabled direct FP8 KV cache modes (`fp8`, `fp8_e4m3`).
  Configs or CLI args that request them now fail explicitly; use `k6v6`,
  `k4v4`, or `bf16` instead.
- Added `--ssh-key-path` / `CFG_SSH_KEY_PATH` for reverse SSH tunnel sharing,
  so managed tunnels can force a specific identity file with `IdentitiesOnly`.

## 1.0.13 - 2026-05-30

- Stable release of the Witsy/Qwen vision compatibility and dynamic HCS safety
  fixes from the `1.0.13` prerelease line.
- When a loaded model supports Qwen image inputs, `/models` now advertises only
  the Witsy/OpenAI-compatible `-vision` model id instead of listing separate
  text and vision entries. Use the same `-vision` id for both text and image
  requests; text-only requests still follow the normal text path.
- Chat completions now accept OpenAI's `max_completion_tokens` field as an
  alias for `max_tokens`, matching current OpenAI SDK clients such as Witsy.
- The Hugging Face downloader now includes `preprocessor_config.json` and
  `processor_config.json`, which are required for Qwen image preprocessing.
- Pillow is now declared as a package dependency because the Qwen image
  preprocessing path requires `PIL` in clean installs and containers.
- Quieted transient socket read timeouts / `EAGAIN` (`os error 11`) during HTTP
  request parsing; incomplete probe connections are ignored without printing a
  scary error, while malformed requests still return `400`.
- Fixed source-mode dynamic HCS VRAM-pressure eviction synchronization. When
  Krasis trims soft HCS chunks after a below-safety VRAM event, it now
  synchronizes the CUDA context before freeing source-backed dynamic HCS chunks,
  matching the decode/prefill boundary safety used by normal prefill eviction.
- Fixed dynamic HCS promotion ordering during graph decode. Promoted expert
  slot contents and the GPU expert pointer table are now updated on the same
  CUDA replay stream, preventing later graph/prefill work from seeing stale or
  incompletely ordered promoted pointers after image-heavy requests on tight
  VRAM systems.

## 1.0.13-rc.2 - 2026-05-29

- When a loaded model supports Qwen image inputs, `/models` now advertises only
  the Witsy/OpenAI-compatible `-vision` model id instead of listing separate
  text and vision entries. Use the same `-vision` id for both text and image
  requests; text-only requests still follow the normal text path.
- Chat completions now accept OpenAI's `max_completion_tokens` field as an
  alias for `max_tokens`, matching current OpenAI SDK clients such as Witsy.
- The Hugging Face downloader now includes `preprocessor_config.json` and
  `processor_config.json`, which are required for Qwen image preprocessing.
- Pillow is now declared as a package dependency because the Qwen image
  preprocessing path requires `PIL` in clean installs and containers.
- Quieted transient socket read timeouts / `EAGAIN` (`os error 11`) during HTTP
  request parsing; incomplete probe connections are ignored without printing a
  scary error, while malformed requests still return `400`.
- Fixed dynamic HCS promotion ordering during graph decode. Promoted expert
  slot contents and the GPU expert pointer table are now updated on the same
  CUDA replay stream, preventing later graph/prefill work from seeing stale or
  incompletely ordered promoted pointers after image-heavy requests on tight
  VRAM systems.

## 1.0.13-rc.1 - 2026-05-29

- Fixed source-mode dynamic HCS VRAM-pressure eviction synchronization. When
  Krasis trims soft HCS chunks after a below-safety VRAM event, it now
  synchronizes the CUDA context before freeing source-backed dynamic HCS chunks,
  matching the decode/prefill boundary safety used by normal prefill eviction.
  This prevents latent async illegal-address failures from surfacing on the
  next request after pressure eviction.

## 1.0.12 - 2026-05-29

- Improved OpenAI-compatible model discovery for clients such as Witsy. The
  model-list endpoint now accepts `/models` as well as `/v1/models`, tolerates
  trailing slashes/query strings, and includes the standard `created` field in
  the returned model object.
- Also accepts root-base OpenAI chat paths (`/chat/completions`) in addition to
  `/v1/chat/completions`, so clients work whether their base URL is configured
  as `http://host:port` or `http://host:port/v1`.
- The server-ready banner now prints client setup details for OpenAI-compatible
  apps: base URL, chat endpoint, models endpoint, API key value, and model name.

## 1.0.11 - 2026-05-29

- Added experimental Qwen image support. Image chat requests now lazily run the
  Qwen3VL BF16 vision tower for image-prefill setup, scatter visual embeddings
  into `<|image_pad|>` positions, use MRoPE rows during prefill, and preserve
  the existing text-only Rust/CUDA path for normal requests.
- Vision staging is transient: the BF16 tower is moved to GPU for image setup
  and released back to CPU before decode. Constrained image requests now return
  an explicit insufficient-VRAM response instead of a generic server failure.
- Added image request validation for OpenAI-style content parts. Video remains
  unsupported, and local image file paths are disabled unless explicitly enabled
  for local testing.
- Recorded final QCN speed-test gates and 35B/122B image smoke validation in
  the benchmark archive.
- Updated the QCN release-test reference path to use the llama-witness artifact
  instead of the archived HF reference.
- Fixed reference validation for first-token witness artifacts so release tests
  validate the captured prefix contract instead of requiring uncaptured tokens.

## 1.0.5 - 2026-05-23

- Added request prefill progress output alongside the existing decode progress
  lines. Chat requests now print real prefill token count, elapsed time, tok/s,
  current VRAM, and min-free VRAM during prefill before decode starts.

## 1.0.4 - 2026-05-23

- Improved startup long VRAM calibration probe selection. The adaptive chooser
  now combines Rust prefill scratch growth with compact-KV stage-exact staging
  cost from the loaded model dimensions, then jumps directly to a predicted
  safe long probe with a validation reserve instead of stepping upward through
  small probes.

## 1.0.3 - 2026-05-23

- Made startup long VRAM calibration adaptive. Krasis now measures the short
  startup prefill first, then ramps the long calibration probe upward from
  observed low-water VRAM instead of immediately probing 80% of the context/KV
  cap. This avoids startup OOMs on WSL/shared-GPU systems where Windows
  processes reduce live free VRAM before Krasis starts.

## 1.0.2 - 2026-05-20

- Fixed VRAM pressure handling after runtime below-safety lows. The monitor now
  latches below-safety pressure events until HCS has reacted, and the HCS drain
  path preserves the measured low-water deficit as extra idle headroom instead
  of clearing the event when idle VRAM has merely recovered above the nominal
  safety margin.

## 1.0.1 - 2026-05-19

- Fixed setup compatibility for mixed Ampere and Blackwell systems. `krasis-setup`
  now selects the PyTorch CUDA wheel index from all visible GPU compute
  capabilities plus the installed driver runtime, so systems with RTX 50-series
  GPUs install CUDA 12.8 PyTorch wheels instead of CUDA 12.6 wheels that lack
  `sm_120` support.
- Added post-install validation for CUDA torch architecture support. If an
  existing CUDA torch build is visible but does not support one of the installed
  GPUs, setup reinstalls the correct wheel and the launcher reports the same
  unsupported-architecture condition clearly.

## 1.0.0 - 2026-05-19

- First stable Krasis release after the 0.1.x prerelease line.
- Moves the performance-sensitive runtime path to Rust/CUDA-focused execution,
  with full GPU prefill and GPU-executed decode.
- Adds HQQ attention cache support, compact KV cache modes, HCS expert
  residency management, measured VRAM budgeting, release wheels with sidecar
  assets, launcher model/download/tunnel workflows, and expanded benchmark and
  release validation.
- Current public benchmark coverage includes Qwen3-Coder-Next, Qwen3.6-35B,
  Qwen3.5-122B, and Qwen3-235B runs across the RTX 5090 and RTX A4500.

## 2026-02-28 — Decode optimisation: serial route matmul + AVX2 sigmoid

Baseline: 152.7 ms/tok (6.55 tok/s) at 12 threads on 5900X (WSL2, DDR4 dual-channel)
After: 119.0 ms/tok (8.40 tok/s) — 22% latency reduction, 28% throughput improvement

### Changes kept
- **Serial route matmul**: Changed MoE routing from parallel to serial dispatch.
  Rayon thread wake-up latency (~30us x 11 threads x 40 layers) dominated for 256
  tiny dot products (8KB per expert). Saved 32ms/tok (moe_route: 35.1ms to 3.1ms).
- **AVX2 vectorized sigmoid**: Replaced scalar sigmoid in SiLU activation with
  8-wide AVX2 fast exp approximation. Marginal (~1ms), but cleaner code with no downside.

### Changes tested and reverted
- Sub-tile splitting: cache line contention from two threads reading same tile data (+7%)
- Flattened MoE dispatch: cross-expert task mixing destroyed L2 cache locality (+22%)
- No nested MoE parallelism: only 8/12 threads active, lost work-stealing benefit (+10%)
- MADV_COLLAPSE (THP alternative): WSL2 returns EINVAL, Hyper-V doesn't support it
- Page table warming: pre-faulting PTEs polluted L1/L2, net slower (+5%)
- PREFETCHNTA (non-temporal prefetch on weights): ~2% improvement but within WSL2 noise

### Key findings
- 24 threads (SMT) is 2.3x slower than 12 threads (physical cores only) on 5900X
- THP does not work in WSL2 (AnonHugePages stays 0 despite madvise calls)
- TLB misses from 4KB pages on a 67GB model account for most of the remaining gap
  to theoretical bandwidth (~11 GB/s effective vs ~38 GB/s practical STREAM)
- MADV_COLLAPSE would likely work on native Linux and is worth retesting there

## 2026-02-28 — Log management and run notes

- On server start, existing `krasis.log` is archived to `logs/krasis_YYYYMMDD_HHMMSS.log` (timestamped from file mtime)
- Fresh `krasis.log` started for each run
- New `--note` parameter writes a run description header at the top of each log
- `logs/` directory gitignored (except `.gitkeep`)
