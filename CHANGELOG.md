# Changelog

## Unreleased

- Replaced the launcher Hugging Face model search/download flow with a curated
  supported-model downloader. The launcher now offers only the Krasis-supported
  model catalog, downloads each entry to the expected `~/.krasis/models/<name>`
  directory, and passes the catalog's pinned Hugging Face revision into
  `snapshot_download` so upstream `main` changes do not silently alter local
  installs. The initial catalog contains Qwen3-Coder-Next, Qwen3.6-35B-A3B,
  Qwen3.5-35B-A3B, Qwen3.5-122B-A10B, Qwen3-235B-A22B, and Gemma4 26B A4B IT.
- Started native Gemma4 text support on branch `gemma-dev`. Added Gemma config
  parsing for `top_k_experts`, `hidden_activation`, full-attention head
  geometry, `attention_k_eq_v`, and per-layer GQA helper methods; corrected the
  Gemma test config model path casing; taught the weight loader about Gemma's
  `router.proj`, router scale/per-expert-scale tensors, extra feed-forward
  norms, dense-MLP-plus-MoE layer shape, and full-attention `v_proj` aliasing.
  Extended the Marlin expert cache builder to detect Gemma's sibling
  `layers.N.experts.gate_up_proj/down_proj` stacked BF16 experts.
- Implemented the Gemma4 text hot path for INT4 routed experts, BF16
  attention, compact BF16 KV, and tied embedding/lm-head weights. The Rust
  prefill and decode paths now handle Gemma4 embedding scaling, final logit
  softcap, plain final RMSNorm, half-split RoPE, per-layer GQA geometry,
  full-attention `attention_k_eq_v`, no-scale V RMSNorm, Gemma4 router
  scale/per-expert scale, extra pre/post feed-forward norms, dense MLP plus
  routed MoE branch composition, and `layer_scalar` residual scaling.
- Added Gemma4 tokenizer handling for `extra_special_tokens` list configs and
  sibling `chat_template.jinja`, plus minijinja support for Python-style
  `.get(key[, default])` calls used by the Gemma chat template.
- Added Gemma4 variable per-layer compressed KV cache support for `k6v6` and
  `k4v4`. The default Gemma4 k6/k4 configs use full per-layer compressed KV
  storage: with `CFG_KV_CACHE_MB=1000`, k6v6 provides a `10624`-token context
  and k4v4 provides a `14880`-token context on the local RTX 5090.
- Implemented an experimental Gemma4 ring-window KV path for sliding-attention
  layers and gated it behind explicit `CFG_RING_WINDOW_KV=1` /
  `--ring-window-kv`. Fixed the k6v6 ring 25K long-prompt corruption by
  passing real prefill chunk starts into GQA attention and by routing
  ring-capped sliding layers through the custom local-window prefill path
  instead of the FA2 local-window path. k6v6 ring remains explicit/diagnostic
  pending witness validation and 100K quality review; k4v4 ring is now rejected
  before model load because it still produced invalid 25K output during
  validation. The k4v4 non-ring config remains supported.
- Gemma4 unsupported quantization modes now fail early and visibly. The model
  constructor rejects non-INT4 routed experts, non-BF16 attention, and KV
  formats outside BF16/k6v6/k4v4 for Gemma4 text instead of allowing known-bad
  modes to reach runtime.
- Gemma4 decode CUDA graphs are disabled by model capability rather than by an
  environment variable. The supported Gemma4 decode path is currently
  ungraphed; per-layer graph capture still needs a Gemma-aware graph segment
  implementation for dense-MLP-plus-routed-MoE layers.
- Improved the diagnostic Gemma4 k6v6 ring-window prefill path after timing
  showed the old custom path charging almost all 39,920-token prefill time to
  GQA/KV work (`151956 ms`, `263 tok/s`). Ring-capped sliding layers still use
  bounded local-window attention, now with an FA2 staging path when available,
  and full-attention layers no longer get routed through the custom ring branch.
  A 25K large-prompt validation improved from `421.6 tok/s` to
  `480.1 tok/s` prefill with unchanged `10.2 tok/s` decode, `100%` HCS hit
  rate, `0` copy failures, and `2098 MB` minimum free VRAM. A Gemma4 CUDA
  graph decode attempt failed during long calibration with
  `CUDA_ERROR_ILLEGAL_ADDRESS`, so the model capability guard remains in place
  and the failed graph experiment was removed from the shared graph path to
  avoid slowing existing graph-capable models.
- Improved Gemma4 k6v6 ring-window long-context decode without enabling CUDA
  graph capture. The ungraphed compressed-GQA decode path now reuses the
  existing tiled k6/k8 attention kernels for long sequences when tiled buffers
  and a device sequence-length scalar are available, and falls back to the
  original single-block kernel for short sequences or k4v4. Clean timing-off
  validation through `large_25k` passed with `498.2 tok/s` prefill,
  `28.4 tok/s` decode, `100%` HCS hit rate, `0` copy failures, and `2346 MB`
  minimum free VRAM. Follow-up prefill experiments were rejected: skipping old
  FP8 stage-cache rows produced invalid 25K output, and reducing only the
  export launch grid regressed 25K prefill to `470.2 tok/s`. A later
  ring-aware temp-KV cap reduced calibration prefill growth (`359.0 KB/tok`)
  but still regressed 25K prefill to `470.4 tok/s`; adding a 4096-token chunk
  cap reduced prefill growth further (`155.3 KB/tok`) but collapsed 25K
  throughput to `296.6 tok/s`. Both were reverted because they traded speed
  for memory headroom. A direct compressed k6 ring prefill prototype was also
  rejected: it completed startup calibration safely and reduced measured
  prefill memory growth to `345.1 KB/tok`, but benchmark warmup took `213.0s`
  and the first 1K timed prefill row still had not completed after a bounded
  wait. The active code was reverted to avoid degrading the accepted
  `498.2 tok/s` k6 ring path.
- Added a Qwen35-vs-Gemma4 timing diagnostic to explain the current Gemma
  prefill/decode gap. Added `tests/q35b-4-4-hqq6-k6v6-diagnostic.conf` because
  the older Qwen35 benchmark configs used disabled KV formats. With timing
  instrumentation enabled, Qwen35 HQQ6 k6v6 reached `9593.2 tok/s` internal
  prefill and `113.92 tok/s` internal decode with CUDA graph replay and `100%`
  HCS hits. Gemma4 k6v6 non-ring reached `4945.1 tok/s` best internal prefill
  only at the 1K row and fell to about `1000 tok/s` at 10K; timing showed the
  10K row spent `9404.8 ms` in stage-exact KV append over 30 GQA layers.
  Qwen35's comparable 10K row spent only `0.3 ms` in KV append over 10 GQA
  layers. Gemma4 decode remained ungraphed at `36.60 tok/s`; HCS was `100%`,
  so the decode gap is graph/layer-path cost, not expert cache misses.
  Fixed Gemma4 prefill timing attribution so the top-level `attn/gqa/moe`
  buckets account for the nested Gemma layer timers instead of charging the
  whole Gemma wrapper as broad MoE/other time. A smoke diagnostic on
  `tests/gemma-4-4-k6v6-a16.conf` now reports an 8419-token calibration row as
  `6434.9 ms` GQA/attention with `6304.7 ms` in KV append and only `63.2 ms`
  MoE, matching the root-cause attribution without changing timing-off paths.
- Ran the next Gemma4 prefill/graph speed implementation pass and rejected the
  unsafe or slower paths by measurement. BF16 temp-KV staging worsened the
  8419-token diagnostic (`6679.2 ms` KV append versus the prior `6304.7 ms`);
  FP8-window FA2 routing, vectorized FP8 append, and a direct decode-KV
  single-chunk bypass all left the 39,920-token timing row around `281 tok/s`
  with about `141 s` in KV append, so their active code was removed. A Gemma4
  CUDA graph decode implementation attempt reached roughly `64-73 tok/s`, but
  failed correctness (`2/10` network prompts passed, early EOS/garbled output),
  showing that Gemma4 needs split graph segmentation for dense MLP, router,
  expert, merge, and `layer_scalar` work before replay is safe. The Gemma4
  graph guard is restored with that explicit reason, and unused FP8-window
  sidecar symbols/loaders were removed. Restored-path validation
  `./dev test tests/gemma-4-4-k6v6-a16.conf` passed benchmark plus network:
  `5035.9 tok/s` best prefill, `38.72 tok/s` best internal decode,
  `67.57 tok/s` best HTTP, `14/14` network prompts, HCS `3840/3840`, and
  `11170 MB` minimum free VRAM.
- Implemented Gemma4 CUDA graph decode safely by making the captured graph path
  execute Gemma4's dense-MLP + routed-MoE semantics instead of the generic MoE
  merge. The graph segment now handles Gemma4 pre/post attention RMSNorm,
  dense branch, pre-FFN2 expert input, router input scaling,
  per-expert router scaling, post-expert/post-FFN norms, residual merge, and
  `layer_scalar`. Graph replay now uses per-layer device sequence-length
  scalars, so Gemma sliding-attention layers use their bounded attention length
  while full-attention layers still see full context. Validation:
  `./dev build` passed; `./dev run tests/gemma-4-4-k6v6-a16.conf --test-endpoints`
  plus `./dev network 18013` passed `14/14`; timing-off
  `./dev benchmark tests/gemma-4-4-k6v6-a16.conf` reached `5230.3 tok/s`
  best prefill, `63.92 tok/s` best internal decode, `116.20 tok/s` best HTTP,
  HCS `3840/3840`, `0` copy failures, and `11156 MB` minimum free VRAM.
  Gemma graph decode is deliberately limited to the validated non-ring `k6v6`
  path; k4v4 and ring-window Gemma remain on their existing non-graph paths
  until separately validated.
  Guard run `./dev speed-test` on Qwen3-Coder-Next HQQ4/k4v4 completed with
  `6664.9 tok/s` prefill, `88.42 tok/s` internal decode, `138.06 tok/s` HTTP,
  HCS `15957/24576`, `0` copy failures, and `896 MB` minimum free VRAM.
- Validation: `./dev build` passed. Short reference probes against the legacy
  Gemma4 HF BF16 artifact matched turn 1 exactly and turn 2 first token exactly
  without setting `KRASIS_NO_GRAPH`. The full legacy reference sweep passed
  turns 1-6 and diverged on longer generations with coherent output; Gemma4
  INT4 is not yet llama-witness validated because no local Gemma witness/GGUF
  artifact exists. `./dev benchmark tests/gemma-4-4-a16.conf` completed with
  a clean health scan: `5051.6 tok/s` best prefill, `39.07 tok/s` internal
  decode, `71.33 tok/s` HTTP, `3840/3840` HCS, and `11084 MB` minimum free
  VRAM. `./dev network 18013` on the gated non-ring k6v6 config passed
  `14/14` with `10624` context, and `./dev network 18014` on the gated
  non-ring k4v4 config passed `14/14` with `14880` context; both had clean
  health scans. k6v6 ring-window reached `106784` context and passed the
  formerly failing 25K large-prompt row after the ring fix (`421.6 tok/s`
  prefill, `2222 MB` min free), then the prefill speed follow-up passed the
  same 25K row at `480.1 tok/s`; the 100K network row timed out while the
  server remained GPU-bound, so 100K is still not a validated practical path.
  k4v4 ring-window is explicitly rejected after
  reproducing invalid 25K output. INT8 Gemma4 remains unsupported after diagnostics produced invalid
  token/logprob output.

## 1.0.15 - 2026-06-05

- Hardened Typhon/16GB startup and CUDA-fault safety. HCS startup now keeps two
  measured soft-tier chunks of headroom above the calibrated decode idle floor,
  and server startup gates `KRASIS SERVER READY` on bounded measured
  drain-and-resample checks after VRAM monitor warnings are enabled. If the
  configured safety floor cannot be restored, startup fails visibly instead of
  publishing an unsafe server. CUDA `ILLEGAL_ADDRESS` in prefill/scratch
  release or source-mode HCS boundary sync now exits the process immediately,
  because the CUDA context is poisoned and cannot be recovered by further HCS
  evictions. Validation: `./dev build` passed.
- Hardened request-time VRAM pressure recovery for external VRAM consumers on
  Typhon. Calibration extrapolation now continues the measured slope for prompts
  beyond the long probe instead of clamping at 1.5x, HCS prefill eviction uses
  the actual request token count instead of a 50K cap, minimum-chunk scratch
  allocation fails visibly if the measured post-allocation floor still cannot
  be met, and chat request entry drains pending HCS pressure before tokenization
  or prefill work. The VRAM monitor now records below-critical-floor pressure
  as an immediate drain event rather than exiting solely because an external
  process temporarily consumed VRAM; CUDA illegal-address errors remain fatal.
  Validation: `./dev build` passed with CPython 3.12 packaging for Typhon.
  Typhon stress with a separate CUDA process holding `768 MB` during a
  `64,125`-token prompt completed with HTTP 200, prefill `3610.4 tok/s`, decode
  `29.7 tok/s`, prefill min free `1902 MB`, decode min free `2646 MB`, and
  `copy_failures=0` on an `800 MB` safety margin. A post-pressure small prompt
  completed with decode `54.5 tok/s`, decode min free `918 MB`, and
  `copy_failures=0`.
- Fixed post-scratch reserve planning to use the measured runtime transient for
  the actual prompt length. This preserves the safety-floor check while avoiding
  a false release-test failure where a tiny QCN warmup request inherited the
  long-prompt post-scratch reserve and refused to run despite enough measured
  headroom for that request size.
- Fixed HCS pressure-drain targets to recover to `safety + measured_deficit`
  after a below-safety low, and to keep two physical soft-HCS chunks of
  headroom during forced readiness/decode drains. This was exposed by the QCN
  INT8 multi-GPU release-test path, where GPU0 published ready with only
  `744 MB` free against a `600 MB` margin and later dipped below the floor
  before the next prefill.
- Added a configured-safety-margin entry guard to HCS prefill eviction. Runtime
  prefill starts from a decode-resident HCS state rather than the no-HCS
  calibration state, so the eviction floor now preserves the measured prefill
  requirement plus HQQ stage growth plus two additional configured safety
  margins before scratch allocation.
- Tightened optional prefill pinning so it cannot consume the last measured
  post-scratch safety band. Pinning now only uses surplus above the calibrated
  post-scratch runtime floor plus one configured safety margin, preserving the
  optimization while preventing short HCS-loaded prefill prompts from launching
  with only a tiny margin above the floor.
- Release validation: `./dev release-test QCN` passed all three Qwen3-Coder-Next
  release configs with llama-witness `4/4` validation on each config. Final
  release-test health scan was clean: no CUDA errors, VRAM monitor warnings,
  hard-floor exits, or nonzero HCS copy failures. Release-test speeds were
  INT4 k4v4 HQQ4 `6433.2 tok/s` prefill, `90.36 tok/s` decode,
  `158.41 tok/s` HTTP; INT4 k6v6 HQQ6+8 `6377.3 tok/s` prefill,
  `88.96 tok/s` decode, `144.13 tok/s` HTTP; INT8 k6v6 HQQ6+8 multi-GPU
  `5000.9 tok/s` prefill, `40.30 tok/s` decode, `68.01 tok/s` HTTP.
- Reduced Q122B prefill scratch for fused MoE by sizing the routed/shared
  accumulator as `[tokens, hidden]` FP32 instead of routing-width scratch.
  The fused Marlin reduction scratch is already provided by `d_fp32_scratch`,
  so `d_moe_accum` only needs to hold the final per-token accumulator. The
  exact estimator, coarse startup budget, and actual scratch allocator were
  updated together.
- Validation: `./dev build` passed. Instrumented Q122B HQQ6/k4v4 diagnostic
  completed with a clean health scan, raised the measured-safe long calibration
  probe to `19268` tokens, and made the `20K` diagnostic row one chunk at
  `4737 tok/s`. Clean timing-off
  `./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf` produced
  `3900.5 tok/s` internal prefill, `27.78 tok/s` internal decode, and
  `45.98 tok/s` HTTP with `4050/12288` HCS, `806 MB` minimum free VRAM, and no
  CUDA errors, VRAM monitor warnings, hard-floor exits, or HCS copy failures.
- Added env-gated request/runtime-stage prefill breakdown diagnostics. The
  diagnostic showed `20K` Q122B core prefill was already close to the old fast
  path (`4156 ms` body time), while the broader request window still paid
  about `1.2 s` in HQQ prefill/decode stage copies. Tested and rejected two
  HQQ-stage optimizations: dual-stage residency removed the copy but consumed
  about `4.9 GB` extra VRAM, reduced HCS to `2862/12288`, and regressed
  decode/HTTP; async copy-stream scheduling did not improve stage-copy time.
  After removing those rejected paths, a clean timing-off Q122B benchmark
  produced `3796.8 tok/s` internal prefill, `28.98 tok/s` internal decode, and
  `46.71 tok/s` HTTP with `4050/12288` HCS, `772 MB` minimum free VRAM, and a
  clean CUDA/VRAM/HCS health scan.
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
