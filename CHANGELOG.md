# Changelog

## 1.0.16-rc.3 - 2026-07-24

- Fixed complete Windows uninstall of the private runtime. A real preflight
  proved Inno's recursive deletion could not enumerate the deepest PyTorch
  header paths and returned success while retaining that directory chain.
  Uninstall now runs a packaged, fail-closed extended-length Win32 traversal
  before normal Inno removal. It refuses to proceed while any installed
  Krasis Python tree is in use, never follows reparse points, removes only the
  installer-owned `runtime`, legacy `python`, and legacy `venv` trees, and
  verifies each is absent. The Windows lifecycle gate now exercises both an
  empty directory and a junction to an external sentinel, proving cleanup
  removes empty nodes without traversing reparse points. It preserves complete
  Inno logs and still rejects any retained runtime. CI now compiles the Inno
  script against a minimal staged payload before the expensive native build,
  so installer-language errors fail early.

- Fixed the private-runtime validation probe under Windows PowerShell 5.1.
  PowerShell's legacy native-command argument handling corrupted the quoted
  multiline Python source passed through `python -c`. The installer now starts
  the absolute private interpreter through .NET with isolated mode and sends
  the probe over redirected standard input, avoiding shell parsing while
  retaining structured output, stderr, and exact process-status checks.

- Fixed private-runtime payload verification on Windows PowerShell 5.1 by
  hashing files directly with .NET SHA-256 instead of depending on the
  unavailable `Get-FileHash` command. The validated runtime is now transported
  through Inno as one immutable ZIP whose exact SHA-256 is compiled into the
  installer. The Windows PowerShell 5.1 hook verifies that archive byte for
  byte, extracts it with .NET into an inactive side-by-side runtime, and runs
  the existing ABI/import validation before atomic activation.

- Replaced the native Windows installer's overlayed Python/venv lifecycle with
  a release-built private runtime. Windows CI now builds CPython 3.12.10 and
  Krasis core dependencies in a clean directory, records a deterministic
  payload manifest, validates relocation and isolated imports, and packages
  that immutable core runtime. On installation, Krasis stages a unique
  side-by-side runtime, verifies its payload hash, Python ABI, 64-bit
  architecture, regex/SSL standard library, native extension, and exact
  Krasis version. Both the official CPython embeddable runtime archive used by
  CI and the CUDA PyTorch 2.9.1/cu128 wheel used on the target are SHA-256
  pinned. The complete runtime is validated before the active pointer changes
  atomically. Failed installs keep the previous runtime active. The launcher
  now invokes only that absolute private interpreter with isolated mode and
  has no system `py`, `python`, `PATH`, or retained-venv fallback.
  Successful upgrades remove inactive legacy mixed `{app}\python` and
  `{app}\venv` trees; if an old server still uses one, it is retained without
  affecting the newly activated runtime, and uninstall owns all runtime paths.
  The Inno payload remains live through the `ssPostInstall` runtime hook,
  runtime setup writes a retained diagnostic transcript, and hook failures set
  a nonzero installer exit code so silent upgrades cannot report success.
  Windows CI now performs a real clean installer run with hostile
  `PYTHONHOME`/`PYTHONPATH`, legacy-directory repair checks, exact runtime
  probes, and uninstall cleanup before uploading the installer.

- Refreshed the GitHub front page for the `v1.0.16-rc.2` release line. Added a
  prominent direct Windows installer link, the prerelease Linux/WSL install
  command, release navigation, and clear Start Menu update behaviour. Replaced
  the stale selected benchmark table with current validated headline results
  linked to the complete speed, quality, and reproducibility records.

## 1.0.16-rc.2 - 2026-07-24

- Validated the published prerelease from a clean Ubuntu 24.04 Podman
  environment: the public installer bootstrapped Python 3.12, `krasis-setup`
  installed CUDA 12.8 and CUDA PyTorch, and the interactive launcher built
  fresh caches and served Qwen3.5-35B-A3B successfully. The clean-test Podman
  setup now exposes the host model library read-only, so install tests can run
  existing models without inheriting host software or copying model weights.
  Raw benchmark and terminal-transcript files now have a narrow Git whitespace
  policy so their original logger/TTY formatting is preserved while source and
  documentation remain covered by normal whitespace checks.

- Added native Windows Start Menu update entries. `Krasis Update` resolves and
  installs the latest stable GitHub release, while `Krasis Prerelease` resolves
  and installs the latest published prerelease. Both use the matching
  `KrasisSetup-*-win64.exe` release asset through a shared packaged PowerShell
  updater, and Inno Setup owns the shortcuts so upgrades and uninstalls manage
  them alongside the existing interactive `Krasis` entry.

## 1.0.16-rc.1 - 2026-07-23

- Fixed multi-GPU GQA/HQQ auxiliary decode-store setup after the release matrix
  exposed a `NameError` while constructing per-layer RoPE tables. The RoPE table
  builder is now a shared model method used by both primary and auxiliary GPU
  stores, preserving the same model-derived dimensions, scaling parameters, and
  per-device cache keys.

- Fixed streamed timing metadata when thinking is disabled. Every measured
  generated token is now reported as an answer token in that mode, including
  the first token produced by prefill, so clients no longer display a real
  decode rate beside an incorrect `0t` token count.

- Fixed the active Nemotron-H output-quality collapse by removing an incorrect
  load-time `1/sqrt(num_hidden_layers)` scaling of trained Mamba2
  `out_proj.weight` tensors. `rescale_prenorm_residual` is checkpoint
  initialization metadata; pretrained tensors must be loaded verbatim. Bumped
  the Mamba2 projection INT4 cache version so scaled caches cannot be reused,
  and made BF16 routed-expert CUDA graph support fail clearly before capture
  because that validation-only expert path is not graph-backed. Added
  llama-witness conversion/capture support for Nemotron Nano and Super,
  including Super LatentMoE metadata and tensor mappings. The production
  HQQ4/k4v4 configurations now pass six-prompt llama-witness gates for both
  models with `6/6` prefill top-10 containment and `4/6` first-token argmax.
  Public quality stats now mark those two HQQ4 rows `PASS`; HQQ6 rows remain
  blocked until separately compared.

- Added an explicit, default-off adaptive cold-mass pruning experiment for
  fine-grained MoE decode. The Rust runtime can omit only exact demand-cold
  routes outside a configured protected router-rank head, subject to a
  per-layer routed-mass cap; surviving weights are not renormalized. A
  shadow-only sweep reports projected drops, saved DMA bytes, dropped mass,
  and rank distribution without changing outputs. Normal Rust MoE decode and
  fixed-shape CUDA-graph replay are supported; GPU route sync and speculative
  decode reject the approximate mode visibly until equivalent implementations
  exist. On Ornith-1.0-397B with 42.8% HCS residency, protecting 75% of router
  ranks with an 8% per-layer mass cap improved 50/100/250-token internal decode
  from `23.58/21.85/20.40` to `25.73/23.81/22.46` tok/s and passed the 14-test
  network suite. In the steady 250-token block it dropped `4754/19918` demand-
  cold routed-expert activations (`23.87%`, or `19.092/79.992` per token) and
  avoided `118.134 MiB/token` of serialized DMA. The `8%` setting is a per-
  token/per-layer ceiling rather than a target: mean dropped routed mass was
  only `1.884329%`, while the largest individual routing event reached
  `7.999461%`. In the matching shadow sample, ranks 9/10 were 20% of router
  slots but contained `31.15%` of cold routes, and `75/8` admitted `76.93%` of
  that cold tail for dropping. Only routed experts at ranks 9/10 were dropped;
  the separately executed, VRAM-pinned shared expert is never eligible. The
  disabled path remains the default, and the mode is not
  classified as production-safe because no Ornith llama-witness artifact is
  available. Cross-model RTX PRO 6000 checks on fully resident QCN found zero
  eligible cold routes and zero drops under explicit `75/8` mode; its
  `90.86` tok/s decode result matched the `90.93` tok/s default-off control.
  The interactive launcher now presents the experiment as **Adaptive cold-mass
  pruning**, defaulting to `Off` and cycling through the measured `75/3`,
  `75/5`, `75/8`, and `75/10` presets. The readable preset is persisted as
  `CFG_ADAPTIVE_COLD_MASS_PRUNING` and translated at server startup into the
  existing Rust environment contract.

- Ran a targeted Ornith-1.0-397B HQQ4/k4v4 prefetch/split smoke matrix on
  `step-main-port` current main. Gated prefetch still issued zero staged
  experts (`issued=0`, `budget_dropped=10925`, measured `10.87 GB/s` H2D), so
  gate-off and prefetch+split were intentionally skipped. Split launch completed
  cleanly at `10.08` best decode tok/s, but the warmed repeat baseline reached
  `9.66` tok/s, so the measured split effect is only small/noise-level and not
  default-worthy. Dynamic HCS remained clean with `budget_skips=0`,
  `no_slot=0`, and `copy_failures=0`.

- Ran the current-main `step-main-port` baseline validation on the local RTX
  5090 with all experimental sync-wait flags off. Results: Qwen3-Coder-Next
  HQQ4/k4v4 `6,880.9` prefill / `88.36` decode tok/s, Step-3.7-Flash
  HQQ4/k4v4 `2,623.8` prefill / `22.12` decode tok/s, and Ornith-1.0-397B
  HQQ4/k4v4 `789.7` prefill / `8.26` decode tok/s. Ornith HCS coverage is back
  to `2720/30720` with `budget_skips=0` and `copy_failures=0`, confirming the
  earlier `5.24` tok/s result was caused by the stale base rather than the
  model or config.

- Added the first native Windows installer/build path. Windows wheels can now
  include Krasis sidecar DLLs, bundled CUDA runtime DLLs, and resolve them via
  `os.add_dll_directory`; Rust prefill sidecar loading now uses a
  cross-platform dynamic loader instead of Unix-only `dlopen`/`dlsym`. Added
  PowerShell installer/launcher scripts that create a per-user install under
  `%LOCALAPPDATA%\Programs\Krasis`, install a bundled private Python runtime
  and Krasis environment, and create a Start Menu shortcut that opens a
  maximized PowerShell window running the interactive launcher. Added a Windows
  installer GitHub workflow that builds sidecar DLLs, builds/verifies a Windows
  wheel, creates an offline wheelhouse, downloads the matching Python
  installer, and packages `KrasisSetup-*-win64.exe` with Inno Setup. The first
  Windows target is Marlin/FlashAttention-backed models; FLA/linear-attention
  models still require a separate native Windows FLA sidecar port. The Windows
  sidecar build now also passes `-std=c++17` explicitly to nvcc so MSVC-hosted
  CUDA builds accept the C++17 inline-variable and `if constexpr` constructs
  already used by the Marlin/FlashAttention sources. Marlin sidecar dispatch
  now returns directly from matching kernel cases instead of expanding one long
  `else if` ladder, avoiding an MSVC/nvcc compiler nesting limit while
  preserving the same kernel instantiations. FlashAttention's vendored CUTLASS
  platform shim now treats `_MSVC_LANG` as the C++17 signal for `_v` type-trait
  aliases, and the FlashAttention vendor/header path defines `M_LOG2E` when
  MSVC does not expose the POSIX math constant. Windows wheel builds now gate
  Unix-only mmap advice and raw POSIX signal handlers behind `cfg(unix)`, and
  the VRAM monitor resolves CUDA runtime symbols through the bundled Windows
  CUDA DLL path instead of Unix `dlopen`. CPU decode weight consolidation now
  uses owned contiguous backing buffers on non-Unix platforms instead of
  anonymous `mmap`, and NUMA policy/CPU-affinity calls are Unix-gated with a
  single-node fallback on Windows. Windows wheel verification now probes
  extracted sidecar DLLs in a short child process so the DLL handle is released
  before the temporary extraction directory is deleted. The Windows installer
  shortcut now relies on PowerShell's `-WindowStyle Maximized` flag rather than
  an unsupported Inno Setup `[Icons]` parameter; CI had already passed sidecar
  DLL build, Windows wheel build, wheel verification, and wheelhouse assembly
  before failing at that installer-script parse step. The shortcut working
  directory now uses `{app}` instead of unsupported `{userprofile}` so the
  Inno script relies only on built-in constants.

- Added Gemma4 image support with the same lazy architecture as Qwen/Step
  vision. Gemma4 support is detected from `gemma4` metadata plus
  `model.vision_tower.*` and `model.embed_vision.*` safetensors entries; image
  requests load the local Gemma4 preprocessor/vision/embedder slice on demand,
  stage it on GPU only while image embeddings are generated, then release it
  back to CPU before text prefill/decode. Vision quantization now defaults to
  INT4 through `--vision-quant int4`, with BF16 available via
  `--vision-quant bf16` for validation. Validated on
  `tests/gemma-4-4-hqq4-k4v4-a16.conf` with the same blue-square image request:
  BF16 answered `Blue` with `1092.5 MB` resident vision payload and `1170 MB`
  released after staging; default INT4 also answered `Blue` with `316.0 MB`
  resident payload and `380 MB` released after staging. Gemma4 sliding
  attention image-token blocks now pass a vision-block mask into the Rust
  prefill path so full-attention layers remain causal while image soft-token
  blocks get the required bidirectional overlay.

- Added explicit Step-3.7 vision INT4 quantization for image requests via
  `--vision-quant int4` / `--step-vision-quant int4` and
  `--vision-group-size` / `--step-vision-group-size`. The path keeps the
  BF16 lazy vision architecture intact, then packs Step vision linears,
  convolution weights, and attention `in_proj_weight` into signed INT4 plus
  BF16 scales. The first validation path dequantizes one module at a time during
  the image forward, so resident VRAM is reduced before fused vision kernels
  exist. Measured on the 5090: BF16 vision staged `3803.0 MiB` allocated with
  `4160.2 MiB` peak, while INT4 staged `1024.0 MiB` allocated with `3145.9 MiB`
  peak in the isolated probe. Full server validation with
  `tests/step37-flash-4-4-hqq4-k4v4-a16.conf --step-vision-quant int4`
  answered blue square, red circle, and green triangle image prompts correctly;
  the server logged `quant=int4`, `vision_resident_mb=1005.1`, and release back
  to CPU after each request.

- Added the Step-3.7-Flash BF16 vision path using the same lazy architecture as
  Qwen vision: Step image support is detected from local model metadata, the
  vision tower/projector load on first image request, move to GPU only while
  image embeddings are produced, then return to CPU. OpenAI-style `image_url`
  parts are normalized for multimodal chat templates, while the original
  request JSON remains the image source. Validated with `Step-3.7-Flash-vision`
  on a synthetic image request: the server returned `The square is blue.`,
  reported `201` prompt tokens and `6` completion tokens, and the release probe
  measured `4072 MiB` freed after moving BF16 vision back off GPU.

- Verified local `main` at `10f0bc2` with QCN and Gemma through the built
  command path. QCN ran
  `./dev test tests/qcn-k4v4-hqq4-int4-benchmark.conf` with only accepted
  Gemma decode env gates enabled and attribution/rejected-candidate envs unset.
  It passed `14/14`, `ALL TESTS PASSED`, with `6356.7` prefill, `87.74`
  internal decode, `150.14` HTTP, HCS `15957/24576`, min free decode VRAM
  `896 MB`, and zero copy failures. Gemma ran
  `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` and passed `14/14`,
  `ALL TESTS PASSED`, with `4936.2` prefill, `92.27` internal decode,
  `157.84` HTTP, HCS `3840/3840`, min free decode VRAM `11474 MB`, and zero
  copy failures. A follow-up `./dev benchmark` Gemma speed check produced
  `5378.2` prefill, `92.43` internal decode, and `160.26` HTTP. Both Gemma
  launches rebuilt the GPU Marlin expert cache, leaving a follow-up cache-reuse
  issue: runtime reports `GPU INT4 g128 (amax)` while the persisted cache file
  is `experts_marlin_int4_g64_calamax.bin`.

- Opened guarded Nano parallel SSD output bottleneck speed gate
  `20260629_2009_nemotron_nano_parallel_ssd_output_bottleneck_speed`.
  This records the `1818` guarded by-chunk result first: long `39920`
  prefill `7.01s` / `5693.8 tok/s`, with candidate averages total
  `177.279ms`, output/chunk-scan `148.531ms`, recurrent `12.646ms`, exact
  bridge `0.000ms`, chunk state `10.763ms`, and state passing `1.781ms`.
  Scope is limited to a further opt-in output-kernel speed candidate inside
  the existing guarded parallel SSD startup-diagnostic path; the `1818`
  by-chunk kernel remains the comparison control, token/logit parity is the
  correctness gate, and default sequential SSD, production behavior,
  decode/HCS, protected configs, and direct Python scripts/one-liners remain
  unchanged. Completed the guarded output bottleneck speed pass under
  `KRASIS_MAMBA2_SSD_PARALLEL_OUTPUT_PRECOMPUTE_CB=1`, still requiring the
  existing guarded parallel-chunked/by-chunk startup diagnostic path. Added a
  CB precompute tile and an ordered loop-unroll in the precompute-CB output
  kernel only; the by-chunk control and default sequential SSD remain
  unchanged. `./dev build` passed. Selected-token smoke passed on the
  462-token reference payload (`1044`, text `,`); separate-process top-k
  deltas remain diagnostic because control-vs-control drift was already
  measured in this gate. Clean long `39920` no-oracle timing exited `0` with
  `23` long records. Final long prefill: `5.14s` / `7765.1 tok/s`. Candidate
  averages: total `98.162ms`, output `66.310ms`, CB precompute `3.002ms`,
  recurrent `12.792ms`, chunk state `10.786ms`, state passing `1.782ms`.
  Versus the `1818` by-chunk baseline: throughput `+36.4%`, total candidate
  `-44.6%`, output `-55.4%`. The failed state-split branch remains gated and
  is not promoted.

- Opened guarded Nano parallel SSD output kernel redesign gate
  `20260629_1818_nemotron_nano_parallel_ssd_output_kernel_redesign`.
  This records the `1408` token/logit-correct guarded path first: long
  `39920` prefill `17.99s` / `2219.4 tok/s`, with output/chunk-scan still
  dominating at `627.436ms` of `656.022ms` candidate time. Scope is limited to
  an additional opt-in output redesign inside the existing guarded startup
  diagnostic path; default sequential SSD, production behavior, decode/HCS, and
  protected configs remain unchanged. Completed the guarded by-chunk output
  redesign under `KRASIS_MAMBA2_SSD_PARALLEL_OUTPUT_BY_CHUNK=1`: final build3
  passed after replacing a literal shared-memory threshold with runtime CUDA
  device attributes. Token/logit comparison against the accepted guarded
  control matched exactly
  (`12/12` top-logprobs, max delta `0.000000000`, token `1044` text `,`), and
  clean long `39920` no-oracle timing exited `0` with `23` long records. Long
  calibration prefill improved to `7.01s` / `5693.8 tok/s`. Candidate
  averages: total `177.279ms`, output `148.531ms`, recurrent `12.646ms`,
  exact bridge `0.000ms`, chunk state `10.763ms`, state passing `1.781ms`.
  Versus `1408`: total candidate `-73.0%`, output `-76.3%`, prefill
  throughput `2.57x`. Path remains guarded and not production-enabled.

- Opened guarded Nano parallel SSD fast handoff/output implementation gate
  `20260629_1408_nemotron_nano_parallel_ssd_fast_handoff_output`. This records
  the corrected remeasurement finding that the opt-in parallel chunked path is
  exact but currently slow because of the exact recurrent bridge and corrected
  output replay. Scope is limited to the opt-in startup-diagnostic path; default
  sequential SSD, production behavior, decode/HCS, and protected configs remain
  unchanged. Completed the guarded fast handoff/output fix: build attempt 3
  compiled prefill PTX, token/logit reference-test parity passed exactly
  against serial/control (`12/12` tracked top-logprobs, max delta
  `0.000000000`, token `1044` text `,`), and clean long no-oracle timing
  exited `0` with `23` long records and no oracle/subloop/request records.
  Long `39920` calibration prefill improved to `17.99s` / `2219.4 tok/s`.
  Candidate averages: total `656.022ms`, output `627.436ms`, recurrent
  `12.638ms`, exact bridge `0.000ms`, chunk state `10.698ms`, state passing
  `1.783ms`. Versus `1038`: total candidate `-93.3%`, output `-93.1%`,
  recurrent `-98.3%`, exact bridge `-100%`. Remaining old serial-oracle
  mismatches are BF16-boundary diagnostics after bridge removal; token/logit is
  the acceptance surface for this architecture. Path remains guarded and not
  production-enabled. Next target is the output/chunk-scan kernel.

- Opened Nemotron Nano guarded parallel SSD corrected-path remeasurement gate
  `20260629_1038_nemotron_nano_parallel_ssd_corrected_remeasure`.
  Recorded `2104` first: the guarded parallel SSD surface is correctness-fixed
  for the 300-token multi-chunk oracle (`69/69` exact, `0` output/state
  mismatches; row97/head48/d24, row64/state0, row82/state14 exact) and the
  42-token token/logit comparison matched serial/control exactly
  (`12/12` top-logprobs, max delta `0.000000`). Baselines carried forward:
  Gemma `5619.6 / 92.43 / 155.69`; Nano `346.5 / 94.48 / 164.75`. Scope:
  remeasure the corrected guarded `KRASIS_MAMBA2_SSD_PARALLEL_CHUNKED=1`
  no-oracle startup diagnostic path against the prior `1647` long timing;
  no code change, build, request, speed benchmark, decode/HCS work,
  production enablement, default-path change, or subloop instrumentation.
  The clean no-oracle startup diagnostic completed (`exit 0`, `276s`,
  `23` long records). Corrected long `39920` timing averaged total candidate
  `9827.873ms`, recurrent total `735.990ms`, output kernel `9075.657ms`,
  exact recurrent bridge `723.448ms`, chunk cumsum `0.040ms`, chunk state
  `10.705ms`, and state passing `1.783ms`. This is not a speed win versus
  the pre-correctness `1647` prototype; it proves the current correctness
  bridge and output surface are the next measured blockers. No oracle,
  subloop, request, speed benchmark, decode/HCS, production, or default-path
  work occurred.

- Opened Nemotron Nano guarded parallel SSD token/logit correctness gate
  `20260628_2104_nemotron_nano_parallel_ssd_token_logit_correctness`.
  Recorded `2017` first: guarded
  `KRASIS_MAMBA2_SSD_PARALLEL_CHUNKED=1` was narrowed from `73071` to `17`
  output mismatches against the old internal serial oracle, but still is not
  accepted by that diagnostic. Baselines carried forward: Gemma
  `5619.6 / 92.43 / 155.69`; Nano `346.5 / 94.48 / 164.75`. Scope: compare
  token/logit behavior of the accepted serial Krasis path and the guarded
  parallel SSD path using the existing 42-token Nano raw payload; keep
  defaults, production behavior, speed benchmark, decode/HCS, and unrelated
  paths unchanged.
  Patch attempt 1 keeps the path guarded and aligns parallel chunk cumsum with
  the accepted coeff-tile parallel-prefix order; no default-path or production
  behavior change. Patch attempt 2 keeps the path guarded and adds an
  exact-state bridge for correctness debugging: the parallel chunked phases
  still run, but chunk-entry/final state used by output/commit is produced by
  the accepted token-order recurrent kernel while the parallel state handoff
  remains under investigation. This is diagnostic only under
  `KRASIS_MAMBA2_SSD_PARALLEL_CHUNKED=1`; no default-path or production
  behavior change. The first 42-token oracle run after that bridge had
  `0` output mismatches but state still mismatched because the bridge started
  from `candidate_state` after parallel state-passing had mutated it; patch
  attempt 3 restores `initial_state` into `candidate_state` immediately
  before the bridge.
  After that fix the 42-token oracle reduced to a single output-only mismatch
  at layer2 row28/head42/d50 with state exact. Patch attempt 5 updates the
  guarded parallel output coefficient arithmetic order to match the accepted
  coeff-tile output path on BF16 boundaries. That made the 42-token oracle
  exact (`161/161` records), while the 300-token multi-chunk oracle still
  failed output-only in chunk>0 with state/probes exact; patch attempt 6 now
  passes `c_state_total_exact` into the guarded parallel output kernel for
  accepted prior reconstruction. Patch attempt 8 completed the output parity
  fix by carrying the accepted absolute `dA_chunk_base + dA_prefix` subtraction
  surface into the guarded parallel output kernel. `./dev build` passed
  (`180s`), the 300-token multi-chunk full-oracle diagnostic passed
  (`69/69` records exact, `0` output/state mismatches, row97/head48/d24,
  row64/state0, and row82/state14 exact), and the no-oracle 42-token
  `/v1/internal/reference_test` comparison matched the accepted serial/control
  first token and all `12` tracked top-logprobs exactly (`max_delta=0.000000`).
  This fixes correctness for the guarded parallel SSD surface; defaults,
  production behavior, speed benchmark, decode/HCS, and unrelated paths remain
  unchanged.

- Opened Nemotron Nano guarded parallel SSD correctness-debug gate
  `20260628_2017_nemotron_nano_parallel_ssd_correctness_debug`. Recorded
  `1647` first: `KRASIS_MAMBA2_SSD_PARALLEL_CHUNKED=1` is implemented and
  gives a large gated state-source timing reduction, but it is not correctness
  accepted yet. Baselines carried forward: Gemma
  `5619.6 / 92.43 / 155.69`; Nano `346.5 / 94.48 / 164.75`. Scope:
  preserve default sequential SSD, full-oracle path, candidate timing,
  decode/HCS, request path, speed benchmark path, and production behavior;
  debug/fix only the guarded parallel SSD correctness path using built
  commands and evidence from phase comparisons.

- Closed Nemotron Nano guarded parallel SSD implementation gate
  `20260628_1647_nemotron_nano_parallel_ssd_architecture_implementation`.
  Recorded `1630` first and implemented only an opt-in
  `KRASIS_MAMBA2_SSD_PARALLEL_CHUNKED=1` prototype under the existing
  block-scan/coefficient-tile/startup-diagnostic gates. Defaults, the
  existing full-oracle path, candidate timing mode, request path, speed
  benchmark path, decode/HCS, and production behavior were preserved.
  `./dev build` passed (`178s`). The smallest full-oracle startup diagnostic
  executed the new path and failed closed against the old serial oracle:
  output mismatches `73071`, first output mismatch row64/head1/d5
  `0xbdef` vs `0xbdee`, output max/mean `16.0` /
  `0.0020988934409373883`; state mismatches `524273`, first state flat0
  `0x3b90c20e` vs `0x3b90c258`, state max/mean `10.61376953125` /
  `0.0012460283959495207`. The no-oracle startup path completed, and the
  long no-oracle candidate-timing diagnostic (`39920` tokens, `23` records)
  averaged total candidate `555.775ms`, recurrent/state source `14.890ms`,
  output kernel `525.565ms`, live state copy `13.148ms`, and commit copies
  `0.424ms`. This prototype is not production-accepted; it remains guarded
  for token/logit correctness debugging. No requests, speed benchmark,
  decode/HCS work, production enablement, or default-path change occurred.

- Opened Nemotron Nano internal reference-design gate
  `20260628_1630_nemotron_nano_parallel_prefill_reference_design`.
  Recorded `1348` first: the guarded state-parallel recurrent prototype
  failed correctness and was not accepted. Baselines carried forward: Gemma
  `5619.6 / 92.43 / 155.69`; Nano `346.5 / 94.48 / 164.75`. Scope:
  static/internal reference analysis only; no implementation, build, runtime
  diagnostic, request, speed benchmark, decode/HCS work, production
  enablement, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD state-parallel recurrent
  correctness prototype gate
  `20260628_1348_nemotron_nano_mamba2_ssd_state_parallel_recurrent_correctness_prototype`.
  Implemented only guarded
  `KRASIS_MAMBA2_SSD_BLOCK_SCAN_STATE_PARALLEL_RECURRENT=1` inside the
  coeff-tile block-scan candidate path. Default sequential SSD, full-oracle
  path, candidate timing, recurrent subloop timing, `c_state_total_exact`,
  coeff-tile output, BF16 cast placement, row/chunk indexing, and probes were
  preserved; no affine summaries, accepted parallel C-dot,
  `local_old_state_exact`, production enablement, default-path change,
  request path, speed benchmark, decode/HCS work, unrelated cleanup, or
  Python one-liners were used. `./dev build` passed (`179s`). The smallest
  full-oracle coeff-tile startup diagnostic failed closed (`exit 1`, `41s`):
  output mismatches `5`, first output mismatch row82/head37/d30
  `0xbe4e` vs `0xbe4d`, output max/mean `0.25` /
  `0.00000025064218789339066`; state mismatches `134008`, first state
  mismatch flat14 `0xbb05886f` vs `0xbb05886e`, state max/mean
  `0.0003662109375` / `0.00000002278079460770008`. Row64/state0 remained
  exact; row82/state14 regressed; row97/head48/d24 term probe still differed
  by one BF16 bin while `c_state_total_exact_minus_replayed` was `0.0`. No
  long no-oracle candidate timing was run.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD state-parallel recurrent
  correctness prototype gate
  `20260628_1348_nemotron_nano_mamba2_ssd_state_parallel_recurrent_correctness_prototype`.
  Recorded `1335` first: static design selected guarded
  `KRASIS_MAMBA2_SSD_BLOCK_SCAN_STATE_PARALLEL_RECURRENT=1` as the next
  prototype target, preserving exact token-order f32 recurrence for every
  `(head,d,s)`, serial ascending-`s` C-dot for `c_state_total_exact`, the
  accepted coeff-tile output path, candidate timing, recurrent subloop
  timing, full-oracle path, row/chunk indexing, BF16 cast placement, probes,
  and default sequential SSD. Baselines carried forward: Gemma
  `5619.6 / 92.43 / 155.69`; Nano `346.5 / 94.48 / 164.75`. Scope:
  implement only the opt-in state-parallel recurrent candidate inside the
  coeff-tile block-scan path, build, run the smallest full-oracle coeff-tile
  startup diagnostic with row97/head48/d24 plus row64/state0 and
  row82/state14 probes, stop on mismatch, and only if exact run one long
  no-oracle candidate-timing diagnostic with recurrent subloop timing enabled.
  No affine summaries, accepted parallel C-dot reduction,
  `local_old_state_exact`, production enablement, default-path change,
  request, speed benchmark, decode/HCS work, unrelated cleanup, or Python
  one-liners.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD recurrent next-optimization
  design gate
  `20260628_1335_nemotron_nano_mamba2_ssd_recurrent_next_optimization_design`.
  Static-only result: choose a guarded state-parallel recurrent kernel as the
  next prototype target, with exact token-order f32 recurrence preserved for
  every `(head,d,s)` state slot and C-dot kept as a serial ascending-`s`
  accumulation for `c_state_total_exact`. The design targets the measured
  `1313` recurrent split: state update `503.074ms` (`69.301%`) first and
  C-dot `197.249ms` (`27.172%`) second. It explicitly rejects affine summaries,
  parallel C-dot reduction as an accepted source, and the failed
  `local_old_state_exact` surface. Existing coeff-tile output, D/prior, BF16
  cast placement, row/chunk indexing, candidate timing, full-oracle path,
  probes, and default sequential SSD are preserved by design. No
  implementation, build, startup diagnostic, benchmark, request, decode/HCS
  work, production enablement, or default-path change occurred.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD recurrent next-optimization
  design gate
  `20260628_1335_nemotron_nano_mamba2_ssd_recurrent_next_optimization_design`.
  Recorded `1313` first: guarded recurrent subloop timing preserved
  coeff-tile correctness and measured long no-oracle recurrent split as state
  update `503.074ms` (`69.301%`), C-dot `197.249ms` (`27.172%`), dt/A/x
  setup `16.022ms`, entry snapshot `4.760ms`, stores `0.471ms`, and residual
  `4.348ms`, with total candidate `1171.691ms`. Baselines carried forward:
  Gemma `5619.6 / 92.43 / 155.69`; Nano `346.5 / 94.48 / 164.75`. Scope:
  static design only; choose the next exactness-preserving target focusing
  first on the state-update loop and second on C-dot, while preserving current
  recurrence order, `c_state_total_exact`, coeff-tile correctness,
  candidate-timing mode, full-oracle path, row/chunk indexing, BF16 cast
  placement, probes, and default sequential SSD. No implementation, build,
  startup run, request, speed benchmark, decode/HCS work, production
  enablement, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD recurrent subloop timing gate
  `20260628_1313_nemotron_nano_mamba2_ssd_recurrent_subloop_timing`.
  Added only opt-in recurrent-kernel subloop timing behind
  `KRASIS_MAMBA2_SSD_BLOCK_SCAN_RECURRENT_SUBLOOP_TIMING=1` for the accepted
  coeff-tile block-scan path. Default sequential SSD, coeff-tile correctness,
  candidate-timing mode, full-oracle path, row probes, and defaults were
  preserved; no optimization was added. `./dev build` passed (`179s`). The
  smallest full-oracle coeff-tile startup diagnostic passed exact (`69/69`,
  `0` output/state/probe mismatches) with recurrent subloop timing enabled;
  row97/head48/d24, row64/state0, and row82/state14 probes were exact. The
  long no-oracle candidate-timing diagnostic ran with oracle and output
  subloop timing disabled, emitted `23` long recurrent/candidate records, and
  sent `0` requests. Long `39920` averages: total candidate `1171.691ms`,
  recurrent kernel `725.924ms`, coeff-tile output `431.187ms`, live state
  copy `12.886ms`, commit copies `0.428ms`. Recurrent split: state update
  `503.074ms` (`69.301%`), C-dot `197.249ms` (`27.172%`), dt/A/x setup
  `16.022ms` (`2.207%`), entry snapshot `4.760ms` (`0.656%`), stores
  `0.471ms` (`0.065%`), residual `4.348ms` (`0.599%`). No speed benchmark,
  decode/HCS work, production enablement, unrelated cleanup, or default-path
  change occurred.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD recurrent subloop timing gate
  `20260628_1313_nemotron_nano_mamba2_ssd_recurrent_subloop_timing`.
  Recorded `1305` first: clean no-oracle candidate timing showed recurrent
  kernel `716.036ms` average, coeff-tile output `431.422ms`, total candidate
  `1161.984ms`, and existing instrumentation could not split recurrent
  internals beyond launch/sync/total. Baselines carried forward: Gemma
  `5619.6 / 92.43 / 155.69`; Nano `346.5 / 94.48 / 164.75`. Scope: implement
  only guarded recurrent subloop timing
  `KRASIS_MAMBA2_SSD_BLOCK_SCAN_RECURRENT_SUBLOOP_TIMING=1` for the accepted
  coeff-tile block-scan path, preserve coeff-tile correctness,
  candidate-timing mode, full-oracle path, probes, defaults, and default
  sequential SSD, then run `./dev build`, a smallest full-oracle coeff-tile
  startup diagnostic with recurrent subloop timing enabled, and only if exact
  one long no-oracle candidate-timing diagnostic with recurrent subloop timing
  enabled and output subloop/oracle disabled. No optimization, requests, speed
  benchmark, decode/HCS work, production enablement, or unrelated cleanup.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD recurrent-kernel decomposition
  target gate
  `20260628_1305_nemotron_nano_mamba2_ssd_recurrent_kernel_decomposition_target`.
  Static-or-measurement result: existing `1242` instrumentation cannot split
  the recurrent kernel beyond launch/sync/total, so no build or startup
  diagnostic was run. Clean no-oracle `39920` candidate timing remains:
  recurrent `716.036ms` average, coeff-tile output `431.422ms`, total
  candidate `1161.984ms`. Static work counts from the recorded dimensions:
  `163,512,320` row-lanes, `20,929,576,960` state recurrence elements,
  `20,929,576,960` `c_state_total` dot elements, `327,155,712` entry snapshot
  elements, and `163,512,320` `c_state_total` stores. Recorded the minimal
  next change as guarded timing-only
  `KRASIS_MAMBA2_SSD_BLOCK_SCAN_RECURRENT_SUBLOOP_TIMING=1`, with a tiny
  `u64` timing buffer and per-thread register-accumulated `clock64` counters.
  No optimization, source change, build, startup diagnostic, requests, speed
  benchmark, decode/HCS work, production enablement, or default-path change
  occurred.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD recurrent-kernel decomposition
  target gate
  `20260628_1305_nemotron_nano_mamba2_ssd_recurrent_kernel_decomposition_target`.
  Recorded `1242` first: guarded coeff-tile candidate timing mode is accepted
  as measurement-only, full-oracle correctness was preserved, and clean
  no-oracle long timing over `39920` tokens showed recurrent kernel
  `716.036ms` average, coeff-tile output `431.422ms`, live state copy
  `12.858ms`, commit copies `0.436ms`, and negligible residual. Baselines
  carried forward: Gemma `5619.6 / 92.43 / 155.69`; Nano
  `346.5 / 94.48 / 164.75`. Scope: static-or-measurement analysis only to
  determine whether existing instrumentation can split recurrent-kernel cost
  enough to choose the next target; if not, record the smallest guarded
  recurrent-kernel decomposition timing plan. Preserve coeff-tile correctness,
  defaults, full-oracle path, probes, and candidate-timing mode. No
  optimization, requests, speed benchmark, decode/HCS work, production
  enablement, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD coeff-tile candidate-timing
  measurement gate
  `20260628_1242_nemotron_nano_mamba2_ssd_coeff_tile_candidate_timing_measurement`.
  Implemented only guarded measurement mode
  `KRASIS_MAMBA2_SSD_BLOCK_SCAN_CANDIDATE_TIMING=1` for the accepted
  coeff-tile block-scan path. The existing full-oracle path, default
  sequential SSD, v5 diagnostics/timing, coefficient-tile path, probes, and
  defaults were preserved; no optimization was added. `./dev build` passed
  (`172s`). The smallest full-oracle coeff-tile startup diagnostic passed
  exact (`69/69`, `0` output/state mismatches) with row97/head48/d24,
  row64/state0, and row82/state14 probes exact. The long no-oracle
  candidate-timing diagnostic ran with oracle/subloop/full-compare records
  absent and emitted visible `NO_ORACLE_TIMING_MODE` records. Long `39920`
  candidate timing across `23` records: total `26725.642ms` sum
  (`1161.984ms` avg), recurrent `16468.832ms` sum (`716.036ms` avg),
  coeff-tile output `9922.715ms` sum (`431.422ms` avg), live state copy
  `295.723ms` sum (`12.858ms` avg), commit copies `10.027ms` sum
  (`0.436ms` avg), residual `0.096ms` sum. No requests, speed benchmark,
  decode/HCS work, production enablement, or default-path change occurred.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD coeff-tile candidate-timing
  measurement gate
  `20260628_1242_nemotron_nano_mamba2_ssd_coeff_tile_candidate_timing_measurement`.
  Recorded `1233` first: existing envs cannot measure the accepted coeff-tile
  block-scan candidate without oracle overhead, and the minimal next change is
  explicit guarded `KRASIS_MAMBA2_SSD_BLOCK_SCAN_CANDIDATE_TIMING=1`.
  Baselines carried forward: Gemma `5619.6 / 92.43 / 155.69`; Nano
  `346.5 / 94.48 / 164.75`. Scope: implement only measurement mode for the
  accepted coeff-tile block-scan path, emit a visible no-oracle timing-mode
  record, preserve the full-oracle path and all defaults, then run
  `./dev build`, a smallest full-oracle coeff-tile startup diagnostic with
  candidate timing enabled, and only if exact one long candidate-timing
  diagnostic with oracle/subloop timing disabled. No optimization, requests,
  speed benchmark, decode/HCS work, production enablement, or default-path
  change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD coeff-tile no-oracle
  candidate-cost gate
  `20260628_1233_nemotron_nano_mamba2_ssd_coeff_tile_no_oracle_candidate_cost`.
  Static-or-measurement result: existing envs do not support the requested
  no-oracle coeff-tile candidate timing. `KRASIS_MAMBA2_SSD_BLOCK_SCAN=1`
  currently requires `KRASIS_MAMBA2_SSD_BLOCK_SCAN_ORACLE=1`, and the current
  `[MAMBA2-SSD-BLOCK-SCAN-TIMING]` record is emitted after sequential oracle,
  full buffer downloads, host comparison, and commit copies. No build or
  startup diagnostic was run because the requested measurement path is
  unsupported as-is. Recorded the smallest next change: add explicit guarded
  `KRASIS_MAMBA2_SSD_BLOCK_SCAN_CANDIDATE_TIMING=1` to skip oracle buffers,
  sequential oracle, full host compare/downloads, term-probe downloads, and
  subloop timing while emitting candidate-only recurrent/output/copy/residual
  timing. No source change, request, speed benchmark, decode/HCS work,
  production enablement, or default-path change occurred.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD coeff-tile no-oracle
  candidate-cost gate
  `20260628_1233_nemotron_nano_mamba2_ssd_coeff_tile_no_oracle_candidate_cost`.
  Recorded `1207` first: guarded coefficient tiling is correctness-accepted
  for the opt-in block-scan candidate path, with `138/138` exact oracle
  records and long `39920` timing of total block-scan path `7179.206ms`,
  sequential oracle `4960.685ms`, recurrent kernel `715.770ms`, coeff-tile
  output kernel `500.629ms`, downloads `391.735ms`, host comparison
  `537.322ms`, and commit copies `0.489ms`. Baselines carried forward:
  Gemma `5619.6 / 92.43 / 155.69`; Nano `346.5 / 94.48 / 164.75`.
  Scope: establish candidate-side long timing without sequential oracle,
  host compare, or subloop instrumentation overhead if existing envs support
  it; otherwise produce the smallest guarded timing-change plan. No new
  optimization, requests, speed benchmark, decode/HCS work, production
  enablement, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD opt-in coefficient-tile
  correctness prototype gate
  `20260628_1207_nemotron_nano_mamba2_ssd_block_scan_coefficient_tile_correctness_prototype`.
  Implemented only guarded per-pair coefficient tiling in the opt-in
  block-scan candidate path behind
  `KRASIS_MAMBA2_SSD_BLOCK_SCAN_COEFF_TILE`; default sequential SSD, v5
  diagnostics/timing, `c_state_total_exact`, D/prior, BF16 cast placement,
  row/chunk indexing, and output accumulation order were preserved. The
  rejected `local_old_state_exact` and affine summaries were not
  reintroduced. `./dev build` passed. Small and long block-scan full-oracle
  startup diagnostics passed exact (`138/138` total oracle records, `0`
  output/state mismatches); row97/head48/d24 term probes matched
  inline-vs-tiled coefficients exactly, and row64/state0 plus row82/state14
  probes were exact. Long `39920`-token timing improved versus `1128`:
  total block-scan path `7179.206ms` vs `10998.270ms` (`-34.724%`), output
  kernel `500.629ms` vs `4318.312ms` (`-88.407%`). Remaining output-kernel
  split: coefficient build `231.740ms`, residual `239.931ms`, D/prior
  `14.491ms`, local-old `5.321ms`, local triangular `8.425ms`, cast/store
  `0.720ms`. No requests, speed benchmark, decode/HCS work, production
  enablement, unrelated cleanup, or default-path change occurred.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD opt-in coefficient-tile
  correctness prototype gate
  `20260628_1207_nemotron_nano_mamba2_ssd_block_scan_coefficient_tile_correctness_prototype`.
  Recorded `1157` first: the next exactness-preserving target is guarded
  per-pair coefficient tiling inside the block-scan output path, preserving
  `c_state_total_exact`, D/prior, BF16 cast placement, row/chunk indexing,
  output accumulation order, default sequential SSD, v5 diagnostics/timing,
  and the accepted `1022/1113` correctness surface. Baselines: Gemma
  `5619.6 / 92.43 / 155.69`; Nano `346.5 / 94.48 / 164.75`. Scope:
  implement only guarded coefficient tiling in the opt-in block-scan candidate
  path; do not reintroduce `local_old_state_exact`, affine summaries,
  production enablement, default-path changes, requests, speed benchmark,
  decode/HCS work, or unrelated cleanup. Required validation: `./dev build`,
  smallest block-scan full-oracle startup diagnostic with row97/head48/d24,
  row64/state0, and row82/state14 probes; only if exact, run one long
  startup diagnostic with component/subloop timing against `1128`.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD static next-optimization design
  gate
  `20260628_1157_nemotron_nano_mamba2_ssd_block_scan_next_optimization_design`.
  Static-only; no implementation, build, runtime diagnostic, benchmark,
  request, decode/HCS work, production enablement, or default-path change.
  The selected next exactness-preserving target is a guarded coefficient-tiled
  block-scan output path: precompute per-pair local triangular and local-old
  scalar coefficients with the same state-loop order and BF16 casts, then keep
  output assembly in the current `u` order. This targets the measured
  `1128` costs, local triangular `2158.513ms` plus local-old cancellation
  `1845.641ms`, while preserving D/prior setup, BF16 cast placement,
  row/chunk indexing, default sequential SSD, v5 diagnostics/timing, and the
  accepted `1022/1113` `c_state_total_exact` correctness surface. Required
  next probes: row97/head48/d24, row64/state0, row82/state14, coefficient
  inline-vs-tile deltas, and full exact BF16/f32 oracle. Final validation
  passed with GPUs idle at `15 MB / 11 MB`.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD static next-optimization design
  gate
  `20260628_1157_nemotron_nano_mamba2_ssd_block_scan_next_optimization_design`.
  Recorded `1128` first: guarded output-subloop timing preserved exact
  block-scan correctness (`69/69` small and long oracle records exact, `0`
  output/state mismatches) and measured long output-kernel averages of
  local triangular `2158.513ms`, local-old cancellation `1845.641ms`,
  residual `291.516ms`, D/prior setup `21.825ms`, and cast/store `0.817ms`.
  Baselines carried forward: Gemma `5619.6 / 92.43 / 155.69`; Nano
  `346.5 / 94.48 / 164.75`. Scope: static-only design for the next
  exactness-preserving optimization target, without changing D/prior, BF16
  cast placement, row/chunk indexing, default SSD, v5 diagnostics, or the
  accepted `1022/1113` correctness surface. No implementation, build,
  runtime diagnostic, benchmark, request, decode/HCS work, production
  enablement, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD guarded block-scan output
  subloop timing gate
  `20260628_1128_nemotron_nano_mamba2_ssd_block_scan_output_subloop_timing`.
  Added only opt-in instrumentation behind
  `KRASIS_MAMBA2_SSD_BLOCK_SCAN_OUTPUT_SUBLOOP_TIMING` to split
  `block_scan_output_kernel`; default sequential SSD, accepted block-scan
  correctness, v5 diagnostics/timing, and all row probes were preserved.
  `./dev build` passed (`179s`). The smallest block-scan full-oracle startup
  diagnostic passed (`69/69` exact), so one long startup diagnostic was run;
  it also passed exact (`69/69`, `23` long records at `39920` tokens, `0`
  output/state mismatches). Long average component timing: block-scan path
  `10998.270ms`, sequential oracle `4962.651ms`, recurrent kernel
  `715.617ms`, output kernel `4318.312ms`, downloads `391.968ms`, host
  comparison `536.393ms`. Output-kernel subloop split: local triangular
  `2158.513ms`, local-old cancellation `1845.641ms`, residual `291.516ms`,
  D/prior setup `21.825ms`, cast/store `0.817ms`. No requests, speed
  benchmark, decode/HCS work, production enablement, or default-path change
  occurred. Final validation passed with GPUs idle at `15 MB / 11 MB`.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD guarded block-scan output
  subloop timing gate
  `20260628_1128_nemotron_nano_mamba2_ssd_block_scan_output_subloop_timing`.
  Recorded `1113` first: cleanup restored the accepted `1022`
  `c_state_total_exact` correctness surface, with block-scan full oracle
  `69/69` exact, `0` output/state mismatches, and row97/head48/d24,
  row64/state0, and row82/state14 probes exact. Baselines carried forward:
  Gemma `5619.6 / 92.43 / 155.69`; Nano `346.5 / 94.48 / 164.75`.
  Scope: add only opt-in timing instrumentation under the block-scan env to
  split `block_scan_output_kernel` into D/prior setup, local-old cancellation,
  local triangular accumulation, cast/store, and residual/sync timing while
  preserving the accepted correctness surface and all probes. Run
  `./dev build`, then the smallest block-scan full-oracle startup diagnostic;
  only if exact, run one long diagnostic for the split. No requests, speed
  benchmark, decode/HCS work, production enablement, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD
  `local_old_state_exact` failed-cleanup gate
  `20260628_1113_nemotron_nano_mamba2_ssd_local_old_state_exact_failed_cleanup`.
  Reverted only the failed `1058` local-old-state materialization and its stale
  probe fields from the opt-in block-scan path. Preserved the accepted `1022`
  `c_state_total_exact` correction, default sequential SSD, v5
  diagnostics/timing, block-scan env gates, D skip, BF16 cast placement,
  row/chunk indexing, and local triangular math. `./dev build` passed
  (`177s`). The smallest block-scan full-oracle startup diagnostic passed
  (`exit 0`, `47s`): `69/69` oracle records exact, `0` output mismatches,
  `0` state mismatches, and row97/head48/d24, row64/state0, and row82/state14
  probes exact. No long diagnostic was needed; no requests, speed benchmark,
  decode/HCS work, production enablement, or default-path change occurred.
  Final validation passed with GPUs idle at `15 MB / 11 MB`.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD
  `local_old_state_exact` failed-cleanup gate
  `20260628_1113_nemotron_nano_mamba2_ssd_local_old_state_exact_failed_cleanup`.
  Recorded `1058` first: the guarded opt-in `local_old_state_exact`
  materialization fixed row97/head48/d24 but failed full oracle with `10`
  output mismatches, first `row=82 head=37 d=30` (`0xbe4e` vs `0xbe4d`),
  while state stayed exact (`0` mismatches). Baselines carried forward:
  Gemma `5619.6 / 92.43 / 155.69`; Nano `346.5 / 94.48 / 164.75`.
  Scope: revert only `local_old_state_exact` materialization and its probe
  fields from the block-scan candidate path; preserve the accepted `1022`
  `c_state_total_exact` correction, default sequential SSD, v5
  diagnostics/timing, block-scan env gates, D skip, BF16 cast placement,
  row/chunk indexing, and local triangular math. Run `./dev build`, then the
  smallest block-scan full-oracle startup diagnostic with row97/head48/d24,
  row64/state0, and row82/state14 probes. No requests, speed benchmark,
  decode/HCS work, production enablement, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD
  `local_old_state_exact` correction prototype gate
  `20260628_1058_nemotron_nano_mamba2_ssd_local_old_state_exact_correction_prototype`.
  Implemented only the guarded opt-in block-scan correction:
  `local_old_state_exact[L,n_heads,head_dim]` is materialized from the exact
  recurrent phase and consumed by block-scan output assembly. Default
  sequential SSD remains unchanged; v5 diagnostics/timing,
  `c_state_total_exact`, D skip, BF16 cast placement, row/chunk indexing, and
  local triangular math were preserved. `./dev build` passed (`178s`). The
  smallest startup diagnostic failed closed (`exit 1`), so no long diagnostic
  ran. Row97/head48/d24 term probe matched candidate/oracle bits (`0xc234`),
  and row64/state0 remained exact, but the full oracle reported `10` output
  mismatches with first mismatch `row=82 head=37 d=30` (`0xbe4e` vs `0xbe4d`),
  output max/mean `0.00390625 / 0.000000007927155820652843`. State remained
  exact (`0` mismatches). No requests, speed benchmark, decode/HCS work,
  production enablement, or default-path change occurred.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD
  `local_old_state_exact` correction prototype gate
  `20260628_1058_nemotron_nano_mamba2_ssd_local_old_state_exact_correction_prototype`.
  Recorded `1049` first: static decomposition of the corrected block-scan
  output kernel found the measured `100060.325232ms` output assembly dominated
  by the lower-triangular loop, with `local_old_state_exact` identified as the
  next guarded diagnostic/candidate buffer but not statically accepted because
  recurrent materialization may change f32 accumulation order. Baselines
  carried forward: Gemma `5619.6 / 92.43 / 155.69`; Nano
  `346.5 / 94.48 / 164.75`. Scope: implement only opt-in materialization of
  `local_old_state_exact[L,n_heads,head_dim]` from the exact recurrent phase
  and feed it to block-scan output assembly; preserve default sequential SSD,
  v5 diagnostics/timing, `c_state_total_exact`, D skip, BF16 cast placement,
  row/chunk indexing, and local triangular math. Run `./dev build`, then the
  smallest full-oracle startup diagnostic; run one long component-timing
  diagnostic only if exact correctness passes. No requests, speed benchmark,
  decode/HCS work, production enablement, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD `1022` block-scan
  output-kernel decomposition gate
  `20260628_1049_nemotron_nano_mamba2_ssd_1022_block_scan_output_kernel_decomposition`.
  Static-only result: the `100060.325232ms` output kernel is dominated by the
  lower-triangular output loop, not D skip, c-state prior read, final BF16
  store, or row/chunk indexing. `c_state_total_exact` is now a cheap accepted
  f32 input from the recurrent phase; the remaining heavy work is
  `local_old_state` cancellation plus unchanged local HF triangular scan.
  The next correction should add a guarded `local_old_state_exact` diagnostic
  buffer from the recurrent phase and compare it against the current
  output-order value before consuming it. Static artifacts cannot prove that
  recurrent materialization is bit-exact because accumulation order may change,
  so the next gate must keep exact full-oracle pass/fail and row97/head48/d24
  plus row64/state0 and row82/state14 probes. No implementation, build,
  runtime diagnostic, benchmark, request, decode/HCS work, production
  enablement, or default-path change occurred.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD `1022` block-scan
  output-kernel decomposition gate
  `20260628_1049_nemotron_nano_mamba2_ssd_1022_block_scan_output_kernel_decomposition`.
  Recorded `1022` first: the opt-in `c_state_total_exact` correction passed
  exact small and long full-oracle diagnostics (`69/69`, `0` output/state
  mismatches), with row97/head48/d24 term probes exact and row64/state0 plus
  row82/state14 regression probes exact. Long component timing showed total
  block-scan path `253940.599995ms`, sequential oracle `114284.518733ms`,
  recurrent kernel `16487.771138ms`, output kernel `100060.325232ms`,
  downloads `9055.823995ms`, host comparison `12359.669588ms`, and commit
  copies `11.014996ms`. Baselines carried forward: Gemma
  `5619.6 / 92.43 / 155.69`; Nano `346.5 / 94.48 / 164.75`. Scope:
  static-only decomposition of the output kernel loop structure: D skip, prior
  assembly from `c_state_total_exact`, local old-state cancellation, local
  triangular accumulation, cast placement, and row/chunk indexing. No
  implementation, build, runtime, benchmark, request, decode/HCS work,
  production enablement, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD block-scan
  `c_state_total_exact` correction prototype gate
  `20260628_1022_nemotron_nano_mamba2_ssd_block_scan_c_state_total_correction_prototype`.
  Implemented only the recorded opt-in correction: the exact recurrent phase
  now materializes `c_state_total_exact[L,n_heads,head_dim]`, and the
  block-scan output kernel consumes that f32 buffer instead of recomputing
  `c_state_total` before `local_old_state` cancellation. Default sequential
  SSD remains unchanged; v5 diagnostics/timing, `2149`/`2251`/`0650` guarded
  diagnostics, D skip, local triangular math, and cast placement were
  preserved. `./dev build` passed (`178s`). The smallest startup diagnostic
  passed exact full oracle (`69/69`, `0` output/state mismatches), so one long
  startup diagnostic was run and also passed exact full oracle (`69/69`,
  `0` output/state mismatches). Row97/head48/d24 term probes matched oracle
  output bits in all records; row64/state0 and row82/state14 regression probes
  were exact. Long component timing: total block-scan path `253940.599995ms`;
  sequential oracle `114284.518733ms`; recurrent kernel `16487.771138ms`;
  output kernel `100060.325232ms`; downloads `9055.823995ms`; host comparison
  `12359.669588ms`; commit copies `11.014996ms`. No requests, speed benchmark,
  decode/HCS work, production enablement, or default-path change occurred.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD block-scan
  `c_state_total_exact` correction prototype gate
  `20260628_1022_nemotron_nano_mamba2_ssd_block_scan_c_state_total_correction_prototype`.
  Recorded `1014` first: row97 output mismatch analysis attributed the
  opt-in block-scan candidate's `row=97 head=48 d=24` one-BF16-bin drift to
  output assembly recomputing `c_state_total` from `entry_state` and then
  cancelling `local_old_state`, while final state and row64/state0 plus
  row82/state14 probes were exact. Baselines carried forward: Gemma
  `5619.6 / 92.43 / 155.69`; Nano `346.5 / 94.48 / 164.75`. Scope:
  implement only the recorded opt-in correction by materializing
  `c_state_total_exact` in the exact recurrent phase and feeding that buffer
  into block-scan output assembly; preserve default sequential SSD, v5
  diagnostics/timing, prior guarded probes, D skip, local triangular math, and
  cast placement. Run `./dev build`, then the smallest full-oracle startup
  diagnostic; run one long startup diagnostic with component timing only if
  exact correctness passes. No requests, speed benchmark, decode/HCS work,
  production enablement, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD `0957` block-scan output mismatch
  analysis gate
  `20260628_1014_nemotron_nano_mamba2_ssd_0957_block_scan_output_mismatch_analysis`.
  Static-only result: row97 maps to chunk `1`, position `33`, head `48`, group
  `6`, d `24`; final state exactness rules out the accepted recurrent
  state/final-state path. D skip, row/chunk indexing, and the known `0650`
  BF16 cast-placement issue are not the leading suspects. The primary static
  suspect is the separate output kernel recomputing `c_state_total` from
  `entry_state` and then cancelling `local_old_state`, rather than consuming a
  `c_state_total` value materialized by the exact recurrent phase in the same
  order as the sequential oracle. Correction plan: add row97 term probes, then
  materialize `c_state_total_exact[L,n_heads,head_dim]` in the recurrent phase
  and feed it to output assembly while leaving local triangular math unchanged.
  No cleanup, implementation, build, runtime diagnostic, benchmark, request,
  decode/HCS work, production enablement, or default-path change occurred.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD `0957` block-scan output
  mismatch analysis gate
  `20260628_1014_nemotron_nano_mamba2_ssd_0957_block_scan_output_mismatch_analysis`.
  Recorded `0957` first: the opt-in block-scan candidate built successfully
  and failed closed on the smallest startup diagnostic with `3` output
  mismatches, first mismatch `row=97 head=48 d=24` (`0xc233` vs `0xc234`,
  abs `0.25`), while final state matched exactly and row64/state0 plus
  row82/state14 probes were exact. Baselines carried forward: Gemma
  `5619.6 / 92.43 / 155.69`; Nano `346.5 / 94.48 / 164.75`. Scope:
  static analysis only unless retained artifacts cannot answer the required
  row97 output-assembly question; focus on D skip, exact-entry prior, local
  triangular accumulation order, BF16 cast placement, and row/chunk/window
  indexing. No cleanup, implementation, build, runtime diagnostic, benchmark,
  request, decode/HCS work, production enablement, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD opt-in block-scan correctness
  prototype gate
  `20260628_0957_nemotron_nano_mamba2_ssd_block_scan_correctness_prototype`.
  Implemented only an opt-in candidate behind `KRASIS_MAMBA2_SSD_BLOCK_SCAN`
  plus required `KRASIS_MAMBA2_SSD_BLOCK_SCAN_ORACLE`; default SSD remains the
  existing sequential path, and v5 diagnostics/timing plus `2149`/`2251`/`0650`
  guarded probes were preserved. The candidate keeps exact token-order f32
  recurrence for chunk-entry/final state, stores per-chunk entry snapshots, and
  moves local/output assembly into a separate candidate kernel. `./dev build`
  passed (`173s`). The smallest startup diagnostic failed closed on layer 0:
  `3` BF16 output mismatches, max/mean `0.25 / 0.00000024959445`, first
  mismatch `row=97 head=48 d=24` (`0xc233` vs `0xc234`); final state matched
  exactly with `0` mismatches, and row64/state0 plus row82/state14 probes were
  exact. No long diagnostic, requests, speed benchmark, decode/HCS work,
  production enablement, or default-path change occurred.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD opt-in block-scan correctness
  prototype gate
  `20260628_0957_nemotron_nano_mamba2_ssd_block_scan_correctness_prototype`.
  Recorded `0944` first: the static design says to keep exact token-order f32
  recurrence for chunk/window entry and final state, avoid affine summaries as
  accepted state, and move unchanged local/output assembly into separate
  candidate buffers. Baselines carried forward: Gemma
  `5619.6 / 92.43 / 155.69`; Nano `346.5 / 94.48 / 164.75`. Scope: opt-in
  candidate only; preserve default sequential SSD, v5 diagnostics/timing
  instrumentation, and all prior guarded probes; run `./dev build`, then the
  smallest full-oracle startup diagnostic with row64/state0 plus row82/state14
  probes; run one long startup diagnostic with component timing only if exact
  correctness passes. No requests, speed benchmark, decode/HCS work,
  production enablement, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD v5 block-scan parallel design
  gate `20260628_0944_nemotron_nano_mamba2_ssd_v5_block_scan_parallel_design`.
  Static-only result: the next optimization should not use compact affine P/Q
  summaries as the accepted state source, because earlier v4/v5 evidence shows
  that changing f32 recurrence rounding order breaks exact state bits. The
  concrete plan is a two-phase/block-scan candidate that precomputes
  sequential-compatible `dt`, `A_bar`, `BF16(B*dt)`, and chunk prefix data;
  parallelizes unchanged local/D/triangular output work; propagates chunk
  entries with an exact recurrent block/window spine using the same
  `h=A_bar*h+BF16(B*dt)*x` token order; assembles prior output from exact entry
  states; and then requires exact BF16 output plus exact f32 state oracle before
  commit. No implementation, build, runtime diagnostic, benchmark, request,
  decode/HCS work, production enablement, or default-path change occurred.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD v5 block-scan parallel design
  gate `20260628_0944_nemotron_nano_mamba2_ssd_v5_block_scan_parallel_design`.
  Recorded `0835` first: opt-in v5 exact oracle remained clean, and the
  39,920-token timing decomposition showed `ssd_scan 251443.8ms`, total v5
  path `251386.108393ms`, sequential oracle `114292.932890ms`, v5 candidate
  `114293.155150ms`, downloads `9043.142551ms`, host comparison
  `12390.200665ms`, and commit copies `11.745775ms`. Baselines carried
  forward: Gemma `5619.6 / 92.43 / 155.69`; Nano
  `346.5 / 94.48 / 164.75`. Scope: static design only; determine a concrete
  two-phase/block-scan parallelization plan that preserves v5
  sequential-compatible entry-state bits without a full serial per-token
  candidate kernel. No implementation, build, runtime, benchmark, request,
  decode/HCS work, production enablement, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD v5 timing decomposition gate
  `20260628_0835_nemotron_nano_mamba2_ssd_v5_timing_decomposition`.
  Added guarded v5-only component timing; default SSD remains sequential and v5
  is not production-enabled. `./dev build` passed (`169s`). Startup diagnostic
  with full v5 oracle reached READY and was stopped after capture because the
  diagnostic exit flag did not exit; it sent `0` requests and ran no benchmark
  path. Correctness remained exact: `207` oracle records, `0` output
  mismatches, `0` state mismatches. Long 39,920-token decomposition:
  `ssd_scan 251443.8ms`, total v5 path `251386.108393ms`, sequential oracle
  `114292.932890ms` (`45.455%`), v5 candidate `114293.155150ms` (`45.455%`),
  full-buffer downloads `9043.142551ms` (`3.596%`), host comparison
  `12390.200665ms` (`4.928%`), commit copies `11.745775ms` (`0.0047%`).
  Conclusion: the v5 diagnostic regression is dominated by sequential-oracle
  plus candidate double execution; downloads/comparison are secondary but
  material, and commit copies are negligible.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD v5 timing decomposition gate
  `20260628_0835_nemotron_nano_mamba2_ssd_v5_timing_decomposition`.
  Recorded `0810` first: the opt-in v5 correctness prototype passed exact
  full oracle in small and long startup diagnostics, but the long diagnostic
  `ssd_scan` was `244701.0ms` versus `1908` baseline `114449.4ms` because the
  timed path included sequential oracle, v5 candidate, full host comparison,
  and commit/copy work. Baselines carried forward: Gemma
  `5619.6 / 92.43 / 155.69`; Nano `346.5 / 94.48 / 164.75`. Scope: do not
  production-enable v5 or change the default sequential SSD path; add only
  guarded component timing under the v5 envs if existing logs cannot separate
  sequential oracle, candidate kernel, buffer downloads, comparison, and commit
  copies. Run `./dev build`, then one minimal startup diagnostic with full
  oracle enabled; only if component timing is clean and correctness remains
  exact, run one long startup diagnostic for decomposition. No requests, speed
  benchmark, decode/HCS work, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD v5 correctness prototype gate
  `20260628_0810_nemotron_nano_mamba2_ssd_v5_correctness_prototype`.
  Implemented only an opt-in candidate behind
  `KRASIS_MAMBA2_SSD_CHUNK_PARALLEL_V5` plus required full-oracle env
  `KRASIS_MAMBA2_SSD_CHUNK_PARALLEL_V5_ORACLE`; default SSD remains the
  existing sequential path. The candidate uses sequential-compatible per-token
  f32 recurrence for state propagation, separate oracle/candidate output and
  state buffers, and commits live `ssm_state` only after exact full-oracle
  success. Preserved `1908` Mamba2 substage timing plus `2149`/`2251`/`0650`
  diagnostics. `./dev build` passed (`176s`). Small startup diagnostic and one
  allowed long startup diagnostic both exited `0`; each produced `69/69` v5
  oracle records with `0` output mismatches and `0` state mismatches across
  row64/state0 plus row82/state14 probes. No requests or benchmark path ran.
  Long diagnostic `ssd_scan` was `244701.0ms` versus `1908` baseline
  `114449.4ms` (`+130251.6ms`, `+113.807%`), so v5 is correctness-accepted as
  an opt-in diagnostic seed but not optimization-accepted.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD v5 correctness prototype gate
  `20260628_0810_nemotron_nano_mamba2_ssd_v5_correctness_prototype`.
  Recorded `0802` first: the design-only gate concluded that v5 must use
  sequential-compatible per-token f32 recurrence for chunk-entry/state
  propagation, while keeping local output math unchanged. Baselines carried
  forward: Gemma `5619.6 / 92.43 / 155.69`; Nano
  `346.5 / 94.48 / 164.75`. Scope: opt-in candidate only; default SSD remains
  sequential; preserve `1908` substage timing and `2149`/`2251`/`0650`
  diagnostics; use separate oracle/candidate output and state buffers; commit
  live state only after full oracle success; build and run the smallest startup
  diagnostic with row64/state0 plus row82/state14 probes. No requests, speed
  benchmark, decode/HCS work, production enablement, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD v5 sequential-compatible
  chunk-entry design gate
  `20260628_0802_nemotron_nano_mamba2_ssd_v5_sequential_compatible_chunk_entry_design`.
  Static analysis only; no implementation, build, startup run, benchmark,
  request, decode/HCS work, production enablement, or default-path change.
  Result: v5 should not revive closed-form chunk-state accumulation for
  accepted output/state. It should use opt-in recurrent-entry-state buffers
  that replay the current sequential update order
  `dt=softplus(dt+bias); A_bar=exp(A*dt); B_bar=BF16(B*dt); h=A_bar*h+B_bar*x`
  in f32 within each chunk. Output/local math stays unchanged: D skip plus
  prior from the recurrent chunk-entry state plus the existing local triangular
  `BF16(CB*decay*dt)*x` term. The design specifies separate oracle/candidate
  buffers, read-only chunk-entry snapshots, diagnostic-only closed-form
  comparison buffers, row64/state0 and row82/state14 probes, exact BF16/f32
  full-oracle comparison, and candidate-only commit after oracle success.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD v5 sequential-compatible
  chunk-entry design gate
  `20260628_0802_nemotron_nano_mamba2_ssd_v5_sequential_compatible_chunk_entry_design`.
  Recorded `0753` first: failed v4 candidate had already been cleaned up,
  default sequential SSD remained confirmed, no runtime/build/benchmark/request
  work ran in the mismatch-analysis gate, and the key finding was that row64
  drift came from candidate entry-state/output-prior assembly using
  closed-form chunk0 state accumulation rather than the current sequential
  per-token f32 recurrence. Baselines carried forward: Gemma
  `5619.6 / 92.43 / 155.69`; Nano `346.5 / 94.48 / 164.75`. Scope:
  static-only v5 design using the `0753` next-plan/probe artifacts, `0715`
  oracle artifacts, `0650` cast/carry diagnostic, and sequential SSD kernel.
  No implementation, build, runtime, benchmark, request, decode/HCS work,
  production enablement, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD v4 full-oracle mismatch analysis
  gate `20260628_0753_nemotron_nano_mamba2_ssd_v4_full_oracle_mismatch_analysis`.
  Static analysis only; no implementation, build, startup run, benchmark,
  request, decode/HCS work, production enablement, or default-path change.
  Result: the `0715` row64 preflights matched because they proved local-tri
  output and cast placement, but not candidate entry-state bit identity. The
  v4 candidate state flat0 value `0.004417699761688709` / `0x3b90c25a` matches
  the closed-form chunk0 accumulation shape, while the current sequential
  oracle exact state is `0.004417698830366135` / `0x3b90c258` from per-token
  recurrent f32 updates. The one-BF16-bit row64 drift is therefore attributed
  to candidate state buffer lifecycle / output prior assembly, not
  reference-style cast placement, local triangular math, or final-state
  writeback. Next plan: use recurrent-entry-state buffers that replay the
  sequential `A_bar*h + BF16(B*dt)*x` f32 update order within each chunk, with
  explicit probes for row64/state0 plus row82/state14 before full oracle.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD v4 full-oracle mismatch analysis
  gate `20260628_0753_nemotron_nano_mamba2_ssd_v4_full_oracle_mismatch_analysis`.
  Recorded `0737` first: failed `0715` v4 prototype additions were cleaned up,
  default sequential SSD was confirmed by `./dev build` plus startup diagnostic,
  HQQ attention/effective Marlin `g64` cache reuse was clean, no requests or
  benchmark path ran, and `1908`/`2149`/`2251`/`0650` diagnostics were
  preserved. Baselines carried forward: Gemma `5619.6 / 92.43 / 155.69`; Nano
  `346.5 / 94.48 / 164.75`. Scope: static analysis of the `0715`
  oracle/log/source artifacts, `0650` cast/carry result, and sequential SSD
  math around `row=64`, head `11`, d `22`, state flat `0`; determine why
  preflights matched but full oracle failed by one BF16 bit and two f32 bits.
  No implementation, build, startup run, benchmark, request, decode/HCS work,
  production enablement, or default-path change unless existing artifacts cannot
  answer the question.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD v4 failed-prototype cleanup gate
  `20260628_0737_nemotron_nano_mamba2_ssd_v4_failed_cleanup`. Reverted only
  the failed `0715` v4 additions: CUDA candidate kernel, Rust v4 env
  gates/temp buffers/oracle/dispatch/PTX symbol checks, and the v4-specific
  Python warmup fail-closed handling. Preserved `1908` substage timing, all
  artifacts, and the `2149`/`2251`/`0650` guarded diagnostics. `./dev build`
  passed (`exit 0`, `177s`). Minimal startup diagnostic without v4 prototype
  envs exited `0`, sent `0` requests, ran no benchmark path, reused HQQ
  attention/effective Marlin `g64`, and confirmed default sequential SSD with
  long `ssd_scan 114404.3ms` over `23` calls. Final state clean: no
  tmux/runtime/build process, no NVIDIA compute process, no Krasis temp/lock
  files outside HF download lock metadata, GPUs `15 MB / 11 MB`.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD v4 failed-prototype cleanup gate
  `20260628_0737_nemotron_nano_mamba2_ssd_v4_failed_cleanup`. Recorded
  `0715` first: the v4 candidate was opt-in only, default SSD remained
  sequential, `./dev build` passed, the smallest startup diagnostic failed
  closed before any long timing, cache reuse was clean, and no requests or
  benchmark path ran. Baselines carried forward: Gemma
  `5619.6 / 92.43 / 155.69`; Nano `346.5 / 94.48 / 164.75`. Scope: remove
  only the v4 correctness prototype additions, including the CUDA candidate
  kernel, Rust v4 env gates/temp buffers/oracle/dispatch/PTX symbol checks,
  and v4-specific Python warmup fail-closed handling. Preserve `1908`
  substage timing, all artifacts, and the `2149`/`2251`/`0650` guarded
  diagnostics. Validation plan: `./dev build` plus one minimal startup
  diagnostic without v4 prototype envs; no requests, speed benchmark,
  decode/HCS work, production enablement, or default-path change.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD v4 correctness prototype gate
  `20260628_0715_nemotron_nano_mamba2_ssd_v4_correctness_prototype`.
  Recorded `0650` first: guarded cast/carry diagnostic only, default
  sequential SSD unchanged, no optimized path added, final `./dev build`
  passed, startup diagnostic exited `0`, cache reuse was clean, and sequential
  cast placement `exp_decay * BF16(B*dt) * x` matched current GPU oracle bits
  while reference-style `BF16(B*exp_decay*dt) * x` did not. Baselines carried
  forward: Gemma `5619.6 / 92.43 / 155.69`; Nano
  `346.5 / 94.48 / 164.75`. Scope: implement only an opt-in v4 candidate with
  separate candidate/oracle buffers, preserve the `0650` diagnostic, keep the
  default SSD path sequential, and run `./dev build` plus the smallest startup
  diagnostic with full oracle enabled. Stop on any output/state mismatch and
  record first mismatch plus max/mean errors; run one long startup diagnostic
  only if correctness passes. No requests, speed benchmark, decode/HCS work,
  production enablement, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD v4 correctness prototype gate
  `20260628_0715_nemotron_nano_mamba2_ssd_v4_correctness_prototype`.
  Implemented only an opt-in v4 candidate behind
  `KRASIS_MAMBA2_SSD_CHUNK_PARALLEL_V4` with required oracle env
  `KRASIS_MAMBA2_SSD_CHUNK_PARALLEL_V4_ORACLE`; default SSD remains the
  existing sequential path and the `0650` cast/carry diagnostic is preserved.
  `./dev build` passed (`exit 0`, `176s`). The smallest startup diagnostic
  exited `1` by fail-closed oracle mismatch, sent `0` requests, ran no
  benchmark path, reused HQQ attention/effective Marlin `g64`, and did not run
  the long diagnostic. Preflights passed: local-tri row `64`, head `11`, d
  `22`, state flat `0` matched sequential GPU bits, and sequential-compatible
  cast placement matched while reference-style cast did not. Full oracle failed:
  first output mismatch `row=64`, head `11`, d `22`, candidate `0x3bb7` versus
  sequential `0x3bb8`, output mismatches `55`, max/mean
  `2.0 / 0.000009420169817531132`; first state mismatch flat `0`, candidate
  `0x3b90c25a` versus sequential `0x3b90c258`, state mismatches `506113`,
  max/mean `0.0048828125 / 0.0000016492683093835812`. Long timing was skipped
  because correctness failed.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD v4 cast/carry diagnostic gate
  `20260628_0650_nemotron_nano_mamba2_ssd_v4_cast_carry_diagnostic`.
  Added guarded diagnostic instrumentation only, behind
  `KRASIS_MAMBA2_SSD_V4_CAST_CARRY_DIAG`, reusing the existing local-tri
  target envs. Default sequential SSD dispatch remains unchanged and no
  optimized path was added. Final `./dev build` passed (`exit 0`, `170s`).
  Startup diagnostic exited `0`, sent `0` requests, ran no benchmark path,
  reused HQQ attention and effective Marlin `g64`, and emitted warmup/short/long
  cast/carry records. Result: sequential-compatible cast placement
  `exp_decay * BF16(B*dt) * x` matched current GPU oracle bits for all records,
  while reference-style `BF16(B*exp_decay*dt) * x` did not (`39920` tokens:
  seq `0xbe11`, ref `0xbe10`, GPU `0xbe11`). Initial carry for state flat `0`
  was `0.0`; long state flat entry seq-ref delta was
  `-0.0000007075723260641098`. Next v4 candidate must use current
  sequential-oracle-compatible chunk-state cast placement unless a separate gate
  deliberately changes the default oracle.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD v4 cast/carry diagnostic gate
  `20260628_0650_nemotron_nano_mamba2_ssd_v4_cast_carry_diagnostic`.
  Recorded `0641` first: default path remained unchanged, no build/runtime
  work was done in that static gate, and the strongest suspect for the failed
  v3 row64 mismatch is chunk0 state accumulation feeding chunk1 entry state,
  especially cast placement. Scope: add only guarded diagnostics if existing
  diagnostics cannot expose chunk0 state flat `0`, chunk1 entry contribution
  for `row=64, head=11, d=22`, sequential cast placement
  `exp_decay * BF16(B*dt) * x`, reference-style
  `BF16(B*exp_decay*dt) * x`, `exp(dA_last)` carry, and `entry_state[1]`.
  Plan: `./dev build`, then the smallest startup diagnostic only. No optimized
  path, benchmark, request, decode/HCS work, production enablement, or
  default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD `2321` v3 first-mismatch
  analysis gate
  `20260628_0641_nemotron_nano_mamba2_ssd_2321_v3_first_mismatch_analysis`.
  Static analysis only; no build, runtime diagnostic, benchmark, request,
  decode/HCS work, production enablement, or default-path change. Result:
  `row=64, head=11, d=22` is `chunk_idx=1`, `chunk_pos=0`, so the direct
  output path is chunk-entry/prior state for chunk 1, not final-state
  writeback. Because startup initial SSM state is expected zero, `exp(dA_last)`
  carry over initial state is unlikely for this first mismatch. The strongest
  static suspect is chunk-state accumulation for chunk 0 feeding
  `entry_state[1]`, especially cast placement: current sequential oracle uses
  `exp(dA_last-dA_u) * BF16(B*dt) * x`, while the vendored reference
  chunk-state path casts after applying the decay scale. The exact removed v3
  faulty line is not provable from retained artifacts, so v4 must add targeted
  probes for row64 prior/entry state and state flat `0` before any long timing.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD `2321` v3 first-mismatch
  analysis gate
  `20260628_0641_nemotron_nano_mamba2_ssd_2321_v3_first_mismatch_analysis`.
  Recorded `2346` first: the failed `2321` v3 prototype was fully cleaned up,
  final `./dev build` passed (`exit 0`, `169s`), minimal startup diagnostic
  without v3/local-tri envs exited `0`, HQQ attention and effective Marlin
  `g64` reused cleanly, the default sequential SSD path was confirmed, no
  temp/lock files remained, and `1908` plus `2149`/`2251` diagnostics were
  preserved. Scope is static analysis of the `2321` first v3 output mismatch
  `row=64, head=11, d=22` and first state mismatch `flat=0`; inspect source
  artifacts/logs and reference SSD state-passing math, then produce a concrete
  v4 correction plan. No prototype implementation, benchmark, request,
  decode/HCS work, production enablement, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD v3 failed-prototype cleanup gate
  `20260627_2346_nemotron_nano_mamba2_ssd_v3_failed_cleanup`. Reverted only
  the failed `2321` v3 additions: CUDA v3 kernel, Rust v3 env gates,
  temporary buffers, oracle/dispatch/PTX symbol checks, and v3-specific Python
  warmup fail-closed handling. Preserved `1908` Mamba2 substage timing, all
  artifacts, and the `2149`/`2251` guarded local-tri/state-flat diagnostics.
  Final `./dev build` passed (`exit 0`, total `169s`). Minimal startup
  diagnostic without v3/local-tri envs exited `0`, reused HQQ attention and
  effective Marlin `g64`, showed no cache-write evidence, sent `0` requests,
  ran no benchmark path, and exercised the default sequential SSD path
  (`ssd_scan 114464.9ms` over `23` calls for the 39,920-token calibration
  prefill). Final cleanup left no tmux/runtime/build process, no NVIDIA
  compute process, no Nano temp/lock files, and GPUs at `15 MB / 11 MB`.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD v3 failed-prototype cleanup gate
  `20260627_2346_nemotron_nano_mamba2_ssd_v3_failed_cleanup`. Recorded
  `2321` first: the v3 path was opt-in only, default sequential SSD was
  reported unchanged, `./dev build` passed, the small startup diagnostic
  failed closed, and no long timing was run. Baselines carried forward: Gemma
  `5619.6 / 92.43 / 155.69`, Nano `346.5 / 94.48 / 164.75`. Scope: revert
  only the v3 additions, including the CUDA v3 kernel, Rust v3 env gates,
  temporary buffers, oracle/dispatch/PTX symbol checks, and v3-specific Python
  warmup fail-closed handling. Preserve `1908` Mamba2 substage timing, all
  artifacts, and the `2149`/`2251` guarded local-tri/state-flat diagnostics.
  Validation plan: run `./dev build`, then a minimal startup diagnostic without
  v3 envs to confirm default sequential SSD loads, cache reuse is clean, no
  temp/lock files remain, and no requests or benchmarks occur.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD chunk-entry/state-carry v3
  correctness prototype gate
  `20260627_2321_nemotron_nano_mamba2_ssd_chunk_entry_state_carry_v3_correctness_prototype`.
  Implemented only an opt-in candidate behind
  `KRASIS_MAMBA2_SSD_CHUNK_PARALLEL_V3` with required oracle env
  `KRASIS_MAMBA2_SSD_CHUNK_PARALLEL_V3_ORACLE`; default sequential SSD remains
  unchanged. Candidate uses separate output/state buffers, materializes
  per-chunk entry state, and would commit live `ssm_state` only after oracle
  success. Added v3 fail-closed startup handling and reordered the preserved
  local-tri oracle to emit before v3 full-oracle failure. Final `./dev build`
  passed (`exit 0`, total `170s`). Small startup diagnostic exited `1` by
  fail-closed oracle path: local-tri preflight emitted for row `82`, head `37`,
  d `30` and showed a one-BF16-bit discrepancy (`0xbe4e` vs `0xbe4d`,
  `0.0009765625`); v3 full oracle failed on layer `0` with first output
  mismatch `row=64, head=11, d=22`, output max/mean
  `2.0 / 0.000009420193521236797`, and first state mismatch `flat=0`,
  state max/mean `0.0048828125 / 0.000001652099002238201`. Long diagnostic
  was not run. No requests, speed benchmark, decode/HCS work, production
  enablement, or default-path change.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD chunk-entry/state-carry v3
  correctness prototype gate
  `20260627_2321_nemotron_nano_mamba2_ssd_chunk_entry_state_carry_v3_correctness_prototype`.
  Recorded `2312` first: design-only gate closed with default SSD unchanged,
  no runtime/benchmark/request work, and the next risk narrowed to candidate
  state lifecycle rather than local triangular math. Baselines carried forward:
  Gemma `5619.6 / 92.43 / 155.69`, Nano `346.5 / 94.48 / 164.75`. Scope:
  implement only an opt-in v3 candidate with separate oracle/candidate output
  and state buffers, per-chunk entry state materialization, candidate state
  update in the `2312` order, delayed live `ssm_state` commit only after oracle
  success, and probes for row `82`, head `37`, d `30` plus state flat `14`.
  Default sequential SSD must remain unchanged; run `./dev build`, then the
  smallest startup diagnostic with term oracle and full sequential oracle.
  Stop on any mismatch; run one long diagnostic only if the small oracle passes.
  No requests, speed benchmark, decode/HCS work, production enablement, or
  default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD chunk-entry/state-carry design
  gate
  `20260627_2312_nemotron_nano_mamba2_ssd_chunk_entry_state_carry_design`.
  Static analysis only; no runtime diagnostic was needed. Inspected the
  sequential state update/writeback math, local/reference Mamba2 SSD pipeline,
  `2025`/`2138` design notes, `2213` mismatch artifacts, and `2251` state
  probe. Result: v3 should not be another local triangular patch. It should
  implement an opt-in candidate with explicit `chunk_state`, `state_passing`,
  `entry_state` scratch, separate candidate/oracle state buffers, and delayed
  live `ssm_state` commit only after oracle success. Required probes: row `82`
  output components and `entry_state[1]` prior term, plus state flat `14`
  chunk0 contribution, chunk1 entry state, and final state. No source
  implementation, benchmark, request, decode/HCS, production enablement, or
  default-path change.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD chunk-entry/state-carry design gate
  `20260627_2312_nemotron_nano_mamba2_ssd_chunk_entry_state_carry_design`.
  Recorded `2251` first: local triangular output at v2 mismatch coordinate
  `row=82, head=37, d=30` is effectively clean, state flat `14` sequential
  recompute is clean, and the next suspected failure area is candidate
  chunk-entry state/carry or state-update/writeback semantics. Scope:
  static analysis only unless a tiny guarded diagnostic is clearly needed;
  produce a concrete v3 patch plan with candidate buffers, chunk-entry state
  source, per-chunk update order, final state commit semantics, and oracle
  probes for row `82` / state flat `14`. No benchmark, request, decode/HCS,
  production enablement, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 `2213` v2 mismatch-analysis gate
  `20260627_2251_nemotron_nano_mamba2_ssd_2213_v2_mismatch_analysis`.
  Used the preserved `2149` local-triangular diagnostic against v2's actual
  first output mismatch coordinate `row=82, head=37, d=30` and added only a
  guarded diagnostic extension for `KRASIS_MAMBA2_SSD_LOCAL_TRI_ORACLE_STATE_FLAT`
  to inspect first state mismatch `flat=14`. `./dev build` passed (`exit 0`,
  total `169s`). Startup diagnostics exited `0`, sent `0` requests, ran no
  benchmark path, reused HQQ attention and effective Marlin `g64`, and showed
  no cache-write evidence. Row `82` local terms were exact for both calibration
  passes and only one BF16 bit off during warmup (`0xbe4e` vs `0xbe4d`,
  delta `0.0009765625`), far smaller than the reverted v2 output mismatch
  magnitude `0.25`. State flat `14` matched sequential state exactly in both
  calibration passes, with only `2.3283064365386963e-10` warmup recompute
  delta. Decision: local triangular math is not the primary v2 failure cause;
  next corrected candidate should focus on chunk-entry state/carry and
  state-update/writeback buffer semantics.

- Opened Nemotron Nano HQQ4+k4v4 `2213` v2 mismatch-analysis gate
  `20260627_2251_nemotron_nano_mamba2_ssd_2213_v2_mismatch_analysis`.
  Recorded `2235` first: failed v2 prototype reverted, `./dev build` and
  startup diagnostic passed, default sequential SSD loaded, HQQ attention and
  effective Marlin `g64` reused cleanly, no temp/lock files remained, and the
  `1908` substage instrumentation plus `2149` guarded local-triangular
  diagnostic were preserved. Scope: use the preserved `2149`
  coordinate-driven oracle against v2's first output mismatch
  `row=82, head=37, d=30` and first state mismatch `flat=14`; no optimized
  path, benchmarks, requests, decode/HCS work, production enablement, or
  default-path changes.

- Closed Nemotron Nano HQQ4+k4v4 corrected chunk-parallel SSD prototype v2
  cleanup gate
  `20260627_2235_nemotron_nano_mamba2_ssd_chunk_parallel_v2_failed_cleanup`.
  Reverted only the failed `2213` v2 additions: CUDA
  `mamba2_ssd_chunk_v2_*` kernels, Rust v2 env/temp-buffer/oracle/dispatch/PTX
  symbol plumbing, and v2-specific Python warmup fail-closed handling.
  Preserved `1908` Mamba2 substage instrumentation and the `2149` guarded
  local-triangular diagnostic. Source marker scan confirmed no v2 source
  markers remain. `./dev build` passed (`exit 0`, total `175s`). Minimal
  startup diagnostic without v2 envs exited `0`, sent `0` requests, ran no
  benchmark path, reused HQQ attention and effective Marlin `g64`, and emitted
  default sequential SSD timing (`ssd_scan` `293.5ms` short / `674.3ms` long
  over `23` calls). No cache-write path or temp/lock files were observed.

- Opened Nemotron Nano HQQ4+k4v4 corrected chunk-parallel SSD prototype v2
  cleanup gate
  `20260627_2235_nemotron_nano_mamba2_ssd_chunk_parallel_v2_failed_cleanup`.
  Recorded `2213` first: `./dev build` passed (`exit 0`, total `175s`),
  term preflight passed for `row=70, head=37, d=50` under runtime
  `mamba_chunk_size=64`, full sequential oracle failed closed on layer `0`
  before any long diagnostic, first output mismatch was
  `row=82, head=37, d=30` (`max/mean 0.250000000 / 0.000000479`), and state
  mismatches were present (`143240`, max/mean
  `0.000244141 / 0.000000021`, first `head=0, d=0, state=14`). Scope: revert
  only the v2 additions (CUDA v2 kernels, Rust v2 env gates/temp buffers/oracle
  dispatch/PTX symbol checks, and v2-specific Python warmup fail-closed
  handling), preserve `1908` Mamba2 substage instrumentation, preserve the
  `2149` guarded local-triangular diagnostic unless clearly entangled with v2,
  run `./dev build`, then run a minimal startup diagnostic without v2 envs. No
  requests, benchmarks, decode/HCS work, production enablement, or artifact
  removal.

- Closed Nemotron Nano HQQ4+k4v4 corrected chunk-parallel SSD prototype v2
  gate
  `20260627_2213_nemotron_nano_mamba2_ssd_chunk_parallel_v2_prototype`.
  Added an explicitly opt-in Rust/CUDA v2 candidate behind
  `KRASIS_MAMBA2_SSD_CHUNK_PARALLEL_V2` plus full oracle mode behind
  `KRASIS_MAMBA2_SSD_CHUNK_PARALLEL_V2_ORACLE`. The default SSD path remains
  sequential. `./dev build` passed (`exit 0`, total `175s`). The smallest
  startup diagnostic with the `2149` term oracle and full sequential oracle
  enabled failed closed on Mamba2 layer `0` before any long run. The term
  preflight passed at `row=70, head=37, d=50`, preserving runtime
  `mamba_chunk_size=64` (`chunk_start=64`, `chunk_pos=6`), but the full oracle
  found `2` output mismatches (`max/mean 0.250000000 / 0.000000479`, first
  mismatch flat `338270` = `row=82, head=37, d=30`) and `143240` state
  mismatches (`max/mean 0.000244141 / 0.000000021`, first flat `14` =
  `head=0, d=0, state=14`). Long diagnostic was not run. Startup reused HQQ
  attention and the effective Marlin `g64` cache, sent `0` requests, ran no
  benchmark path, and left no temp/lock files. Decision: v2 is not worth
  keeping as-is; analyze or replace output/state math before any performance
  measurement.

- Opened Nemotron Nano HQQ4+k4v4 corrected chunk-parallel SSD prototype v2
  gate
  `20260627_2213_nemotron_nano_mamba2_ssd_chunk_parallel_v2_prototype`.
  Recorded `2149` first: Gemma baseline `5619.6 / 92.43 / 155.69`, Nano
  baseline `346.5 / 94.48 / 164.75`, the local-triangular oracle matched
  existing sequential GPU output bits for `300`, `500`, and `39,920` token
  diagnostics, and the corrected target coordinate is production
  `chunk_start=64`, `chunk_pos=6` under runtime `mamba_chunk_size=64`.
  Scope: implement only an explicitly opt-in correctness-focused candidate
  that preserves runtime chunk boundaries exactly, runs the `2149` term oracle
  as preflight before full sequential oracle comparison, and fails closed on
  any term/output/state mismatch. Default SSD path must remain sequential; no
  silent fallback, smoke requests, speed benchmark, decode/HCS work, or
  production enablement in this gate.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD local-triangular
  oracle/instrumentation gate
  `20260627_2149_nemotron_nano_mamba2_ssd_local_triangular_oracle_instrumentation`.
  Added guarded diagnostic-only Rust instrumentation behind
  `KRASIS_MAMBA2_SSD_LOCAL_TRI_ORACLE` plus explicit layer/row/head/d env
  coordinates. Default behavior remains the existing sequential SSD path.
  `./dev build` passed (`exit 0`, total `170s`). Startup diagnostic with
  `KRASIS_STARTUP_DIAG=1`, prefill timing/liveness, and the local-triangular
  oracle env exited `0`, sent `0` requests, ran no benchmark path, reused HQQ
  attention and the effective Marlin `g64` cache, and left no cache write
  evidence. The oracle emitted three records (`300`, `500`, and `39,920`
  tokens); all CPU local-triangular recomputes matched the existing sequential
  GPU output bits. Corrected finding: the prior `2039` "chunk 0" label was
  under prototype chunking, but the current production sequential runtime uses
  `mamba_chunk_size=64`, so `t=70, head=37, d=50` is production
  `chunk_start=64`, `chunk_pos=6`. The exact local terms for the long
  diagnostic were recorded: lower triangle includes only `u=64..70`, self
  sample `u=70` has `decay=1.0`, reverse `C@B` is used, and
  `BF16(CB*decay*dt)` is cast before multiplying by `x`. Corrected patch plan:
  the next chunk-parallel candidate must either preserve runtime
  `mamba_chunk_size`/chunk-start semantics or prove its state-passing makes
  larger candidate chunks algebraically identical before any performance run.
  No optimized path, smoke request, speed benchmark, decode/HCS work,
  production enablement, cache generation, or default-path change occurred.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD local-triangular
  oracle/instrumentation gate
  `20260627_2149_nemotron_nano_mamba2_ssd_local_triangular_oracle_instrumentation`.
  Recorded `2138` first: Gemma baseline `5619.6 / 92.43 / 155.69`, Nano
  baseline `346.5 / 94.48 / 164.75`, and the first failed `2039`
  chunk-prototype output mismatch was inside chunk `0` at
  `t=70, head=37, d=50`. Scope: add only minimal guarded diagnostics, disabled
  by default, around the existing sequential SSD local-triangular math to expose
  A/dt prefix, lower-triangle/self-decay handling, reverse `C@B`, BF16 cast
  point, `dt*x` multiplication, and accumulation at the first mismatch
  coordinate. Use built commands only (`./dev build`, then the smallest startup
  diagnostic with the diagnostic env enabled). No new optimized path, benchmark,
  smoke request, decode/HCS work, production enablement, cache generation, or
  default-path change in this gate.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD reference-math/mismatch-analysis
  gate
  `20260627_2138_nemotron_nano_mamba2_ssd_reference_math_mismatch_analysis`.
  Static analysis only. Inspected the active sequential SSD kernel and trace
  path, the `2025` design artifacts, the `2039` oracle failure artifacts, and
  the local vendored vLLM/state-spaces Mamba2 SSD reference implementation in
  `krasisx/python/krasis/mamba2_ops`. Result: the corrected next design is not
  the previous four-stage shortcut. It should mirror the reference pipeline:
  `chunk_cumsum`, `chunk_state`, `state_passing`, `cb_matrix`,
  `chunk_scan_output`, and final-state commit, with mandatory comparison
  against the existing sequential kernel. The first `2039` output mismatch was
  at `t=70, head=37, d=50`, inside chunk `0`, so cross-chunk carry cannot be
  the sole cause; the local triangular output path must exactly preserve
  lower-triangle masking, self-sample decay identity, reverse-order `C@B`, and
  `BF16(CB*decay*dt)` coefficient casting. Exact removed-line attribution for
  the failed `2039` candidate remains unknown because its code was reverted and
  no candidate intermediate terms were captured before cleanup. No
  implementation, benchmark, request, decode/HCS work, production enablement,
  or default-path change occurred.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD reference-math/mismatch-analysis
  gate
  `20260627_2138_nemotron_nano_mamba2_ssd_reference_math_mismatch_analysis`.
  Recorded `2102` first: the failed `2039` chunk-parallel SSD prototype was
  reverted, `./dev build` passed, the minimal startup diagnostic passed,
  default sequential SSD loads, cache reuse is clean, and the `1908` Mamba2
  substage instrumentation is preserved. Scope is static analysis only:
  inspect the sequential SSD kernel, trace path, `2025` design artifacts,
  `2039` mismatch artifacts, and local/reference Mamba2 SSD implementations;
  reconcile chunk entry state, A/dt prefix, B/C group indexing, local
  triangular output, chunk carry/final state, D skip, z/norm placement, and
  BF16 conversion. No implementation, benchmark, smoke request, decode/HCS
  work, production enablement, or default-path change in this gate.

- Closed Nemotron Nano HQQ4+k4v4 failed chunk-parallel SSD prototype cleanup
  gate
  `20260627_2102_nemotron_nano_mamba2_ssd_chunk_parallel_failed_prototype_cleanup`.
  Reverted only the failed `2039` prototype additions: CUDA
  `prepare_prefix`, `cb_matrix`, `output`, and `state_update` candidate
  kernels; Rust chunk env gates, scratch/oracle buffers, oracle comparison,
  candidate dispatch, and required PTX symbol checks; and chunk-specific
  Python warmup fail-closed handling. Preserved the `1908` Mamba2 substage
  instrumentation and existing benchmark artifacts. `./dev build` passed
  (`exit 0`, total `175s`). Minimal startup diagnostic without chunk prototype
  envs exited `0`, sent `0` requests, ran no benchmark path, validated HQQ
  attention from cache, loaded the effective Marlin `g64` cache, emitted the
  default sequential SSD timing path (`ssd_scan` `293.6ms` over `23` calls for
  the 128-token calibration), and exited after calibration. No cache-write
  marker or temp/lock files remained.

- Opened Nemotron Nano HQQ4+k4v4 failed chunk-parallel SSD prototype cleanup
  gate
  `20260627_2102_nemotron_nano_mamba2_ssd_chunk_parallel_failed_prototype_cleanup`.
  Recorded `2039` first: `./dev build` passed, the small oracle diagnostic
  failed closed on Mamba2 layer `0`, output max/mean abs error
  `2.000000000` / `0.000003399`, state max/mean abs error `0.000366211` /
  `0.000000023`, and no long diagnostic was run. Scope: revert only the
  `2039` prototype additions (CUDA `prepare_prefix`, `cb_matrix`, `output`,
  and `state_update` kernels; Rust chunk env gates, scratch, oracle, dispatch,
  and PTX symbol checks; and chunk-specific Python warmup fail-closed
  handling), preserve the `1908` Mamba2 substage instrumentation and all
  benchmark artifacts, run `./dev build`, then run a minimal startup
  diagnostic without chunk prototype envs to confirm default sequential SSD
  still loads with clean cache reuse, no temp/lock files, and `0` requests or
  benchmarks.

- Closed Nemotron Nano HQQ4+k4v4 gated Mamba2 SSD/scan chunk-parallel
  prototype gate
  `20260627_2039_nemotron_nano_mamba2_ssd_chunk_parallel_gated_prototype` as
  a correctness-failed prototype stop. Implemented the opt-in Rust/CUDA
  candidate only behind `KRASIS_MAMBA2_SSD_CHUNK_PARALLEL=1`, with required
  oracle gate `KRASIS_MAMBA2_SSD_CHUNK_PARALLEL_ORACLE=1` for validation.
  Default behavior remains the existing sequential SSD kernel. `./dev build`
  passed (`exit 0`). The small startup oracle diagnostic used the built
  `./dev run tests/nemotron-nano-4-4-hqq4-k4v4-a16.conf` path with startup
  diagnostics, prefill timing, liveness timing, and 256-token warmup/short/long
  calibration. It failed closed on Mamba2 layer `0`: output mismatches `27`,
  output max/mean abs error `2.000000000` / `0.000003399`, state mismatches
  `134008`, state max/mean abs error `0.000366211` / `0.000000023`. The long
  diagnostic was not run. HQQ attention and effective Marlin `g64` loaded from
  cache before the oracle failure. No smoke requests, speed benchmark,
  decode/HCS work, production enablement, cache generation, or default-path
  change occurred. Decision: this chunk-parallel candidate is not worth
  keeping; leave it disabled and revert/replace it in a follow-up cleanup or
  redesign gate.

- Opened Nemotron Nano HQQ4+k4v4 gated Mamba2 SSD/scan chunk-parallel
  prototype gate
  `20260627_2039_nemotron_nano_mamba2_ssd_chunk_parallel_gated_prototype`.
  Recorded `2025` first: the prior gate was design-only, made no
  implementation/default-path changes, and preserved the `1908` Mamba2
  substage instrumentation. Current baselines carried forward: Gemma
  `5619.6 / 92.43 / 155.69`, Nano HQQ4+k4v4 `346.5 / 94.48 / 164.75`, and
  the rejected `1938` scalar-reuse prototype regressed SSD/scan by `+5.472%`.
  Scope: implement only an opt-in chunk-parallel SSD path using the existing
  sequential kernel as oracle. Default path must remain sequential. Candidate
  path must derive dimensions from runtime tensors/config, allocate scratch,
  require PTX symbols, dispatch only under explicit env gates, run
  `prepare_prefix -> cb_matrix -> output -> state_update`, and fail closed on
  oracle mismatch with max/mean output and state error recorded. No smoke
  requests, speed benchmark, decode/HCS work, production enablement, cache
  generation, or default-path change.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD/scan algorithm-design gate
  `20260627_2025_nemotron_nano_mamba2_ssd_scan_algorithm_design` as a
  design-only stop. Source audit confirmed the active SSD path is
  Rust-launched CUDA PTX, not Torch/Python, and the trace path mirrors the same
  sequential math. Recorded a concrete chunk-parallel follow-up design: keep
  the current sequential kernel as the default and oracle; add an explicit
  gated path with per-chunk `prepare_prefix`, `cb_matrix`, `output`, and
  `state_update` kernels; carry state sequentially between chunks; compare
  candidate output/state against the existing sequential kernel in a mandatory
  small startup diagnostic oracle before any long run. Tensor shapes, state
  carry semantics, launch plan, oracle strategy, and source spans are recorded
  in benchmark artifacts. The design is coherent but not small/safe enough to
  implement in this same gate because it requires new scratch buffers, CUDA
  kernels, Rust dispatch/symbol wiring, and oracle comparison. No source
  runtime path, benchmark, request, decode/HCS work, cache generation, or
  production enablement occurred.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD/scan algorithm-design gate
  `20260627_2025_nemotron_nano_mamba2_ssd_scan_algorithm_design`.
  Recorded `2011` first: the rejected `1938` scalar-reuse prototype was
  removed, `./dev build` passed, the default sequential SSD path still loads,
  HQQ attention and effective Marlin `g64` reuse remained clean, and no
  temp/lock files remained. Current speed baselines: Gemma `5619.6 / 92.43 /
  155.69`, Nano HQQ4+k4v4 `346.5 / 94.48 / 164.75`, and the rejected `1938`
  prototype regressed SSD/scan by `+5.472%` (`+6,262.531ms`). Scope: inspect
  the existing sequential SSD kernel, trace path, prefix/scan helpers, and
  Mamba2 math needed for a real chunk-parallel CUDA implementation; produce a
  concrete plan with tensor shapes, state carry semantics, launch structure,
  oracle strategy, and source spans. No implementation, benchmark, request,
  decode/HCS work, production enablement, or default-path change unless the
  next patch is unambiguously safe.

- Closed Nemotron Nano HQQ4+k4v4 rejected SSD/scan prototype cleanup gate
  `20260627_2011_nemotron_nano_mamba2_ssd_rejected_prototype_cleanup`.
  Reverted only the rejected `1938` prototype additions: CUDA head-shared
  prototype kernel/wrapper, Rust prototype env gates/oracle/dispatch/PTX symbol
  checks, and the prototype-specific Python warmup fail-closed handling. The
  `1908` Mamba2 substage instrumentation remains in place, and existing
  benchmark/artifact records were not removed. `./dev build` passed (`exit 0`,
  maturin phase `169s`). Minimal startup diagnostic without prototype envs
  exited `0` after calibration with `0` requests and no benchmark path:
  `KRASIS_STARTUP_DIAG=1 KRASIS_PREFILL_TIMING=1
  KRASIS_PREFILL_LIVENESS_TIMING=1 KRASIS_STARTUP_WARMUP_TOKENS=128
  KRASIS_STARTUP_CAL_SHORT_TOKENS=128 KRASIS_STARTUP_CAL_LONG_TOKENS=128
  KRASIS_STARTUP_EXIT_AFTER_CALIBRATION=1 ./dev run
  tests/nemotron-nano-4-4-hqq4-k4v4-a16.conf`. HQQ attention validated from
  cache, effective Marlin `g64` loaded from cache, no cache-write line appeared,
  and default sequential SSD timing/liveness still emitted (`ssd_scan`
  `293.5ms` over `23` calls for the 128-token long calibration). No smoke
  requests, speed benchmark, decode/HCS changes, cache generation, or
  production enablement occurred.

- Opened Nemotron Nano HQQ4+k4v4 rejected SSD/scan prototype cleanup gate
  `20260627_2011_nemotron_nano_mamba2_ssd_rejected_prototype_cleanup`.
  Recorded `1938` first: the gated SSD/scan prototype oracle passed exactly
  (`69` rows, `0` output/state mismatches, max/mean error `0`), but the long
  diagnostic was performance-negative. SSD/scan regressed from `114,449.4ms`
  to `120,711.931ms` (`+6,262.531ms`, `+5.472%`), and long prefill regressed
  from `117,356.0ms` to `123,870.0ms` (`+5.551%`). Scope: revert only the
  `1938` prototype additions (CUDA prototype kernel/wrapper, Rust prototype
  env gates/oracle/dispatch/symbol checks, and prototype-specific Python
  warmup fail-closed handling), preserve the `1908` Mamba2 substage
  instrumentation and all benchmark/artifact records, run `./dev build`, then
  run a minimal startup diagnostic without prototype envs to confirm the
  default sequential path still loads with clean cache reuse, no temp/lock
  files, and `0` requests/benchmarks.

- Closed Nemotron Nano HQQ4+k4v4 gated SSD/scan prototype gate
  `20260627_1938_nemotron_nano_mamba2_ssd_scan_gated_prototype` as
  correctness-clean but performance-negative. Implemented an explicitly gated
  Rust/CUDA head-shared SSD prototype behind `KRASIS_MAMBA2_SSD_REUSE_PROTO`
  and an oracle mode behind `KRASIS_MAMBA2_SSD_REUSE_PROTO_ORACLE`; the
  default path remains the existing sequential kernel. `./dev build` passed
  after adding the CUDA host wrapper/required PTX symbol. The small built
  startup diagnostic compared the prototype against the existing sequential
  kernel at `128` tokens and passed exactly: `69` oracle rows, `0` output/state
  mismatches, max/mean error `0`. The long built startup diagnostic then reused
  HQQ attention and effective Marlin `g64`, sent `0` requests, and ran no
  benchmark path, but the prototype made SSD/scan slower: `120,711.931ms`
  versus baseline `114,449.4ms` (`+6,262.531ms`, `+5.472%`). Long prefill
  worsened from `117,356.0ms` to `123,870.0ms`. Decision: do not promote this
  prototype; keep it disabled only as recorded diagnostic evidence or
  revert/replace it with a real chunk-parallel SSD design in a follow-up gate.
  No smoke requests, speed benchmark, decode/HCS changes, cache generation, or
  production enablement occurred.

- Opened Nemotron Nano HQQ4+k4v4 gated SSD/scan prototype gate
  `20260627_1938_nemotron_nano_mamba2_ssd_scan_gated_prototype`.
  Recorded `1928` first: SSD/scan is `114,449.4ms`, `99.661%` of Mamba2 and
  `97.523%` of total long prefill, while GQA/HQQ attention and MoE/Marlin are
  ruled out. The current source-level target is the correctness-first
  Rust-launched CUDA `mamba2_ssd_sequential_kernel`, which recomputes scalar
  `C[t] @ B[u]`, decay, and old-state terms across `head_dim` lanes. Scope:
  implement only the smallest opt-in Rust/CUDA path that reduces duplicate SSD
  scalar work across `head_dim` lanes, keep the existing sequential kernel as
  the default production path, derive dimensions from config/runtime tensors,
  add a correctness oracle mode against the existing kernel on a small
  diagnostic prefill, fail closed on mismatch without silent fallback, run
  `./dev build`, then use built startup diagnostics only. No smoke requests,
  speed benchmark, decode/HCS changes, cache generation, or production
  enablement.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 SSD/scan optimization target analysis
  gate `20260627_1928_nemotron_nano_mamba2_ssd_scan_optimization_target_analysis`.
  Source/data audit found the concrete target: the current SSD prefill kernel
  is the correctness-first `mamba2_ssd_sequential_kernel`, explicitly described
  in source as a simple sequential scan equivalent to running decode `N` times.
  Normal timing/liveness diagnostics use the non-trace Rust-launched CUDA PTX
  kernel, not Torch/Python and not the trace fallback. For Nano (`39,920`
  tokens, chunk size `128`, `64` heads, `64` head_dim, state size `128`) the
  kernel launches only `64` CTAs per Mamba2 layer and then loops sequentially
  over tokens plus local chunk scan terms. The local `C[t] @ B[u]`, decay, and
  old-state scalar terms are independent of `head_dim` but are recomputed
  across all `64` `d` lanes, producing an estimated
  `2,694,647,382,016` inner state multiplications per layer and
  `61,976,889,786,368` across the measured `23` Mamba2 layers. Per-layer
  spread is flat (`0.6184%` of mean), and SSD wall/event/sync align, so launch
  count, CPU/GPU transfer, queue sync, HQQ attention, and Marlin are not the
  next target. Proposed next patch: a gated alternative chunk-parallel/reuse
  SSD prefill kernel that shares per-head/group/t/u scalar terms across
  `head_dim` lanes and keeps the sequential kernel as correctness oracle until
  validated. No source optimization, runtime rerun, speed benchmark, smoke
  request, decode/HCS work, or production enablement occurred.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 SSD/scan optimization target analysis
  gate `20260627_1928_nemotron_nano_mamba2_ssd_scan_optimization_target_analysis`.
  Recorded `1908` first: SSD/scan is `114,449.4ms`, `99.661%` of Mamba2
  wall, and `97.523%` of total long prefill. GQA/HQQ attention (`476.8ms`)
  and MoE/Marlin (`1,931.6ms`) are ruled out for this symptom. Scope: inspect
  the exact `forward_mamba2` SSD/scan implementation and called kernels before
  changing code; determine whether the time is kernel algorithm, launch count,
  tensor layout/contiguity, CPU/GPU transfer, queue sync/event timing, or a
  fallback path; capture source spans, tensor dimensions, per-layer timing
  spread, and whether the current implementation is Rust/CUDA/Torch/Python
  bound. No speed benchmark. Use built diagnostic commands only if runtime
  confirmation is needed.

- Closed Nemotron Nano HQQ4+k4v4 Mamba2 substage instrumentation gate
  `20260627_1908_nemotron_nano_mamba2_substage_instrumentation`.
  Added minimal guarded Rust instrumentation in `forward_mamba2` for in-proj,
  split/extract, conv1d+silu, SSD/scan, gated RMSNorm, out-proj, and
  queue/event/sync timing. The instrumentation is off unless existing
  `KRASIS_PREFILL_TIMING` or `KRASIS_PREFILL_LIVENESS_TIMING` diagnostics are
  enabled; no model-specific constants or behavior changes were added.
  `./dev build` passed (`exit 0`, `172s`). The requested startup diagnostic
  exited `0`, sent `0` requests, reused HQQ attention and effective Marlin
  `g64` caches, and exited after calibration. Long diagnostic prefill:
  `39,920` tokens in `117,356.0ms` (`339.4 tok/s` calibration line). Mamba2 was
  `114,839.1ms` wall / `114,838.9ms` event / `114,836.4ms` sync. The
  dominant substage is SSD/scan: `114,449.4ms`, `99.661%` of Mamba2 wall and
  `97.523%` of total prefill wall. In-proj was `213.9ms`, split/extract
  `18.5ms`, conv1d+silu `41.9ms`, gated RMSNorm `25.1ms`, and out-proj
  `90.3ms`. GQA/HQQ attention was only `476.8ms`; MoE/Marlin was `1,931.6ms`.
  Decision: the next optimization target is Mamba2 SSD/scan, not HQQ attention
  or Marlin expert prefill. No optimization attempt, smoke request, benchmark,
  decode/HCS change, cache generation, or production enablement occurred.

- Opened Nemotron Nano HQQ4+k4v4 Mamba2 substage instrumentation gate
  `20260627_1908_nemotron_nano_mamba2_substage_instrumentation`.
  Recorded `1846` first: Gemma reference remains `5619.6 / 92.43 /
  155.69`; Nano HQQ4+k4v4 baseline remains internal prefill `346.5 tok/s`,
  internal decode `94.48 tok/s`, and network round-trip `164.75 tok/s`.
  The long diagnostic prefill was `39,920` tokens in `117,324.8ms`; Mamba2
  accounted for `114,836.447ms` (`97.884%`), while actual GQA/HQQ attention
  was `477.4ms` and MoE/Marlin was `1,934.0ms`. Current `attn` timing is a
  broad mixer bucket, not just HQQ/full attention. Scope: add only guarded
  Rust instrumentation around Mamba2 prefill substages (in-proj,
  split/extract, conv1d+silu, SSD/scan, gated RMSNorm, out-proj, and
  queue/event/sync), keep it disabled unless existing prefill timing/liveness
  envs are enabled, then run `./dev build` and the same startup diagnostic
  command. No optimization attempt, smoke requests, benchmark, decode/HCS, or
  production enablement.

- Closed Nemotron Nano HQQ4+k4v4 instrumented prefill diagnosis gate
  `20260627_1846_nemotron_nano_hqq4_k4v4_prefill_instrumented_diagnosis`.
  Used built diagnostic commands only, with instrumentation enabled and no
  benchmark/request path:
  `KRASIS_STARTUP_DIAG=1 KRASIS_PREFILL_TIMING=1 KRASIS_STARTUP_EXIT_AFTER_CALIBRATION=1 ./dev run tests/nemotron-nano-4-4-hqq4-k4v4-a16.conf`,
  followed by the same built startup diagnostic with
  `KRASIS_PREFILL_LIVENESS_TIMING=1` to split layer types. Both exited `0`.
  Cache reuse stayed clean: HQQ attention validated, effective Marlin `g64`
  loaded, no safetensors cache-build line, no requests, and no temp/lock files.
  Long prefill reproduced the slow path at `39,920` tokens:
  `117,324.8ms` (`340 tok/s`). The printed top-level `attn` bucket was
  `115,293.9ms`, but source inspection plus liveness timing showed this is a
  broad mixer envelope, not HQQ attention. Actual GQA/HQQ attention was only
  `477.4ms` (`0.407%`), and MoE/Marlin expert prefill was `1,934.0ms`
  (`1.649%`). Mamba2 layers accounted for `114,836.447ms` across `23` layers
  (`97.884%` of layer time, `4,992.889ms/layer`). Decision: the bottleneck is
  Nano Mamba2 SSM prefill. Next minimal instrumentation is Mamba2 substage
  timing inside `forward_mamba2`; do not optimize HQQ attention or Marlin
  expert prefill first for this symptom.

- Opened Nemotron Nano HQQ4+k4v4 instrumented prefill diagnosis gate
  `20260627_1846_nemotron_nano_hqq4_k4v4_prefill_instrumented_diagnosis`.
  Recorded `1821` first: baseline speed for this exact config was internal
  prefill `346.5 tok/s`, internal decode `94.48 tok/s`, and network
  round-trip `164.75 tok/s`; Gemma reference remains `5619.6 / 92.43 /
  155.69`. Cache startup reused HQQ attention and effective Marlin `g64`
  cleanly, with no cache-write evidence and no `.tmp`/lock files. Scope:
  inspect built timing/instrumentation flags first, then use only the smallest
  built diagnostic run that exposes component-level prefill timing with
  instrumentation enabled. This is not a speed benchmark. If the built command
  surface cannot expose prefill component timing, stop with the exact gap and
  proposed minimal instrumentation point.

- Closed Nemotron Nano HQQ4+k4v4 baseline speed measurement gate
  `20260627_1821_nemotron_nano_hqq4_k4v4_baseline_speed_measurement`.
  Used the built benchmark path only:
  `./dev benchmark tests/nemotron-nano-4-4-hqq4-k4v4-a16.conf`.
  No `--timing`, `--build-cache`, `--force-rebuild-hqq-cache`, custom Python,
  source changes, decode/HCS changes, cache generation, or production
  enablement. The benchmark exited `0`, reused HQQ attention cache and existing
  effective Marlin `g64`, reached READY, and completed the standard internal
  engine plus network benchmark. Internal prefill best was `346.5 tok/s`
  (best at `1,001` tokens; 50K run `337.1 tok/s`). Internal decode best was
  `94.48 tok/s` (best at `100` tokens; `10.6ms/tok`). Network round-trip best
  was `164.75 tok/s` and is recorded separately as HTTP/SSE client timing, not
  internal decode. HCS coverage was `2944/2944` and internal decode min free
  VRAM was `8,716 MB` versus the `600 MB` safety margin.

- Opened Nemotron Nano HQQ4+k4v4 baseline speed measurement gate
  `20260627_1821_nemotron_nano_hqq4_k4v4_baseline_speed_measurement`.
  Recorded `1800` first: after the `1742` cache-lookup fix the HQQ4+k4v4
  smoke reused HQQ attention and effective Marlin `g64` caches, reached READY,
  returned `3/3` HTTP `200` through `/v1/internal/reference_test`, selected
  token `1044 ","` for all cases, and kept positive `1044-1321` margins
  (`0.198773`, `0.149701`, `0.313407`). Top-k membership/rank drift remained
  `3/3`, so the result was selected-token smoke only, not production quality
  acceptance. Scope: identify the smallest built-command benchmark path for
  exactly `tests/nemotron-nano-4-4-hqq4-k4v4-a16.conf`, run it with
  instrumentation/timing disabled and no cache-rebuild flags if unambiguous,
  capture internal prefill and decode numbers separately, and avoid confusing
  client/server HTTP timing with internal decode. If the built benchmark
  command is ambiguous, stop with the exact ambiguity and proposed command.

- Closed Nemotron Nano HQQ4+k4v4 non-attribution smoke-after-reuse-fix gate
  `20260627_1800_nemotron_nano_hqq4_k4v4_non_attribution_smoke_after_reuse_fix`.
  Reused the `1742` source fix and existing caches. The first startup without
  `--test-endpoints` reached READY with clean cache reuse but `/v1/internal/reference_test`
  returned HTTP `404`; reran with `--test-endpoints` because it is required for
  the requested endpoint and is not a build/cache flag. Corrected run reached
  READY, HQQ attention validated, effective Marlin `g64` loaded, no safetensors
  cache-write line or temp/lock files appeared. Sent cases `2`, `0`, and `1`
  through `/v1/internal/reference_test` with `debug_reference_trace=true`;
  all returned HTTP `200`, selected token `1044 ","`, and had positive
  `1044-1321` margins (`0.198773`, `0.149701`, `0.313407`). Selected-token
  behavior matches BF16 and accepted HQQ6 g32 for the three-case smoke set, but
  top-k membership/rank drift remains `3/3`, so this is not production quality
  acceptance. Expert-HQQ consume rows/bytes are not applicable for this
  attention-HQQ k4v4 config. No source changes, cache generation, benchmarks,
  decode/HCS, or production enablement.

- Opened Nemotron Nano HQQ4+k4v4 non-attribution smoke-after-reuse-fix gate
  `20260627_1800_nemotron_nano_hqq4_k4v4_non_attribution_smoke_after_reuse_fix`.
  Recorded `1742` first: source fix implemented in `src/weights/mod.rs`;
  `./dev build` exited `0`; startup-only reuse check reached
  `KRASIS SERVER READY`; HQQ attention cache validated; existing effective
  Marlin `g64` expert cache loaded; `0` requests were sent; and no cache
  `.tmp`/lock files remained. Scope: run exactly `./dev run
  tests/nemotron-nano-4-4-hqq4-k4v4-a16.conf` with no build/cache flags and
  attribution/checkpoint envs off. Stop before requests on any cache-write
  evidence. If READY is clean, send case `2` first through
  `/v1/internal/reference_test` with `debug_reference_trace=true`; continue to
  cases `0` and `1` only if case `2` does not flip and the `1044-1321` margin
  is stable versus BF16 and accepted g32. No benchmarks, decode/HCS,
  production enablement, cache generation, or source changes.

- Closed Nemotron Nano Marlin cache lookup effective group-size source-fix gate
  `20260627_1742_nemotron_nano_marlin_cache_lookup_effective_group_size_fix`.
  Added a shared Marlin effective-group helper in `src/weights/mod.rs`, updated
  both HF and GGUF-hybrid startup lookup paths to derive the same effective
  group size as the builder before choosing `cache_path_marlin`, and replaced
  the builder's duplicate reduction loop with the same helper. The calculation
  derives from model dimensions and requested group size; there is no
  Nano-specific branch and no runtime hardcoded `64`. Added a focused helper
  test for the `1856` intermediate / requested `128` case. `./dev build`
  passed (`169s`). Re-ran exactly `./dev run
  tests/nemotron-nano-4-4-hqq4-k4v4-a16.conf` with no build/cache flags and
  attribution/checkpoint envs unset; sent `0` requests; reached
  `KRASIS SERVER READY`; HQQ attention validated from cache; Marlin loaded the
  existing effective `g64` cache (`15.8 GB in 10s`); no safetensors cache-build
  line and no `.tmp`/lock files appeared. No smoke requests, benchmarks,
  decode/HCS, or production enablement.

- Opened Nemotron Nano Marlin cache lookup effective group-size source-fix gate
  `20260627_1742_nemotron_nano_marlin_cache_lookup_effective_group_size_fix`.
  Recorded `1733` first: Gemma carried speeds `5619.6 / 92.43 / 155.69`;
  HQQ attention cache reused/validated; Nano `moe_intermediate_size=1856`;
  requested expert group size `128` reduces to effective `64`; final
  `experts_marlin_int4_g64_calamax.bin` exists at `15,849,200,704` bytes with
  header group size `64`; startup still checked the requested `g128` path,
  entered the Marlin write path before READY, sent `0` requests, and left no
  temp/lock files after cleanup. Scope: make the startup cache lookup derive
  the same effective expert group size as the builder before choosing the
  Marlin cache path, without hardcoding Nano or `64`; run `./dev build`; then
  repeat the startup-only reuse check with no requests and attribution/
  checkpoint envs off, stopping at READY or first cache-write evidence.

- Closed Nemotron Nano effective Marlin g64 reuse check gate
  `20260627_1733_nemotron_nano_effective_marlin_g64_reuse_check` as a
  cache-reuse failure before READY. Ran exactly `./dev run
  tests/nemotron-nano-4-4-hqq4-k4v4-a16.conf` with no build/cache flags and
  attribution/checkpoint envs unset. HQQ attention reused successfully
  (`115 MB cached`), and the effective `g64` Marlin artifact existed unchanged
  at `15,849,200,704` bytes with header group size `64`. Startup did not load
  it: it checked the requested `g128` path, entered `Building GPU INT4 Marlin
  expert cache (one-time)` before READY, and created a partial
  `experts_marlin_int4_g128_calamax.bin.tmp` plus `.lock`. Stopped immediately,
  sent `0` requests, recorded then removed the partial files. Source mismatch:
  the load path falls back to requested `group_size=128` for non-prequantized
  models before cache lookup, while the builder later reduces to effective
  `64` and renames the completed cache to the effective path. No source changes,
  benchmarks, decode/HCS, production enablement, or smoke requests.

- Opened Nemotron Nano effective Marlin g64 reuse check gate
  `20260627_1733_nemotron_nano_effective_marlin_g64_reuse_check`. Recorded
  `1720` first: Gemma carried speeds `5619.6 / 92.43 / 155.69`; HQQ attention
  cache validated; Nano `moe_intermediate_size=1856`; requested Marlin expert
  group size `128` reduced to effective `64`; final effective artifact
  `/home/main/.krasis/cache/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/experts_marlin_int4_g64_calamax.bin`;
  `0` requests; and no `.tmp`/lock files. Scope: run exactly
  `./dev run tests/nemotron-nano-4-4-hqq4-k4v4-a16.conf` with no build/cache
  flags and attribution/checkpoint envs off, send no requests, and stop at
  READY or first cache-write evidence. Capture whether HQQ attention and the
  effective Marlin g64 cache are reused without writes.

- Closed Nemotron Nano Marlin g128 expert-cache generation prerequisite gate
  `20260627_1720_nemotron_nano_marlin_g128_expert_cache_generation_prerequisite`
  as an exact-`g128` prerequisite failure. Ran only the built cache path
  `./dev run tests/nemotron-nano-4-4-hqq4-k4v4-a16.conf --build-cache`, with
  no `--force-rebuild-hqq-cache`, no smoke requests, and attribution/checkpoint
  env gates unset. The command exited `0` and printed `BUILD CACHE COMPLETE`;
  HQQ attention artifacts validated without a force rebuild. The final
  requested
  `/home/main/.krasis/cache/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/experts_marlin_int4_g128_calamax.bin`
  remained absent. The runtime-valid effective cache was
  `/home/main/.krasis/cache/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/experts_marlin_int4_g64_calamax.bin`,
  `15,849,200,704` bytes, with cache-header group size `64`. Source/data
  check: server resolved `expert_group_size=128`, but Nano
  `moe_intermediate_size=1856` is not divisible by `128`; the Marlin builder
  reduces to effective group size `64` and renames the completed cache to the
  effective `g64` path. No `.tmp`/lock files remain. No source changes,
  benchmarks, decode/HCS, production enablement, or smoke requests.

- Opened Nemotron Nano Marlin g128 expert-cache generation prerequisite gate
  `20260627_1720_nemotron_nano_marlin_g128_expert_cache_generation_prerequisite`.
  Recorded `1711` first: Gemma carried speeds
  `5619.6 / 92.43 / 155.69`; HQQ attention cache reuse was validated on the
  normal startup path; the exact final
  `/home/main/.krasis/cache/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/experts_marlin_int4_g128_calamax.bin`
  was missing; startup detected a Marlin expert-cache write before READY;
  `0` `/v1/internal/reference_test` requests were sent; and the partial
  `experts_marlin_int4_g128_calamax.bin.tmp` plus `.lock` from the aborted
  run were recorded and removed. Scope: use only a built cache path to create
  and verify the exact final `g128` Marlin expert cache required by
  `tests/nemotron-nano-4-4-hqq4-k4v4-a16.conf`, capture the full log, artifact
  path, final size, no `.tmp`/lock files, and whether HQQ attention cache was
  reused without rebuild. No smoke requests, source changes, benchmarks,
  decode/HCS, or production enablement.

- Closed Nemotron Nano Marlin expert-cache reuse prerequisite gate
  `20260627_1711_nemotron_nano_marlin_expert_cache_reuse_prerequisite` as a
  cache-reuse failure before READY. Pre-run inventory found no final
  `experts_marlin_int4_g128_calamax.bin`; only the prior
  `experts_marlin_int4_g64_calamax.bin` and `g64_calsearchrmse` files were
  present, with no `.tmp`/pending files. The startup-only run used exactly
  `./dev run tests/nemotron-nano-4-4-hqq4-k4v4-a16.conf` with no
  build/cache flags and attribution/checkpoint env gates off. HQQ attention
  cache reuse was confirmed, but startup entered `Building GPU INT4 Marlin
  expert cache (one-time)` and `Building GPU INT4 Marlin cache: 23 layers from
  safetensors`, creating a partial `experts_marlin_int4_g128_calamax.bin.tmp`
  and lock before READY. Per gate scope, stopped before requests: exit `137`,
  READY absent, `0` `/v1/internal/reference_test` requests. The partial `.tmp`
  and `.lock` created by this run were recorded and removed. No smoke requests,
  source changes, benchmarks, decode/HCS, or production enablement.

- Opened Nemotron Nano Marlin expert-cache reuse prerequisite gate
  `20260627_1711_nemotron_nano_marlin_expert_cache_reuse_prerequisite`.
  Recorded `1701` first: the HQQ4 attention cache validated on the normal
  server path, but before READY and before any request the run built/wrote the
  missing Marlin INT4 expert cache (`GPU INT4 Marlin cache built: 15.8 GB in
  42s`) at
  `/home/main/.krasis/cache/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/experts_marlin_int4_g64_calamax.bin`;
  the server was killed before smoke, exit `137`, READY absent, `0`
  `/v1/internal/reference_test` requests, and no case `2` was sent. Scope:
  use existing artifacts first, verify the current Marlin expert cache is
  complete/final with no `.tmp` files, then run exactly
  `./dev run tests/nemotron-nano-4-4-hqq4-k4v4-a16.conf` once with no
  build/cache flags and attribution/checkpoint envs off. Stop after READY or
  immediately on any cache write; do not send smoke requests in this gate.

- Closed Nemotron Nano HQQ4+k4v4 non-attribution smoke gate
  `20260627_1701_nemotron_nano_hqq4_k4v4_non_attribution_smoke` as a
  stop-before-smoke. Reused the tests-only config and launched only
  `./dev run tests/nemotron-nano-4-4-hqq4-k4v4-a16.conf`, with no
  `--build-cache` or `--force-rebuild-hqq-cache`. The existing HQQ4 attention
  cache validated (`115 MB cached`, `27,940 MB free`), but before the server
  reached READY the regular path built/wrote the Marlin INT4 expert cache:
  `GPU INT4 Marlin cache built: 15.8 GB in 42s`, updating
  `/home/main/.krasis/cache/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/experts_marlin_int4_g64_calamax.bin`.
  Because this gate required existing generated cache only and no cache
  generation, the server was killed before requests; exit `137`, `0`
  `/v1/internal/reference_test` requests, no case `2`, no selected-token/top-k
  result. Expert-HQQ consume was not enabled because the loaded HQQ4+k4v4 path
  is attention-HQQ and had no expert-HQQ diagnostic cache spec. No source
  changes, benchmarks, decode/HCS, or production enablement.

- Opened Nemotron Nano HQQ4+k4v4 non-attribution smoke gate
  `20260627_1701_nemotron_nano_hqq4_k4v4_non_attribution_smoke`. Recorded
  `1648` first: tests-only config
  `tests/nemotron-nano-4-4-hqq4-k4v4-a16.conf`; complete HQQ4 attention
  manifest at
  `/home/main/.krasis/cache/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/attention_hqq_v5_grid65_33/manifest.json`;
  backend `hqq4`, `nbits=4`, `group_size=128`, layout
  `row_major_axis1_grouped_uint4_packed`, `30` tensor records across layers
  `5,12,19,26,33,42`; cache-generation command exited `0`; and `0` requests
  were sent in the cache-generation prerequisite gate. Scope: run only the
  existing generated cache via `./dev run
  tests/nemotron-nano-4-4-hqq4-k4v4-a16.conf`, send `/v1/internal/reference_test`
  case `2` first with `debug_reference_trace=true`, HQQ consume on if supported,
  and attribution/checkpoint env gates off. Stop on selected-token flip or
  unstable `1044-1321` margin versus BF16/accepted g32; if case `2` passes,
  send cases `0` and `1` in the same server run. No source changes, cache
  generation, benchmarks, decode/HCS, or production enablement.

- Closed Nemotron Nano HQQ4+k4v4 cache-generation prerequisite gate
  `20260627_1648_nemotron_nano_hqq4_k4v4_cache_generation_prerequisite`.
  Created the tests-only config
  `tests/nemotron-nano-4-4-hqq4-k4v4-a16.conf` after verifying the target
  fields against existing working HQQ4+k4v4 configs: `CFG_KV_DTYPE="k4v4"`,
  `CFG_ATTENTION_QUANT="hqq4"`, and `CFG_HQQ_GROUP_SIZE="128"`. Did not touch
  `krasis/testconfigs` or `testconfigs`. Ran only the built cache path
  `./dev run tests/nemotron-nano-4-4-hqq4-k4v4-a16.conf --build-cache
  --force-rebuild-hqq-cache` in tmux. The command exited `0` and printed
  `BUILD CACHE COMPLETE`; HQQ attention artifacts validated with `115 MB`
  cached and `27,940 MB` free. Generated
  `/home/main/.krasis/cache/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/attention_hqq_v5_grid65_33/manifest.json`
  with `complete=true`, backend `hqq4`, `nbits=4`, `group_size=128`,
  layout `row_major_axis1_grouped_uint4_packed`, `30` tensor records across
  full-attention layers `5,12,19,26,33,42`, and manifest tensor bytes
  `120,767,064`. No quality smoke requests, benchmarks, decode/HCS,
  production enablement, variants, or source changes beyond the tests-only
  config.

- Opened Nemotron Nano HQQ4+k4v4 cache-generation prerequisite gate
  `20260627_1648_nemotron_nano_hqq4_k4v4_cache_generation_prerequisite`.
  Recorded `1640` first: Gemma carried speeds remain
  `5619.6 / 92.43 / 155.69`; HQQ6 g32 non-attribution selected-token
  behavior is restored for the three saved cases (`0/3` selected-token
  changes), but top-k drift remains `3/3`; HQQ6 g64 case `2` is
  quality-failing; and no valid existing Nano HQQ+k4v4 config/cache/spec was
  found. Scope: verify exact HQQ4+k4v4 fields against existing working
  k4v4/HQQ4 configs, create a tests-only Nano config under `tests/` without
  touching `krasis/testconfigs`, then run only the built cache path needed to
  create and verify the Nano k4v4 HQQ cache/manifest. Stop after cache/spec
  artifacts and logs; no quality smoke requests in this gate because `1640`
  found the cache/spec missing.

- Closed Nemotron Nano HQQ k4v4 inventory and non-attribution smoke gate
  `20260627_1640_nemotron_nano_hqq_k4v4_inventory_non_attribution_smoke`.
  Used existing artifacts/configs only. Inventory found no unambiguous Nano
  HQQ+k4v4 config/cache/spec to run: Nano configs are BF16 attention with
  `fp8_e4m3` or `k6v6` KV; `/home/main/.krasis/cache/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
  contains only `auto_heatmap.json` and Marlin INT4 expert caches; and the
  existing HQQ diagnostic specs are expert-HQQ g64/g32/g16/HQQ8 specs, not a
  k4v4 target path by themselves. No server was launched and no cache was
  generated. Proposed next gate: create an explicit `tests/` Nano HQQ4+k4v4
  config from the existing Nano k6v6 config, changing only KV dtype,
  attention quant, and HQQ group size, then use the built `./dev run` path with
  `--force-rebuild-hqq-cache` to build/validate the attention-HQQ cache before
  smoke requests. No source changes, benchmarks, decode/HCS, production
  enablement, new variants, or selected-expert retargeting occurred.

- Opened Nemotron Nano HQQ k4v4 inventory and non-attribution smoke gate
  `20260627_1640_nemotron_nano_hqq_k4v4_inventory_non_attribution_smoke`.
  Recorded `1603` first: Gemma carried speeds remain
  `5619.6 / 92.43 / 155.69`; HQQ6 g32 non-attribution selected-token behavior
  is restored for the three saved cases (`0/3` selected-token changes), but
  top-k drift remains `3/3`; HQQ6 g64 is quality-failing and reproduced the
  case-2 flip to token `1321` with margin `1044-1321 = -0.242314` versus BF16
  `0.370508`. Scope: use existing artifacts/configs first and inventory
  whether a valid Nemotron Nano HQQ k4v4 cache/spec already exists. If valid,
  run one server with `/v1/internal/reference_test`, `debug_reference_trace=true`,
  HQQ consume on, and all attribution/checkpoint env gates off; send case `2`
  first and stop on selected-token flip or unstable margin. If the k4v4
  cache/spec is missing or ambiguous, do not generate it in this gate; record
  the exact missing asset and propose a built-command generation gate. No
  source changes, benchmarks, decode/HCS, production enablement, new variants,
  or cache generation.

- Closed HQQ6 g64 non-attribution quality smoke gate
  `20260627_1603_nemotron_nano_hqq6_g64_non_attribution_quality_smoke`.
  Used existing artifacts first, then ran one HQQ6 g64 server with the existing
  all-MoE g64 cache/spec through `/v1/internal/reference_test`,
  `debug_reference_trace=true`, and HQQ consume on. Projection,
  consume-boundary, and read-only checkpoint env gates were off. Sent suspect
  case `2` first; it returned HTTP `200` in `31.267936s`, selected token
  `1321` (`" and"`) instead of BF16 token `1044` (`,`), and margin
  `1044-1321` was `-0.242314` versus BF16 `0.370508`. The new row matches the
  old g64 failure margin/top-k, so cases `0` and `1` were not sent per the
  gate rule. HQQ consume emitted `23` rows and copied `330,882,048` bytes;
  long-prefill min free was `14,070 MB`, still well above the `600 MB` safety
  margin. Decision: record HQQ6 g64 instability/quality failure for the
  non-attribution path. No cache generation, variants, production enablement,
  benchmark, decode/HCS, or source changes occurred.

- Opened HQQ6 g64 non-attribution quality smoke gate
  `20260627_1603_nemotron_nano_hqq6_g64_non_attribution_quality_smoke`.
  Recorded `1540` first: Gemma carried speeds remain
  `5619.6 / 92.43 / 155.69`; HQQ6 g32 non-attribution smoke selected-token
  behavior is restored for the three saved cases (`0/3` selected-token
  changes vs BF16 and old g32), with positive `1044-1321` margins on cases
  `0`, `1`, and `2` (`0.585973`, `0.642010`, `0.253093`). Caveat: top-k
  rank/membership drift remains `3/3`, so production HQQ remains disabled.
  Scope: use existing artifacts first, then run one HQQ6 g64 server using the
  existing g64 cache/spec through `/v1/internal/reference_test`,
  `debug_reference_trace=true`, and HQQ consume on, with projection,
  consume-boundary, and read-only checkpoint env gates off. Send suspect case
  `2` first and stop on a flip or margin instability; only if case `2` passes,
  send cases `0` and `1` in the same server run. No cache generation, variants,
  production enablement, benchmark, decode/HCS, or source changes unless a
  concrete bug is exposed.

- Closed HQQ6 g32 non-attribution quality acceptance gate
  `20260627_1540_nemotron_nano_hqq6_g32_non_attribution_quality_acceptance`.
  Used existing artifacts first, then ran only missing case `1` with the
  existing HQQ6 g32 cache/spec through `/v1/internal/reference_test`,
  `debug_reference_trace=true`, and HQQ consume on. Projection,
  consume-boundary, and read-only checkpoint env gates were off. Case `1`
  returned HTTP `200` in `36.599477s`, selected token `1044` (`,`), and kept a
  positive `1044-1321` margin `0.642010` versus BF16 `0.585266` and old g32
  `0.495824`. Combined with the `1515` cases `0` and `2`, HQQ6 g32
  non-attribution selected-token behavior is restored for the three-case smoke
  set (`0/3` selected-token changes vs BF16 and old g32). Top-k drift remains
  `3/3`, so this is not quality acceptance and production HQQ remains disabled.
  No g64, cache generation, variants, production enablement, benchmark, or
  decode/HCS work occurred.

- Opened HQQ6 g32 non-attribution quality acceptance gate
  `20260627_1540_nemotron_nano_hqq6_g32_non_attribution_quality_acceptance`.
  Recorded `1515` first: HQQ6 g32 production-compatible non-attribution path
  returned `2/2` HTTP `200`, selected token `1044` for cases `0` and `2`,
  kept positive `1044-1321` margins (`0.585973`, `0.253093`), and restored
  selected-token behavior for those checked cases. Caveat: top-k drift remains,
  so this is not yet quality acceptance. Scope: use existing artifacts first,
  then run only missing case `1` with the existing HQQ6 g32 cache/spec through
  `/v1/internal/reference_test`, `debug_reference_trace=true`, and HQQ consume
  on while projection, consume-boundary, and read-only checkpoint env gates
  remain off. Capture HTTP/timing, selected token, top-k, and `1044-1321`
  margin if available, then compare to old g32 quality and BF16. No g64, cache
  generation, variants, production enablement, benchmark, or decode/HCS work.

- Closed HQQ6 g32 production-path stability gate
  `20260627_1515_nemotron_nano_hqq6_g32_production_path_stability`.
  Ran only HQQ6 g32 cases `0` and `2` through
  `/v1/internal/reference_test` with the existing g32 cache/spec,
  `debug_reference_trace=true`, and HQQ consume enabled. Projection,
  consume-boundary, and read-only checkpoint env gates were explicitly off;
  their trace objects were unavailable as intended. Both requests returned
  HTTP `200` (`36.553779s`, `30.943533s`) and selected token `1044` (`,`).
  Margins `1044-1321`: case `0` `0.585973` versus BF16 `0.300524` and old
  g32 `0.474478`; case `2` `0.253093` versus BF16 `0.370508` and old g32
  `0.069721`. HQQ consume emitted `46` rows and copied `663,989,760` bytes.
  Decision: selected-token behavior is restored for the non-attribution path;
  this is not HQQ6 g32 quality acceptance because top-k drift remains. No g64,
  cache generation, new variants, production enablement, benchmark, or
  decode/HCS work occurred.

- Opened HQQ6 g32 production-path stability gate
  `20260627_1515_nemotron_nano_hqq6_g32_production_path_stability`.
  Recorded `1452` first: live read-only checkpoint confirmation selected
  token `1044` (`,`), all `23/23` pre/post layer-delta checkpoint hashes
  matched, no route/top-k/gather/final-hidden mutation point was exposed, and
  the older uncheckpointed `1244` case-0 flip was not reproduced. Scope: run
  only the production-compatible diagnostic path for HQQ6 g32 cases `0` and
  `2` using the existing g32 cache/spec through `/v1/internal/reference_test`
  with `debug_reference_trace=true` and HQQ consume enabled. Projection,
  consume-boundary, and read-only checkpoint env gates must remain off.
  Capture HTTP/timing, selected token, top-k, and `1044-1321` margin, then
  compare with old g32 quality and BF16 margins. Stop on any selected-token
  flip or unstable margin. No g64, cache generation, variants, production
  enablement, benchmark, or decode/HCS work.

- Closed live read-only checkpoint confirmation gate
  `20260627_1452_nemotron_nano_live_read_only_checkpoint_confirmation`.
  Ran only HQQ6 g32 case `0` through `/v1/internal/reference_test` with the
  existing g32 cache/spec, `debug_reference_trace=true`, projection env,
  consume-boundary diagnostics, and `KRASIS_READ_ONLY_CHECKPOINT_DIAG=1`.
  HTTP `200` in `45.388963s`; selected token stayed `1044` (`,`), margin
  `1044-1321` was `0.6258100000000002` versus saved BF16 margin
  `0.300523758`. Both `first_token_margin_projection` and
  `read_only_checkpoints` were emitted. All `23/23` pre/post layer-delta
  checkpoint hashes matched, with final-hidden-before-LM hash
  `0xc0317cde307060bf`; no route/top-k/gather/final-hidden mutation point was
  exposed. The checkpointed run did not reproduce the older uncheckpointed
  `1244` case-0 flip, so no source span was identified for that older flip.
  No g64, cache generation, new variants, production enablement, benchmark, or
  decode/HCS work occurred.

- Opened live read-only checkpoint confirmation gate
  `20260627_1452_nemotron_nano_live_read_only_checkpoint_confirmation`.
  Recorded `1426` first: diagnostic-only read-only checkpoint support was
  implemented behind request-scoped `KRASIS_READ_ONLY_CHECKPOINT_DIAG`;
  startup/warmup/calibration ignore the env; reference-test requests require
  `debug_reference_trace=true`; focused checkpoint test `1/0`,
  `./dev test-kernels expert_hqq` `108/0`, and `./dev build` passed; final
  state was clean with GPUs `15 MB / 11 MB`. Scope: run only HQQ6 g32 case
  `0` through `/v1/internal/reference_test` using the existing g32 cache/spec
  with `debug_reference_trace=true`, projection env, consume-boundary
  diagnostics, and `KRASIS_READ_ONLY_CHECKPOINT_DIAG=1`. Capture HTTP/timing,
  selected token/top-k/margin, `first_token_margin_projection`, and
  `read_only_checkpoints`. Pass/fail hinges on pre/post layer-delta hashes and
  whether route/top-k/gather/final-hidden hashes expose a mutation point. No
  g64, cache generation, new variants, production enablement, benchmark, or
  decode/HCS work.

- Closed read-only checkpoint instrumentation support gate
  `20260627_1426_nemotron_nano_read_only_checkpoint_instrumentation_support`.
  Implemented diagnostic-only read-only checkpoint tracing behind
  `KRASIS_READ_ONLY_CHECKPOINT_DIAG`, request-scoped to reference-test prefill
  and requiring `debug_reference_trace=true`. The trace now emits
  `debug_reference_trace.read_only_checkpoints` with final-hidden-before-LM
  hash, route/top-k/gather hashes, selected expert/output state hashes,
  routed selected-row hashes, and pre/post hashes around consume-boundary
  layer-delta construction. Non-request startup/warmup/calibration ignore the
  env, while reference-test diagnostic requests without debug tracing still
  fail closed. Focused checkpoint test `1/0`, `./dev test-kernels expert_hqq`
  `108/0`, and `./dev build` passed. No live server, cache generation, new
  HQQ variants, production enablement, speed benchmark, decode/HCS, or
  selected-expert retargeting occurred. Combined projection plus
  consume-boundary attribution remains invalid for quality decisions until a
  live read-only checkpoint confirmation proves the combined path is read-only.

- Opened read-only checkpoint instrumentation support gate
  `20260627_1426_nemotron_nano_read_only_checkpoint_instrumentation_support`.
  Recorded `1418` first: Gemma `5619.6 / 92.43 / 155.69`; no source/live
  changes occurred in `1418`; source audit found no combined-only live-buffer
  mutation, scratch reuse, selected expert/output mutation, or production
  behavior change; combined projection plus consume-boundary attribution
  remains invalid for quality decisions. Scope: add only diagnostic-gated
  checkpoint instrumentation for route/top-k/gather hashes, selected
  expert/output state hashes, final-hidden-before-LM hash, and pre/post hashes
  around consume-boundary layer-delta construction. Production paths must
  remain untouched and checkpoint requests must fail closed outside the
  diagnostic gate. Required validation: focused trace/instrumentation tests,
  `./dev test-kernels expert_hqq`, and `./dev build`. No live server in this
  support gate.

- Closed combined-diagnostic read-only root-cause gate
  `20260627_1418_nemotron_nano_combined_diagnostic_read_only_root_cause` as a
  source-audit stop. Audited only projection and consume-boundary diagnostic
  paths. No combined-only live GPU buffer mutation, scratch reuse, selected
  expert/output state mutation, or production behavior change was exposed. The
  only combined-only operation is CPU construction of
  `FirstTokenMarginProjectionLayerDelta` from already-downloaded
  consume-boundary vectors after post-scatter sync. Existing artifacts cannot
  distinguish a true combined diagnostic side effect from HQQ consume
  run-to-run sensitivity. Recorded required next instrumentation:
  route/top-k/gather hashes, final-hidden-before-LM hash, and pre/post
  layer-delta GPU hashes. No source code, live server, g64, cache generation,
  new variants, production enablement, benchmark, decode/HCS, or
  selected-expert retargeting occurred.

- Opened combined-diagnostic read-only root-cause gate
  `20260627_1418_nemotron_nano_combined_diagnostic_read_only_root_cause`.
  Recorded `1328` first: Gemma `5619.6 / 92.43 / 155.69`; the HQQ6 g32
  case-0 selected-token change is isolated to the combined projection plus
  consume-boundary diagnostic path; the single diagnostic flags did not flip
  output. Combined projection plus consume-boundary attribution must not be
  used for quality decisions yet. Scope: source audit only until a concrete
  minimal-confirmation hypothesis exists; inspect projection and
  consume-boundary paths for live-buffer mutation, ordering/synchronization
  changes, scratch reuse, and selected expert/output state mutation. No cache
  generation, new variants, production enablement, benchmarks, decode/HCS, or
  selected-expert retargeting.

- Closed diagnostic-perturbation check gate
  `20260627_1328_nemotron_nano_diagnostic_perturbation_check`. Compared the
  old HQQ6 g32 quality proof and the `1244` projection rerun first, then ran
  only HQQ6 g32 case `0` through `/v1/internal/reference_test` with the
  existing g32 cache/spec. Results: debug-trace-only selected `1044` with
  margin `+0.270762` in `36.943106s`; projection-only selected `1044` with
  margin `+0.522049` in `36.428060s`; consume-boundary-only selected `1044`
  with margin `+0.315884` in `44.020051s` and emitted `69`
  consume-boundary rows. The saved `1244` projection-plus-consume-boundary run
  is the only condition that flipped case `0` to `1321` with margin
  `-0.087939`. Decision: neither projection nor consume-boundary diagnostics
  alone perturb the selected token; the perturbation is isolated to their
  combined diagnostic interaction. No g64, cache generation, new variants,
  production enablement, benchmark, decode/HCS, or selected-expert retargeting
  occurred.

- Opened diagnostic-perturbation check gate
  `20260627_1328_nemotron_nano_diagnostic_perturbation_check`. Recorded `1244`
  first: the projection diagnostic rerun completed with HQQ6 g32/g64 using
  existing caches/specs only, but HQQ6 g32 changed behavior relative to the
  earlier quality proof. The earlier g32 quality proof had `0/3`
  selected-token flips; the `1244` projection/consume diagnostic rerun flipped
  case `0` from token `1044` (`,`) to token `1321` (` and`) while case `2`
  stayed at `1044`. Scope: use existing artifacts first to compare request
  flags/env between the old g32 quality proof and the projection rerun, then
  run only HQQ6 g32 case `0` through `/v1/internal/reference_test` with the
  existing g32 cache/spec. First live check: `debug_reference_trace=true` with
  projection and consume-boundary diagnostics off. If that matches old
  behavior, second check: projection env on with consume-boundary diagnostics
  still off. Stop as soon as the perturbing diagnostic flag is isolated.
  Forbidden: g64, cache generation, new variants, production enablement,
  benchmark, decode/HCS, or selected-expert retargeting.

- Closed projection diagnostic rerun gate
  `20260627_1244_nemotron_nano_projection_diagnostic_rerun`. Reran the
  previously blocked first-token margin projection diagnostic with existing
  HQQ6 g32 and g64 caches/specs only. No BF16 server was rerun. HQQ6 g32 cases
  `0` and `2` returned HTTP `200/200`, emitted valid
  `first_token_margin_projection` traces, then the server was killed cleanly
  before running HQQ6 g64. HQQ6 g64 cases `0` and `2` also returned HTTP
  `200/200` with valid projection traces. Each variant emitted `46` consume
  rows and copied `663,989,760` HQQ output bytes. Results against saved BF16
  margins: g32 flipped case `0` from token `1044` (`,`) to `1321` (` and`) with
  margin shift `-0.3884630203901367` and kept case `2` with shift
  `-0.21515178683447267`; g64 kept case `0` with shift
  `+0.1718997954667969` and flipped case `2` with shift
  `-0.612822532684082`. Top-k movement persisted in all four result rows.
  Projection attribution is distributed: g32 case `0` is led by layers `40`
  and `51`; g64 case `2` is led by layer `38`, with layer `51` contributing
  but not solely explaining the final token margin. The projection summaries
  are immediate selected-row routed-delta projections through the final LM-head
  row difference, not full nonlinear downstream causal decomposition. No cache
  generation, HQQ8, new variants, production enablement, speed benchmark,
  decode/HCS, or selected-expert retargeting occurred.

- Opened projection diagnostic rerun gate
  `20260627_1244_nemotron_nano_projection_diagnostic_rerun`. Recorded `1227`
  first: the request-scoped projection fix is implemented; startup
  warmup/calibration no longer trips `KRASIS_FIRST_TOKEN_MARGIN_PROJECTION_DIAG`;
  `/v1/internal/reference_test` still fails closed when projection is requested
  without `debug_reference_trace=true`; focused `expert_hqq` tests passed
  `104/0`; `./dev build` passed; final cleanup was clean. Scope: rerun only
  the previously blocked diagnostic, HQQ6 g32 cases `0` and `2` first using the
  existing g32 cache/spec and `/v1/internal/reference_test` with
  `debug_reference_trace=true`, projection target tokens `1044` and `1321`, and
  HQQ consume diagnostics. If g32 emits a valid
  `first_token_margin_projection`, kill cleanly and run HQQ6 g64 cases `0` and
  `2` the same way. Capture HTTP status/timing, projection trace, token
  margins, LM-head row-difference projection, per-layer/per-expert
  projections, selected/top-k movement, and compare to saved BF16 margins.
  Forbidden: cache generation, HQQ8, new variants, production enablement,
  speed benchmark, decode/HCS, or selected-expert retargeting.

- Closed projection request-scoping fix gate
  `20260627_1227_nemotron_nano_projection_request_scoping_fix`. Implemented a
  request-scoped first-token margin projection activation flag in the Rust
  prefill engine. Startup warmup/calibration keep the flag disabled, so
  projection env no longer affects non-request prefill. The Rust
  `/v1/internal/reference_test` handler enables the flag only around
  reference-test prefill and clears it on setup failure and immediately after
  prefill; the existing fail-closed behavior is preserved for diagnostic
  requests that set projection env without `debug_reference_trace=true`.
  Focused tests cover non-request suppression, missing-debug fail-closed
  behavior, and debug-enabled projection target parsing/trace shape.
  Validation: `./dev test-kernels expert_hqq` passed `104/0`; `./dev build`
  passed. No live diagnostic rerun, cache generation, HQQ8, new variants,
  production enablement, speed benchmark, decode/HCS, or selected-expert
  retargeting occurred.

- Opened projection request-scoping fix gate
  `20260627_1227_nemotron_nano_projection_request_scoping_fix`. Recorded
  `1216` first: the first-token margin projection diagnostic run failed closed
  before requests during startup short VRAM calibration with
  `RuntimeError: KRASIS_FIRST_TOKEN_MARGIN_PROJECTION_DIAG requires
  debug_reference_trace=true`; no `/v1/internal/reference_test` request was
  sent, no projection trace was emitted, and HQQ6 g64 was not run. Scope:
  implement only the missing request-scoping fix so the projection env does not
  affect startup warmup/calibration, while actual reference-test diagnostic
  requests still fail closed when projection is requested without
  `debug_reference_trace=true`. Prefer server/request-scoped enable/clear over
  weakening the Rust fail-closed behavior. Required validation:
  `./dev test-kernels expert_hqq` and `./dev build`. Forbidden: live diagnostic
  rerun, cache generation, HQQ8, new variants, production enablement, speed
  benchmark, decode/HCS, or selected-expert retargeting.

- Closed first-token margin projection diagnostic run gate
  `20260627_1216_nemotron_nano_first_token_margin_projection_diagnostic_run`
  as fail-closed before requests. Recorded `1145` first, then attempted only
  the HQQ6 g32 run with existing g32 cache/spec and projection env. The server
  failed closed before READY during startup short VRAM calibration with
  `RuntimeError: KRASIS_FIRST_TOKEN_MARGIN_PROJECTION_DIAG requires
  debug_reference_trace=true`; warmup had logged the same condition as a
  warning. No `/v1/internal/reference_test` request was sent, no projection
  trace was emitted, and HQQ6 g64 was not run. Decision: the projection gate is
  currently global, while startup warmup/calibration call Rust prefill without
  reference-debug request context. Next fix must make projection activation
  request-scoped for reference-test requests, or suppress projection target
  evaluation during non-request startup warmup/calibration while preserving the
  fail-closed requirement for actual diagnostic requests without
  `debug_reference_trace=true`. No cache generation, HQQ8, new variants,
  production enablement, speed benchmark, decode/HCS, or selected-expert
  retargeting occurred.

- Opened first-token margin projection diagnostic run gate
  `20260627_1216_nemotron_nano_first_token_margin_projection_diagnostic_run`.
  Recorded `1145` first: Gemma carried speeds `5619.6 / 92.43 / 155.69`;
  first-token margin projection support was implemented diagnostic-only behind
  `KRASIS_FIRST_TOKEN_MARGIN_PROJECTION_DIAG`, requires
  `debug_reference_trace=true` plus target-token env values, focused
  `expert_hqq` tests passed `101/0`, `./dev build` passed, and final cleanup
  was clean with no tmux server, no matching runtime/build/generator process,
  no NVIDIA compute process, and GPUs at `15 MB / 11 MB`. Scope: run only
  HQQ6 g32 and HQQ6 g64 for cases `0` and `2` through
  `/v1/internal/reference_test` using existing caches/specs, with target
  tokens `1044` and `1321`, and capture
  `debug_reference_trace.first_token_margin_projection`, token margins,
  LM-head row-difference projection, per-layer routed-delta projections,
  per-expert contributor projections where present, selected/top-k movement,
  and HTTP status/timing. Forbidden: cache generation, HQQ8, new variants,
  production enablement, speed benchmark, decode/HCS, or selected-expert
  retargeting.

- Closed first-token margin projection support gate
  `20260627_1145_nemotron_nano_first_token_margin_projection_support`.
  Implemented diagnostic-only first-token margin projection instrumentation,
  gated by `KRASIS_FIRST_TOKEN_MARGIN_PROJECTION_DIAG` and requiring
  `debug_reference_trace=true`. The trace now exposes the requested LM-head
  row-difference vector, selected-position hidden row, raw and softcapped
  target margin projection, and per-layer/per-expert BF16-vs-HQQ contribution
  summaries when consume-boundary data is present. The path fails closed for
  missing target-token env, identical tokens, disabled reference debug trace,
  unavailable BF16 LM-head rows, missing hidden/logit capture, or mismatched
  delta vector lengths. Validation: `./dev test-kernels expert_hqq` passed
  `101/0`; `./dev build` passed; formatter/syntax/diff checks passed. No live
  server, cache generation, speed benchmark, production HQQ enablement, HQQ
  variant change, decode/HCS change, or selected-expert retargeting occurred.

- Opened first-token margin projection support gate
  `20260627_1145_nemotron_nano_first_token_margin_projection_support`.
  Recorded `1134` first: Gemma carried speeds `5619.6 / 92.43 / 155.69`;
  targeted first-token margin attribution was data-only and used existing
  saved artifacts with no live rerun. Case 0 `1044-1321` margin: BF16
  `+0.300523758`, HQQ6 g64 `+0.472423553`, HQQ6 g32 `+0.474477768`. Case 2:
  BF16 `+0.370508194`, HQQ6 g64 `-0.242314339` after shift
  `-0.612822533` and selected-token flip to `1321`, HQQ6 g32 `+0.069721222`
  after shift `-0.300786972` and selected token remains `1044`. Exact missing
  instrumentation from `1134`: LM-head target row difference for tokens `1044`
  and `1321`, full selected-row hidden vector at final LM-head input,
  per-layer BF16-vs-HQQ selected-row hidden deltas, routing/contributor
  metadata tied to those deltas, and projection of each delta onto the target
  token margin. Scope: implement only diagnostic-gated instrumentation for
  margin projection support; keep it off production paths and fail closed when
  requested without the diagnostic gate. Forbidden: server runs, cache
  generation, benchmarks, HQQ variant changes, decode/HCS work, or expert
  retargeting.

- Closed targeted first-token margin attribution gate
  `20260627_1134_nemotron_nano_first_token_margin_attribution` as data-only.
  Used existing saved BF16/HQQ6 g64/HQQ6 g32 artifacts only; ran no live
  server and generated no cache. Token-level margin was available from saved
  first-token top-k/logit traces. Case 0: BF16 `1044-1321` margin
  `+0.300523758`; HQQ6 g64 `+0.472423553`; HQQ6 g32 `+0.474477768`, so both
  kept token `1044` and widened the target margin. Case 2: BF16 margin
  `+0.370508194`; HQQ6 g64 shifted by `-0.612822533` to `-0.242314339` and
  flipped to token `1321`; HQQ6 g32 shifted by `-0.300786972` but remained
  `+0.069721222` and kept token `1044`. Available per-layer evidence remains
  HQQ6 g64 case-2 expert-output attribution (`0640`, layer 51 expert 60), but
  current artifacts do not expose exact per-layer/per-expert contribution to
  the final `1044` vs `1321` token margin. Decision: stop and record missing
  instrumentation; an unmodified live rerun would not provide the missing
  LM-head target row difference, full selected-row hidden vector, per-layer
  BF16-vs-HQQ hidden deltas, and projection onto the target token margin.

- Opened targeted first-token margin attribution gate
  `20260627_1134_nemotron_nano_first_token_margin_attribution`. Recorded
  `1125` first: the HQQ variant decision/root-cause gate concluded that no
  tested HQQ variant passes Nano quality acceptance and further blind HQQ
  bit/group sweeps are not justified. HQQ6 g32 is least-bad by selected-token
  stability, HQQ6 g64 is least-bad by logprob distance but flips case 2, and
  the common first-token boundary is token `1044` (`,`) versus token `1321`
  (`" and"`). Scope: use existing saved artifacts first; if a live diagnostic
  is required, run only HQQ6 g32 and HQQ6 g64 for cases `0` and `2` against
  `/v1/internal/reference_test` using existing diagnostic specs/caches. Capture
  first-token margins for tokens `1044` and `1321`, top-k membership/rank
  movement, selected/top-k logprob deltas, and any available per-layer/expert
  attribution. Forbidden: cache generation, HQQ8, new bit/group sweeps,
  production enablement, speed benchmark, decode/HCS, selected-expert
  retargeting, or workaround instrumentation if margin attribution is not
  currently exposed.

- Opened HQQ variant decision/root-cause gate
  `20260627_1125_nemotron_nano_hqq_variant_decision_root_cause`. Recorded
  `1058` first: guarded HQQ8 g64 quality proof returned `3/3` HTTP 200 through
  `/v1/internal/reference_test`, request times `39.900834s`, `35.062542s`,
  and `39.806775s`, emitted `69` consume rows, copied `1,029,740,544` bytes,
  used `nbits=8`, `group_size=64`, layout `row_major_axis1_grouped_uint8`,
  changed selected token on `2/3` payloads, kept top-k rank drift at `3/3`,
  and was not a material Nano quality improvement. Compared selected-token
  changes g64/g32/g16/HQQ8 were `1/3`, `0/3`, `0/3`, `2/3`; average selected
  output logprob deltas were `0.110919667`, `0.125852000`, `0.339994333`,
  `0.287812000`; max common top-k deltas were `0.493397000`, `0.578915000`,
  `0.571293000`, `0.570168000`; max rank-aligned top-k deltas were
  `0.429088000`, `0.407414000`, `0.571293000`, `0.489098000`. `1058` made no
  production HQQ, speed benchmark, fallback, decode/HCS, selected-expert
  routing, INT4/HQQ4, or k/v changes. Scope: consolidate existing HQQ6
  g64/g32/g16 and HQQ8 g64 saved response/log artifacts only; identify the
  least-bad candidate and common failure pattern; propose a next diagnostic
  gate only if evidence points to one. Forbidden: server run, cache generation,
  production enablement, speed benchmark, fallback, decode/HCS, selected-expert
  routing, INT4/HQQ4, or k/v work. Closed data-only decision gate. Consolidated
  saved HQQ6 g64/g32/g16 and HQQ8 g64 quality artifacts without server or cache
  runs. HQQ6 g32 is least-bad for selected-token stability (`0/3` flips and
  lower aggregate selected delta than g16), while HQQ6 g64 is least-bad by
  logprob distance but flips the selected token on case 2. No candidate passes
  acceptance: every variant has `3/3` top-k rank drift, and quality is
  non-monotonic across group/bit changes. Common failure pattern is a top-2
  first-token boundary between BF16 token `1044` (`,`) and token `1321`
  (`" and"`); all selected-token flips are only this pair, and case 2 has
  top-k membership drift for all variants. Decision: further blind HQQ
  bit/group sweeps are not justified. If continuing, next gate should be a
  targeted first-token margin attribution on cases 0 and 2 for tokens `1044`
  vs `1321`, comparing HQQ6 g32 and HQQ6 g64 only, with no new cache generation
  or broad variant sweep.

- Opened guarded HQQ8 g64 quality proof gate
  `20260627_1058_nemotron_nano_hqq8_g64_quality_proof`. Recorded `1038`
  first: generator status `ok`, cache bytes `33,047,177,280`, payload bytes
  `33,046,659,072`, `5,888` tensor records, `nbits=8`, `group_size=64`,
  layout `row_major_axis1_grouped_uint8`, after-generation headroom `614G`
  disk / `964Gi` RAM, final clean state, carried Gemma speeds
  `5619.6 / 92.43 / 155.69`, and no source changes, live server, production
  HQQ, speed benchmark, fallback, decode/HCS, selected-expert routing,
  INT4/HQQ4, or k/v work in `1038`. Scope: reuse the saved BF16 `1023`
  baseline responses and same three payloads; run one guarded HQQ-consume live
  pass with the HQQ8 g64 diagnostic spec only; submit requests only to
  `/v1/internal/reference_test`; capture HTTP status/timing, responses,
  consume rows, copied bytes, `nbits/group_size/layout`; compare HQQ8 against
  BF16, HQQ6 g64, HQQ6 g32, and HQQ6 g16 for selected-token changes, top-k rank
  drift, selected logprob deltas, and top-k logprob deltas. Forbidden: BF16
  server rerun, production HQQ, speed benchmark, fallback, decode/HCS,
  selected-expert routing, INT4/HQQ4, or k/v work. Closed guarded quality
  proof. Reused saved BF16 `1023` baselines and ran only the HQQ8 g64 guarded
  consume live pass through `/v1/internal/reference_test`. Runtime returned
  `3/3` HTTP 200 with request times `39.900834s`, `35.062542s`, and
  `39.806775s`; emitted `69` consume rows; copied `1,029,740,544` bytes; all
  consume rows were `nbits=8`, `group_size=64`, layout
  `row_major_axis1_grouped_uint8`. HQQ8 g64 changed selected tokens on `2/3`
  payloads (`case0` and `case2`, BF16 token `1044` to HQQ8 token `1321`) and
  top-k rank drift remained `3/3`. Compared with HQQ6 variants: selected-token
  changes g64/g32/g16/HQQ8 were `1/3`, `0/3`, `0/3`, `2/3`; average selected
  output logprob deltas were `0.110919667`, `0.125852000`, `0.339994333`,
  `0.287812000`; max common top-k deltas were `0.493397000`,
  `0.578915000`, `0.571293000`, `0.570168000`; max rank-aligned top-k deltas
  were `0.429088000`, `0.407414000`, `0.571293000`, `0.489098000`.
  Decision: HQQ8 g64 is not a material Nano quality improvement. No BF16
  server, production HQQ, speed benchmark, fallback, decode/HCS,
  selected-expert routing, INT4/HQQ4, or k/v work occurred.

- Opened HQQ8 g64 all-MoE-layer cache/spec generation prerequisite gate
  `20260627_1038_nemotron_nano_hqq8_g64_all_moe_layer_cache_spec_generation`.
  Recorded `1007` first: HQQ8 diagnostic support was implemented in the
  diagnostic path only, existing HQQ6 behavior was preserved, HQQ8 is currently
  accepted for g64 only, `./dev test-kernels expert_hqq` passed `98/0`,
  `./dev build` passed, final headroom was `645G` disk / `965Gi` RAM, and
  Gemma carried speeds are `5619.6 / 92.43 / 155.69`. Scope: re-check current
  disk/RAM; if sufficient, derive the HQQ8 g64 all-layer manifest from the
  existing all-layer template by changing only `nbits=8`, `group_size=64`, and
  output cache/spec paths; run only
  `./dev expert-hqq-cache-generate <hqq8-g64-manifest>`; capture generator log,
  exact output paths, cache/payload bytes, tensor records, layer/expert counts,
  quantization metadata, and before/after headroom; then run lightweight
  metadata/spec validation only. Forbidden: live server, production HQQ, speed
  benchmark, fallback, decode/HCS, selected-expert routing, INT4/HQQ4, and k/v
  work. Closed cache/spec generation prerequisite. Current headroom was
  sufficient (`645G` disk / `965Gi` RAM before generation). Generated with the
  built command only:
  `./dev expert-hqq-cache-generate benchmarks/20260627_1038_nemotron_nano_hqq8_g64_all_moe_layer_cache_spec_generation_all_layer_generation_manifest.json`.
  Result: status `ok`, exit `0`, `23` MoE layers, `128` experts/layer,
  `5,888` tensor records, payload bytes `33,046,659,072`, cache bytes
  `33,047,177,280`, `nbits=8`, `group_size=64`, layout
  `row_major_axis1_grouped_uint8`. After-generation headroom was `614G` disk /
  `964Gi` RAM. Lightweight metadata/spec validation passed. No source changes,
  live server, production HQQ, speed benchmark, fallback, decode/HCS,
  selected-expert routing, INT4/HQQ4, or k/v work occurred.

- Opened HQQ8 diagnostic-support implementation gate
  `20260627_1007_nemotron_nano_hqq8_diagnostic_support_implementation`.
  Recorded `1000` first: HQQ8 feasibility stopped before generator enablement
  because the expert-HQQ generator/reference/runtime diagnostic path was not
  HQQ8-ready end to end, even though storage was practical
  (`33,047,177,280` estimated cache bytes, `30.78 GiB`, with `645G` disk /
  `966Gi` RAM headroom) and low-level CUDA already had
  `hqq8_prefill_gemm_bf16_kernel`. `1000` made no HQQ8 source support changes,
  generated no HQQ8 cache, ran no live server, and did not enable production
  HQQ. Scope: implement only the missing diagnostic-path support for HQQ8:
  generator manifest acceptance, safetensors/cache spec quantizer path,
  reference dequant/dispatch, and guarded runtime launch plumbing. Preserve
  HQQ6 behavior, keep production HQQ disabled, fail closed for unsupported bit
  widths/groups, add focused tests for HQQ8 accepted, HQQ6 unchanged, invalid
  nbits rejected, and reference/runtime metadata validation, then run
  `./dev test-kernels expert_hqq` and `./dev build`. Forbidden: HQQ8 cache
  generation, live server, benchmark, decode/HCS, selected-expert routing,
  INT4/HQQ4, k/v work, or production HQQ enablement. Closed diagnostic support
  implementation. HQQ8 support is now wired through the diagnostic path only:
  manifest generation accepts HQQ8 g64, safetensors quantization emits
  row-major uint8 with `qmax=255`, reference/test dispatch dequantizes HQQ8,
  and guarded runtime diagnostic launch validates `row_major_axis1_grouped_uint8`
  and selects `hqq8_prefill_gemm_bf16_kernel`. HQQ6 `[16,32,64]` behavior is
  preserved; HQQ8 non-g64 and unsupported bit widths still fail closed.
  Validation passed: `./dev test-kernels expert_hqq` `98/0`, `./dev build`,
  `cargo fmt --check`, `bash -n dev`, and diff checks. No HQQ8 cache
  generation, live server, production HQQ, benchmark, fallback, decode/HCS,
  selected-expert routing, INT4/HQQ4, or k/v work occurred.

- Opened HQQ8 feasibility/support gate
  `20260627_1000_nemotron_nano_hqq8_feasibility_support`. Recorded `0929`
  first: HQQ6 g16 guarded quality proof returned `3/3` HTTP 200, emitted `69`
  consume rows, had `0/3` selected-token changes and `3/3` top-k rank drift,
  and was not a material Nano quality improvement because aggregate deltas were
  worse than g64/g32: average selected-output logprob delta g64 `0.110919667`,
  g32 `0.125852000`, g16 `0.339994333`; max rank-aligned top-k delta g64
  `0.429088000`, g32 `0.407414000`, g16 `0.571293000`. No source changes,
  production HQQ enablement, BF16 server rerun, speed benchmark, fallback,
  decode/HCS, selected-expert retargeting, HQQ8, or INT4/HQQ4 work occurred in
  `0929`. Scope: inspect every `nbits == 6` / HQQ6 assumption in generator,
  cache/spec writer-reader, reference dequant, runtime launch descriptors, and
  CUDA kernels; estimate HQQ8 cache size/disk/RAM before any generation; if the
  path is descriptor-generic, implement only narrow HQQ8 generator/test support
  and run `./dev test-kernels expert_hqq` plus `./dev build`; if runtime/kernel
  is HQQ6-specific, stop and record the unsupported path. Forbidden: cache
  generation, live server, production HQQ, speed benchmark, fallback,
  decode/HCS, selected-expert retargeting, INT4/HQQ4, and k/v work. Closed as a
  support stop before generator enablement. HQQ8 cache size/headroom is
  practical for the all-MoE g64 estimate (`33,047,177,280` cache bytes,
  `30.78 GiB`; current headroom `645G` disk / `966Gi` RAM), and low-level CUDA
  has `hqq8_prefill_gemm_bf16_kernel`, but the expert-HQQ path is not
  descriptor-generic: manifest planning requires `nbits=6`, safetensors builder
  and quantizer support only HQQ4/HQQ6, reference dequant/dispatch reject HQQ8,
  and guarded expert-HQQ runtime diagnostic validation/launch accepts only
  HQQ4/HQQ6. No HQQ8 source support, cache generation, live server, production
  HQQ, speed benchmark, fallback, decode/HCS, selected-expert retargeting,
  INT4/HQQ4, or k/v work occurred.

- Opened guarded HQQ6 g16 quality proof gate
  `20260627_0929_nemotron_nano_hqq6_g16_quality_proof`. Recorded `0909`
  first: all-MoE-layer HQQ6 g16 cache/spec generation completed with
  `status=ok`; cache bytes `36,719,028,288`; tensor records `5,888`;
  `nbits=6`; `group_size=16`; after-generation headroom `645G` disk /
  `964Gi` RAM; no live server and no source changes occurred in `0909`.
  Scope: reuse saved BF16 `1023` baseline responses, run one guarded
  HQQ-consume live pass with the g16 diagnostic spec for the same three
  payloads, and compare g16 against BF16, g64, and g32 for selected token
  changes, top-k rank drift, selected/top-k logprob deltas, consume rows,
  copied bytes, and material quality improvement. Forbidden: BF16 server,
  production HQQ, speed benchmark, fallback, decode/HCS, selected-expert
  retargeting, HQQ8, and INT4/HQQ4 work. Closed quality proof. Did not run a
  BF16 server; reused the saved BF16 `1023` baseline responses. Ran one
  guarded HQQ-consume live pass with the HQQ6 g16 diagnostic spec; the first
  chat endpoint attempt returned `3/3` HTTP 400 because the saved requests are
  raw `input_token_ids`, then the corrected `/v1/internal/reference_test` run
  returned `3/3` HTTP 200. Runtime emitted `69` consume rows (`23` MoE layers x
  `3` payloads), all `nbits=6`, `group_size=16`, and copied `1,029,740,544`
  bytes. G16 kept the selected token stable on all three payloads (`0/3`
  selected-token changes), matching g32 and improving over g64 (`1/3`), but
  top-k rank drift remained `3/3`. Aggregate quality was worse: average
  selected-output logprob delta was g64 `0.110919667`, g32 `0.125852000`, g16
  `0.339994333`; max rank-aligned top-k delta was g64 `0.429088000`, g32
  `0.407414000`, g16 `0.571293000`. Decision: HQQ6 g16 is not a material Nano
  quality improvement. No production HQQ enablement, BF16 server rerun, speed
  benchmark, fallback, decode/HCS, selected-expert retargeting, HQQ8, or
  INT4/HQQ4 work occurred.

- Opened all-MoE-layer HQQ6 g16 cache/spec generation prerequisite gate
  `20260627_0909_nemotron_nano_hqq6_g16_all_moe_layer_cache_spec_generation`.
  Recorded `0853` first: focused `./dev test-kernels expert_hqq` passed
  `94/0`; `./dev build` passed; HQQ6 g16 is feasible through the
  descriptor-driven HQQ6 path; generation allowlist is `[16, 32, 64]`; HQQ8
  remains rejected by the `nbits=6` contract; estimated all-MoE g16 cache is
  `36,719,028,288` bytes (`34.19 GiB`); final `0853` headroom was `680G`
  disk and `965Gi` RAM; and no live server or cache generation ran in `0853`.
  Scope: re-check current disk/RAM, create the g16 manifest from the existing
  all-layer HQQ6 g32/g64 template by changing only `group_size=16` and output
  paths, run only `./dev expert-hqq-cache-generate <g16-manifest>`, and record
  generator log, cache/spec paths, exact bytes, tensor/block counts,
  `nbits=6`, `group_size=16`, and before/after disk/RAM. Forbidden: live
  server, production HQQ, speed benchmark, fallback, decode/HCS,
  selected-expert retargeting, HQQ8, and INT4/HQQ4 work. Closed passed.
  Current headroom before generation was `680G` disk and `965Gi` RAM. Generated
  the all-MoE-layer HQQ6 g16 cache/spec through the built command only:
  `./dev expert-hqq-cache-generate benchmarks/20260627_0909_nemotron_nano_hqq6_g16_all_moe_layer_cache_spec_generation_all_layer_generation_manifest.json`.
  Generator exited `0` with `status=ok`. Output cache:
  `36,719,028,288` bytes; payload bytes: `36,718,510,080`; MoE layers: `23`;
  experts/layer: `128`; layer-expert refs: `2,944`; tensor records: `5,888`;
  `nbits=6`; `group_size=16`. Lightweight metadata/spec validation passed for
  manifest, diagnostic spec, and cache size. Headroom after generation was
  `645G` disk and `964Gi` RAM. No source changes were required, so focused
  tests/build were not rerun in this gate. No live server, production HQQ,
  speed benchmark, fallback, decode/HCS, selected-expert retargeting, HQQ8, or
  INT4/HQQ4 work occurred.

- Opened HQQ6 g16 feasibility/support gate
  `20260627_0853_nemotron_nano_hqq6_g16_feasibility_support`. Recorded
  `0826` first: carried Gemma speeds `5619.6 / 92.43 / 155.69`; HQQ6 g32
  guarded consume returned `3/3` HTTP 200, emitted `69` consume rows, and
  copied `1,029,740,544` HQQ output bytes; selected-token changes improved
  versus g64 from `1/3` to `0/3`, but top-k rank drift remained `3/3`.
  Aggregate quality worsened versus g64: average selected-output logprob delta
  `0.110919667 -> 0.125852000`, and max common top-k delta
  `0.493397000 -> 0.578915000`. Scope: inspect whether generator/cache/spec
  and runtime HQQ6 dequant paths can safely support `group_size=16` through the
  same descriptor-driven HQQ6 path; estimate cache size, disk, and RAM before
  any generation; if safe, implement only narrow generator/test support for
  HQQ6 g16 while preserving existing HQQ6 g32/g64 behavior and keeping
  `nbits != 6` fail-closed. Forbidden: live server, cache generation, HQQ8,
  production HQQ, speed benchmark, fallback, decode/HCS. Closed passed. Runtime
  inspection found no HQQ6 g16 blocker: cache/spec writer-reader, reference
  dequant, runtime diagnostic contract, runtime GEMM launch, and CUDA HQQ6
  dequant/GEMM derive layout from descriptor `group_size`; the CUDA helper only
  special-cases `128`, while g16 uses the existing division path. Implemented
  only generator/test support by extending the HQQ6 generation allowlist to
  `[16, 32, 64]` and adding a valid HQQ6 g16 cache/spec fixture; HQQ6 g32/g64
  behavior remains covered and HQQ8 remains rejected by `nbits=6`.
  `./dev test-kernels expert_hqq` passed `94/0`; `./dev build` passed. G16
  all-MoE cache estimate before generation is `36,719,028,288` bytes
  (`34.19 GiB`) with final headroom `680G` disk and `965Gi` RAM. No live
  server, cache generation, HQQ8 work, production HQQ, speed benchmark,
  fallback, or decode/HCS change occurred.

- Opened guarded HQQ6 g32 quality proof gate
  `20260627_0826_nemotron_nano_hqq6_g32_quality_proof`. Recorded `0804`
  first: all-MoE-layer HQQ6 g32 cache/spec generation succeeded through the
  built command; cache bytes `29,375,326,272`; tensor records `5,888`;
  `nbits=6`; `group_size=32`; headroom moved from `707G` disk / `966Gi` RAM
  before generation to `680G` disk / `962Gi` RAM after generation, with final
  cleanup RAM `965Gi`; no source changes and no live server ran in `0804`.
  Carried Gemma speeds: `5619.6 / 92.43 / 155.69`. Scope: reuse existing
  saved `1023` BF16 baseline responses, run one guarded HQQ-consume live pass
  using the new g32 diagnostic spec for the same three payloads, compare
  selected token, top-k ranks, selected/top-k logprob deltas, and material
  improvement versus HQQ6 g64. Forbidden: BF16 server rerun, production HQQ,
  speed benchmark, INT4/HQQ4, fallback, decode/HCS, and selected-expert
  retargeting. Closed quality proof. Reused saved BF16 `1023` baselines and
  ran only the guarded HQQ-consume live pass with the HQQ6 g32 diagnostic spec;
  all `3/3` requests returned HTTP 200. The run emitted `69` consume rows
  (`23` layers x `3` payloads), all `nbits=6`, `group_size=32`, copying
  `1,029,740,544` output bytes. G32 removed the g64 case-2 selected-token flip:
  all cases selected token `1044` `","` like BF16, while g64 changed case 2 to
  `1321` `" and"`. However, all `3/3` payloads still had top-k rank drift;
  average selected-output logprob delta worsened from g64 `0.110919667` to g32
  `0.125852000`, and max common top-k delta worsened from g64 `0.493397000` to
  g32 `0.578915000`. Decision: HQQ6 g32 is a selected-token improvement versus
  g64 but not a material Nano quality acceptance. No BF16 server, production
  HQQ, speed benchmark, INT4/HQQ4, fallback, decode/HCS, or selected-expert
  retargeting occurred.

- Opened all-MoE-layer HQQ6 g32 cache/spec generation prerequisite gate
  `20260627_0804_nemotron_nano_hqq6_g32_all_moe_layer_cache_spec_generation`.
  Recorded `0749` first: focused `./dev test-kernels expert_hqq` passed
  `93/0`, `./dev build` passed, HQQ6 generation accepts only `group_size`
  `32|64`, HQQ8 remains rejected by the `nbits=6` contract, current headroom
  before generation is `707G` disk and `966Gi` available RAM, and the `0749`
  gate ran no live server or all-layer cache generation. Scope: use the
  existing all-layer HQQ6 g64 manifest/spec shape as the template, change only
  HQQ6 g32 parameters and output paths, run only the built
  `./dev expert-hqq-cache-generate ...` entrypoint, capture generator log,
  cache/spec paths, exact byte size, tensor/block counts, `nbits=6`,
  `group_size=32`, and before/after disk/RAM. Forbidden: live server,
  production HQQ, speed benchmark, fallback, decode/HCS, selected-expert
  retargeting, and INT4/HQQ4 work. Closed passed. Generated the all-MoE-layer
  HQQ6 g32 cache/spec through the built command only:
  `./dev expert-hqq-cache-generate benchmarks/20260627_0804_nemotron_nano_hqq6_g32_all_moe_layer_cache_spec_generation_all_layer_generation_manifest.json`.
  The generator exited `0` with `status=ok`. Output cache:
  `29,375,326,272` bytes; payload bytes: `29,374,808,064`; MoE layers: `23`;
  experts/layer: `128`; layer-expert refs: `2,944`; tensor records: `5,888`;
  `nbits=6`; `group_size=32`. Lightweight metadata validation passed for the
  manifest and diagnostic spec. Headroom changed from `707G` disk / `966Gi`
  RAM available before generation to `680G` disk / `962Gi` RAM available after
  generation. No source changes were required, so focused tests/build were not
  rerun in this gate. No live server, production HQQ, speed benchmark, fallback,
  decode/HCS, selected-expert retargeting, or INT4/HQQ4 work occurred.

- Opened HQQ6 g32 generator-support gate
  `20260627_0749_nemotron_nano_hqq6_g32_generator_support`. Recorded `0737`
  first: HQQ6 g32 is the next candidate after HQQ6 g64 proved
  implementation-correct but quality-failing for Nano; the current generator
  rejected HQQ6 g32 with `requires group_size=64, got 32`; HQQ8 g64 remained
  fail-closed with `requires nbits=6 for HQQ6, got 8`; estimated HQQ6 g32
  all-MoE cache size is `29,375,326,272` bytes (`27.36 GiB`) with prior
  headroom `707G` disk and `965Gi` RAM; no live server, full cache generation,
  or production enablement occurred. Scope: inspect every `group_size == 64`
  assumption in generator, manifest validation, cache/spec writer/reader, and
  runtime HQQ6 dequant path; implement only narrow HQQ6 `group_size=32`
  support while preserving HQQ6 g64 behavior and keeping `nbits != 6`
  fail-closed; add focused g32/g64/invalid-group/HQQ8 tests; run
  `./dev test-kernels expert_hqq` and `./dev build`; re-record disk/RAM
  headroom before any large cache generation. Forbidden: live server, full
  all-layer cache generation before tests/build/headroom, production HQQ,
  speed benchmark, fallback, decode/HCS, and selected-expert retargeting.
  Closed passed. The only g64 blocker found in the generator path was the
  manifest planner; cache descriptor writer/reader, diagnostic spec
  writer/reader, reference dequant, runtime diagnostic contract, runtime HQQ6
  GEMM launch, and CUDA HQQ6 dequant/GEMM already derive layout/strides from
  descriptor `group_size`. Implemented an explicit HQQ6 generation group-size
  allowlist `[32, 64]`, preserving HQQ6 g64 and keeping `nbits != 6`
  fail-closed. Added focused tests for valid HQQ6 g32 cache/spec generation,
  preserved HQQ6 g64 generation, invalid group sizes `0` and `16`, and HQQ8
  rejection. Validation: `./dev test-kernels expert_hqq` passed `93/0` and
  `./dev build` passed. Re-recorded headroom after tests/build: `707G` disk
  available and `965Gi` RAM available. No live server, full all-layer cache
  generation, production HQQ, speed benchmark, fallback, decode/HCS, or
  selected-expert retargeting.

- Opened next-quantization-variant prerequisite gate
  `20260627_0737_nemotron_nano_next_quantization_variant_prerequisite`.
  Recorded `0728` first: HQQ6 g64 is implementation-correct but quality-failing
  for Nano, case 2 selected token changed `1044 -> 1321`, all three saved
  `1023` payloads showed top-k rank drift, and HQQ was not enabled for
  production. Scope: inspect existing KRHQ/HQQ cache-generator parameters and
  artifacts to identify the smallest next higher-quality variant to test,
  likely reduced group size or higher bit-width before any k4v4/k6v6 work. Do
  not generate a large cache until generator support and artifact-size/disk/RAM
  estimates are confirmed. If supported, create only the prerequisite cache/spec
  generation gate and focused tests; if unsupported, stop with exact generator
  changes required. Forbidden: live server, production HQQ, speed benchmark,
  fallback, decode/HCS, and selected-expert retargeting. Closed
  inspection/probe-only as blocked by current generator support. HQQ6 g32 is
  the smallest next higher-quality candidate: same HQQ6 path with group size
  reduced `64 -> 32`. Estimated all-MoE cache size is `29,375,326,272` bytes
  (`27.36 GiB`), `+3,671,851,008` payload bytes over HQQ6 g64, with enough
  current headroom (`707G` disk, `965Gi` RAM). Built-command probes failed
  closed before cache generation: HQQ6 g32 rejected with `requires
  group_size=64, got 32`; HQQ8 g64 rejected with `requires nbits=6 for HQQ6,
  got 8`. HQQ8 g64 is larger (`33,047,177,280` estimated cache bytes) and
  broader because builder/reference/runtime validation are HQQ4/HQQ6-only.
  Next gate should add HQQ6 group-size manifest support and focused valid/
  fail-closed g32 tests, then generate only the all-layer HQQ6 g32 cache/spec
  prerequisite. No live server, large cache generation, production HQQ, speed
  benchmark, fallback, decode/HCS, selected-expert retargeting, or runtime
  source change.

- Opened HQQ6 quality/acceptance gate
  `20260627_0728_nemotron_nano_hqq6_quality_acceptance`. Recorded `0640`
  first: Gemma remains `5619.6 / 92.43 / 155.69`, focused `expert_hqq` tests
  passed `92/0`, `./dev build` passed, HQQ GPU equals KRHQ/HQQ6 offline
  exactly at the targeted row/selected routed value, the worst attributed row
  is layer `51`, expert `60`, sorted row `1837`, col `1599`, BF16 `1072.0`
  versus HQQ/KRHQ `1104.0`, selected routed BF16-vs-HQQ delta is
  `9.10776762664318`, and HQQ was not enabled for production. Scope: use
  existing `0427`, `0524`, and `0640` artifacts first, no live server
  initially, summarize all three saved `1023` payloads for selected-token
  changes, top-k rank changes, selected/top-k logprob deltas, and whether the
  observed deltas concentrate in the layer `51` / expert `60` attribution
  path. Decide whether HQQ6 g64 is implementation-correct but quality-failing
  for Nano and whether to skip it for the next quantization variant.
  Forbidden: production HQQ, tolerance changes, speed benchmarks, INT4/HQQ4
  code, fallback, and decode/HCS changes. Closed data-only using existing
  artifacts. All three saved `1023` payloads changed top-k ordering: case 0
  kept token `1044` but moved `3` common top-k tokens with selected/top-k
  deltas `0.09120299999999992` / `0.13776600000000006`; case 1 kept token
  `1044` but moved `4` common top-k tokens with deltas
  `0.03495599999999999` / `0.42908800000000014`; case 2 changed token `1044`
  to `1321`, changed top-k membership, moved `8` common top-k tokens, and had
  selected/top-k deltas `0.20660000000000012` / `0.32201199999999996`
  (`0524` corrected max common top-k delta `0.49339699999999986`). Case 2 is
  proven concentrated at layer `51` / expert `60`; cases 0/1 are not per-layer
  attributed from existing artifacts. Decision: HQQ6 g64 is
  implementation-correct but quality-failing for Nano; skip Nano HQQ6 g64 for
  production and move to the next quantization variant. No live server rerun,
  source change, production HQQ, tolerance change, speed benchmark,
  INT4/HQQ4 code, fallback, or decode/HCS change.

- Opened targeted exact-row HQQ6 quantization attribution gate
  `20260627_0640_nemotron_nano_hqq6_exact_row_quantization_attribution`.
  Recorded `0629` first: Gemma remains `5619.6 / 92.43 / 155.69`, and exact
  offline attribution was blocked because the existing `0524` synced artifacts
  identify the worst deltas but lack exact HQQ-consumed gathered input rows,
  W13/activation rows, and selected-row routed contributor decomposition.
  Worst global expert-output coordinate: layer `51`, expert `60`, sorted row
  `1837`, column `1599`, BF16 `1072.0`, HQQ `1104.0`, abs delta `32.0`.
  Worst selected-row routed coordinate: layer `51`, column `1599`, BF16
  `485.826904296875`, HQQ `494.9346923828125`, abs delta
  `9.1077880859375`. HQQ was not enabled for production. Scope: implement only
  the smallest env-gated case-2 diagnostic capture for the exact gathered
  input row, W13 output, activation output, W2/output row, routing/contributor
  mapping, and matching KRHQ/HQQ6 dequant values; fail closed if absent; then
  run focused tests, `./dev build`, one live case-2 capture, and offline
  BF16-vs-KRHQ/HQQ6 comparison. Forbidden: broad dumps, tolerance changes,
  production HQQ, speed benchmarks, INT4/HQQ4, fallback, decode/HCS, and
  selected-expert retargeting. Closed after adding only the env-gated exact-row
  diagnostic capture plus offline Rust attribution analyzer. Focused
  `expert_hqq` tests passed `92/0`; `./dev build` passed. One live case-2 run
  returned HTTP 200, emitted `23` HQQ-consume rows, copied `330,882,048` bytes,
  and produced exactly one exact-row attribution stage. The analyzer captured
  `7` rows (`1` requested worst row plus `6` selected-row contributors). At
  the worst coordinate, BF16 output was `1072.0`, HQQ GPU output was `1104.0`,
  and offline KRHQ/HQQ6 dequant math was `1104.0`, so HQQ GPU vs KRHQ abs was
  `0.0`. At the selected routed coordinate, BF16 was
  `485.82693196833134`, HQQ GPU was `494.9346995949745`, and KRHQ/HQQ6 was
  `494.9346995949745`, so the selected routed abs delta from BF16 was
  `9.10776762664318` and HQQ GPU vs KRHQ abs was `0.0`. Decision: the
  divergence is attributable to HQQ6 dequantized-weight math versus captured
  BF16 expert output, not consume/scatter/layout/sync/output-conversion or a
  KRHQ cache-generation/GPU-dequant mismatch. No production HQQ enablement,
  tolerance change, fallback, decode/HCS, INT4/HQQ4, selected-expert
  retargeting, broad dump, or speed benchmark was added.

- Opened quantization-error attribution gate
  `20260627_0629_nemotron_nano_hqq6_quantization_error_attribution`. Recorded
  `0524` first: Gemma remains `5619.6 / 92.43 / 155.69`, focused
  `expert_hqq` tests passed `86/0`, `./dev build` passed, the corrected
  case-2 live proof returned HTTP 200, HQQ consume emitted `23` rows and
  copied `330,882,048` bytes, synced GPU scatter vs CPU replay max was
  `7.63e-6`, post-output conversion was exact, and the selected token changed
  from BF16 `1044` to HQQ `1321`. Scope: do not run a live server first; reuse
  the `0524` synced boundary artifacts to identify worst layers/rows/columns
  and active experts, then compare those exact rows against original BF16
  expert math and KRHQ/HQQ6 dequant math offline. Add only a focused offline
  analyzer if existing artifacts are insufficient. Forbidden: tolerance
  changes, production HQQ enablement, speed benchmarks, INT4/HQQ4, fallback,
  and decode/HCS changes. Closed data-only without source changes or a live
  rerun. The existing `0524` artifacts identify the worst global expert-output
  delta as layer `51`, expert `60`, sorted row `1837`, column `1599`: BF16
  `1072.0` versus HQQ `1104.0`, abs delta `32.0`; the worst selected-row
  routed delta is layer `51`, column `1599`: BF16 `485.826904296875` versus
  HQQ `494.9346923828125`, abs delta `9.1077880859375`. The exact offline
  BF16-vs-KRHQ recomputation cannot be performed from the existing artifacts
  because the `0524` capture exported summaries/top-N only, not the exact
  HQQ-consumed gathered input rows, W13/activation rows, or selected-row routed
  contributor decomposition. Decision: current evidence still excludes
  consume/scatter/sync/output-conversion defects and broad KRHQ/GPU layout
  defects, but exact inherent-quantization-vs-cache-generation attribution
  needs a targeted next capture. No tolerance change, production HQQ,
  benchmark, INT4/HQQ4, fallback, decode/HCS, or source change was added.

- Opened HQQ-consume BF16 mismatch diagnosis gate
  `20260627_0524_nemotron_nano_hqq_consume_boundary_diagnosis`. Recorded
  `0427` first: Gemma remains `5619.6 / 92.43 / 155.69`, focused
  `expert_hqq` tests passed `83/0`, `./dev build` passed, BF16 and HQQ-gated
  live runs each returned HTTP `200/200/200`, the HQQ consume path emitted
  `69` rows and copied `1,029,740,544` bytes, cases 0/1 preserved selected
  token `1044` while top-k/logprobs changed, case 2 changed from BF16 token
  `1044` to HQQ token `1321`, max selected-logprob delta was `0.2066`, max
  top-k-logprob delta was `0.429088`, and HQQ was not enabled for production.
  Scope: diagnose whether the divergence is expected HQQ6 quantization error
  or a consume/scatter/layout/sync bug by adding only the smallest consume
  boundary capture for one saved `1023` payload. Do not change comparator
  tolerances, production behavior, fallback behavior, decode/HCS, INT4/HQQ4,
  selected-expert targeting, or speed benchmark paths. Closed after adding a
  diagnostic-only consume-boundary capture and running one saved case-2 `1023`
  payload. The first live capture exposed a diagnostic read bug: post-scatter
  and post-output downloads needed an explicit stream sync before reading
  non-default-stream work. After the sync patch, focused `expert_hqq` tests
  passed `86/0` and `./dev build` passed. The corrected live proof returned
  HTTP 200, emitted `23` consume rows for the one payload, and copied
  `330,882,048` HQQ output bytes. Result: HQQ6 g64 expert output differs from
  BF16 as expected for quantization (`expert_out` max layer max-abs `32.0`,
  sum-abs across layers `10,178,743.488197472`; selected-row CPU routed delta
  max `9.1077880859375`, sum-abs `3,966.7455259748385`), while synced GPU
  scatter matches CPU replay within FP32 roundoff (max `7.62939453125e-06`,
  sum-abs `0.006416842260478006`) and final output conversion is exact
  (`0/0`). Case 2 still changes BF16 token `1044` (`,`) to HQQ token `1321`
  (` and`), matching the `0427` HQQ result; selected log-prob delta is
  `-0.20660000000000012`. Decision: divergence is HQQ6 quantization/local
  expert-output error, not consume DtoD copy, scatter layout, stream sync, or
  BF16 output conversion. No production default enablement, tolerance change,
  fallback, decode/HCS change, INT4/HQQ4 work, selected-expert retargeting, or
  speed benchmark was added.

- Opened guarded HQQ prefill output-consumption proof gate
  `20260627_0427_nemotron_nano_guarded_hqq_prefill_output_consumption_proof`.
  Recorded `0327` first: Gemma remains `5619.6 / 92.43 / 155.69`, focused
  `expert_hqq` tests passed `81/0`, `./dev build` passed, the full all-layer
  offline comparator completed with `status=ok`, `passes_contract=true`,
  `3` cases, `5,870` blocks, input/output exact, W13 within the existing
  tolerance, activation accepted only by the zero-vs-positive-BF16-subnormal
  FTZ rule, elapsed `30:15.09`, max RSS `81,839,016 KB`, and no production
  output was consumed. Scope: inspect the prefill accumulation path and
  identify/implement the smallest explicit diagnostic gate that can consume
  HQQ6 g64 expert output for prefill while preserving BF16 as the default and
  reference path. Required proof: focused tests, `./dev build`, and a minimal
  live correctness comparison against the same saved `1023` payloads or the
  smallest equivalent existing reference path. No speed benchmark, INT4/HQQ4,
  fallback, selected-expert retargeting, decode/HCS, or default behavior
  change. Closed fail-closed after implementing only the guarded diagnostic
  consume path: HQQ consume now requires both `KRASIS_EXPERT_HQQ_PREFILL_CONSUME`
  and the existing reference/debug request context, with optional
  `KRASIS_EXPERT_HQQ_PREFILL_CONSUME_LAYERS`; BF16 default behavior is
  preserved. Focused `expert_hqq` tests passed `83/0`; `./dev build` passed.
  Live BF16 and HQQ-gated runs both reached READY after long calibration
  (`343.37s/4.24s` BF16, `343.40s/4.27s` HQQ) and the same three `1023`
  payloads returned HTTP 200. The HQQ-gated run emitted all expected consume
  stages: `69` rows (`23` MoE layers x `3` requests), HQQ6 g64, total copied
  output bytes `1,029,740,544`. Final BF16-output comparison failed closed:
  cases 0/1 kept token `1044`, but top-k/logprobs changed; case 2 changed
  selected token from BF16 `1044` (`,`) to HQQ `1321` (` and`). Max diffs:
  selected logprob `0.2066`, top-k logprob `0.429088`, selected raw logit
  `0.27006053924560547`, top raw logit `0.5203342437744141`. No production
  default, decode/HCS, fallback, INT4/HQQ4, selected-expert retargeting, or
  speed path was added.

- Opened full all-layer offline comparator replay optimization gate
  `20260627_0327_nemotron_nano_full_all_layer_offline_comparator_replay_optimization`.
  Recorded `0304` first: Gemma remains `5619.6 / 92.43 / 155.69`, focused
  `expert_hqq` tests passed `81/0`, `./dev build` passed, the filtered
  comparator reported `status=ok` and `passes_contract=true` on the four
  prior failing blocks, all `9` activation mismatches are observed BF16
  `0x0000` versus positive BF16 subnormal references
  (`0x0001` x6, `0x0002` x1, `0x0005` x1, `0x000f` x1), full all-layer
  replay was not rerun in `0304`, and production output was not consumed.
  Scope: reuse the existing `2239` trace/cache, instrument comparator runtime
  and memory behavior, then implement only the smallest comparator-side
  improvement needed to make the full all-layer offline replay complete while
  preserving full-buffer comparison, the activation FTZ/subnormal rule, and
  fail-closed behavior. No live server, production HQQ wiring, fallback,
  decode/HCS, selected-expert retargeting, or speed benchmark. Closed after a
  comparator-only implementation: opt-in phase profiling, buffered trace JSON
  parsing without the extra raw trace string copy, and per-request parallel
  block comparison with the existing BF16 oracle and exact/full-buffer
  contracts. Focused `expert_hqq` tests passed `81/0`; `./dev build` passed.
  The existing `2239` trace/cache full replay completed with
  `status=ok`, `passes_contract=true`, cases `3`, blocks `5,870`, metric rows
  `23,576`, and mismatch-detail rows `9`. Input/output were exact; W13 stayed
  within the existing subnormal tolerance; activation passed only through the
  documented zero-vs-positive-BF16-subnormal rule with the same `9` mismatch
  rows. Full invocation elapsed `30:15.09` with max RSS `81,839,016 KB`;
  measured comparator phases were cache load `31.844s`, trace parse `97.685s`,
  and all-case CPU oracle comparison `1,438.780s`. No live server, production
  HQQ wiring/output consumption, fallback, decode/HCS, selected-expert
  retargeting, or speed benchmark was added.

- Opened diagnostic activation-subnormal comparator contract gate
  `20260627_0304_nemotron_nano_activation_subnormal_comparator_contract_update`.
  Recorded `0132` first: Gemma remains `5619.6 / 92.43 / 155.69`, focused
  `expert_hqq` tests passed `77/0`, `./dev build` passed, every activation
  mismatch is actual BF16 `0x0000` versus positive BF16 subnormal references
  (`0x0001` x6, `0x0002` x1, `0x0005` x1, `0x000f` x1), and production output
  was not consumed. Scope is diagnostic comparator only: activation
  exact-contract may pass only when the sole activation mismatch class is
  zero-vs-positive-BF16-subnormal FTZ/subnormal classification. No W13
  tolerance broadening, normal activation delta tolerance, runtime output
  change, live server, production HQQ wiring, fallback, decode/HCS,
  selected-expert retargeting, or speed benchmark. Closed after implementing
  only that scoped diagnostic comparator rule. Focused `expert_hqq` tests
  passed `81/0`; `./dev build` passed. The filtered offline comparator reused
  the existing `2239` trace/cache/failure rows and now reports `status=ok`,
  `passes_contract=true`, cases `3`, blocks `4`. Input/output remain exact;
  W13 remains under the existing tolerance; activation still records
  sum/max `2.571393892423753924e-39 / 1.377532442369868173e-39` with `9`
  mismatch details, all actual `0x0000` versus positive BF16 subnormal
  references. Full all-layer replay was not rerun because the prior `0132`
  full replay attempt was already measured as impractical (`>45m`,
  approximately `90GB` RSS, no artifacts). No runtime output change,
  production HQQ wiring, fallback, decode/HCS, selected-expert retargeting, or
  speed benchmark.

- Opened offline activation-subnormal diagnosis gate
  `20260627_0132_nemotron_nano_all_moe_layer_activation_subnormal_offline_diagnosis`.
  Recorded `2239` first: Gemma remains `5619.6 / 92.43 / 155.69`, focused
  `expert_hqq` tests passed `75/0`, `./dev build` passed, the same three
  `1023` payloads succeeded, all `23` MoE layers and `5,870` active blocks
  were compared, input/output were exact, W13 was subnormal-only within
  existing tolerance, activation failed only on `9` subnormal BF16 mismatches
  in layers `6` and `8`, comparator `status=ok` but `passes_contract=false`,
  and production output was not consumed. Scope is offline-only first: reuse
  the existing `2239` trace/cache/metrics, inspect failing rows, and add only
  the smallest diagnostic enhancement if needed to emit exact mismatch
  locations, BF16 bits, values, and flush-to-zero/subnormal classification. No
  comparator tolerance relaxation, production HQQ wiring, fallback, decode/HCS,
  production output consumption, live-server rerun first, or speed benchmark.
  Closed after adding only offline comparator diagnostics. Added exact
  mismatch-detail export and optional failure-row filtering to
  `./dev expert-hqq-trace-compare`; the aggregate comparator contract remains
  unchanged. Full 9.0G trace replay was stopped as impractical before writing
  artifacts, so the successful filtered run reused the existing `2239` failure
  TSV and compared only the four failing activation blocks. Result:
  `status=ok`, `passes_contract=false`, `4` blocks, `9` exact mismatch rows.
  Every mismatch is actual BF16 `0x0000` versus a positive BF16 subnormal
  reference (`0x0001` x6, `0x0002` x1, `0x0005` x1, `0x000f` x1), classified
  as `bf16_zero_vs_subnormal_flush_to_zero_candidate`; no normal activation
  mismatch was found. Focused `expert_hqq` tests passed `77/0` and `./dev
  build` passed. Next gate should justify a diagnostic activation subnormal
  contract/tolerance classification change before rerunning the all-layer
  comparator.

- Opened all-MoE-layer diagnostic HQQ runtime proof gate
  `20260626_2239_nemotron_nano_all_moe_layer_diagnostic_hqq_runtime_proof`.
  Recorded `2208` first: Gemma remains `5619.6 / 92.43 / 155.69`, focused
  `expert_hqq` tests passed `74/0`, `./dev build` passed, the first all-layer
  generator attempt failed closed on config hash, the corrected retry passed,
  Nano has `23` MoE layers, the all-layer cache/spec covers `5,888` tensors
  and cache bytes `25,703,475,264`, and `2208` did not run a live proof or
  consume production output. Scope is only to extend the existing
  branch-replay/debug-gated diagnostic export/comparator from layer-1
  all-active blocks to all MoE layers using the `2208` cache/spec. Before
  live proof, estimate trace size and disk/RAM headroom and stop with exact
  artifact requirements if impractical. No production HQQ wiring, fallback,
  selected-expert retargeting, decode/HCS, or speed benchmark.
  Closed fail-closed after implementation and live proof. Preflight estimated
  an all-layer trace around `11,960,121,969` bytes with sufficient disk/RAM
  headroom, so the proof proceeded. Extended the existing diagnostic export
  under the branch-replay/debug gate to follow
  `KRASIS_REFERENCE_BRANCH_REPLAY_FULL_LAYERS`, while keeping production output
  untouched. Focused `expert_hqq` tests passed `75/0`; `./dev build` passed.
  The live BF16 Nano run reached READY after long calibration (`343.64s`
  prefill, `4.36s` decode); the same three `1023` payloads returned HTTP 200
  and wrote a `9.0G` trace. Offline comparator covered all `23` MoE layers,
  `5,870` active routed expert blocks, and `23,480` block-stage comparisons.
  Result: input/output exact, W13 passed existing subnormal tolerance
  (`5.041435275203618233e-32` sum, `7.523163845262640051e-37` max), but the
  comparator failed closed because activation has zero tolerance and saw `9`
  subnormal BF16 mismatches (`2.571393892423753924e-39` sum,
  `1.377532442369868173e-39` max) on layers `6` and `8`. No production HQQ
  wiring, fallback, selected-expert retargeting, decode/HCS, production output
  consumption, or speed benchmark was added.

- Opened diagnostic HQQ prefill all-MoE-layer correctness gate
  `20260626_2208_nemotron_nano_diagnostic_hqq_prefill_all_moe_layers_correctness`.
  Recorded `2043` first: Gemma remains `5619.6 / 92.43 / 155.69`, focused
  `expert_hqq` tests passed `72/0`, `./dev build` passed, the same three
  `1023` payloads returned HTTP 200, layer-1 export/comparator covered `384`
  active expert blocks, input/activation/output were exact, W13 had only
  subnormal noise, and production output was not consumed. Scope is to inspect
  whether the existing KRHQ cache/spec generator can produce all-layer expert
  payloads. If yes, implement the smallest diagnostic-only all-layer
  export/comparator extension and run the same three `1023` payloads. If no,
  implement only the all-layer cache/spec generation prerequisite and stop with
  exact artifact requirements. No production HQQ wiring, decode/HCS, fallback,
  selected-expert retargeting, production output consumption, or speed
  benchmark.
  Closed as an all-layer cache/spec generation prerequisite. The existing
  generator was single-layer/layer-1 only, so runtime all-layer
  export/comparator proof was deferred. Added explicit `layers` manifest
  support with exact model MoE layer validation, complete W13/W2 source tensor
  validation, and per-layer diagnostic spec groups while preserving the
  single-layer `layer_idx` path. Focused `expert_hqq` tests passed `74/0`;
  `./dev build` passed. The first generator attempt failed closed on wrong
  manifest config hash; the corrected retry passed and generated the Nano
  all-MoE-layer HQQ6 g64 cache/spec: `23` MoE layers, `128` experts/layer,
  `5,888` tensor records, `25,702,957,056` payload bytes, and
  `25,703,475,264` cache bytes. No runtime proof, production HQQ wiring,
  production output consumption, fallback, selected-expert retargeting,
  decode/HCS, or speed benchmark.

- Opened diagnostic HQQ prefill all-active-block numeric export/comparator
  gate
  `20260626_2043_nemotron_nano_diagnostic_hqq_prefill_all_active_blocks_numeric_export_comparator`.
  Recorded `2003` first: Gemma remains `5619.6 / 92.43 / 155.69`, focused
  `expert_hqq` tests passed `70/0`, `./dev build` passed, the same three
  `1023` payloads returned HTTP 200, the current diagnostic proof covers
  layer-1 expert 0 rows `16/21/19`, input/activation/output are exact against
  the BF16-path oracle, W13 has only subnormal noise, and production output is
  not consumed. Scope is only to extend the existing branch-replay/debug-gated
  diagnostic launch/export and offline comparator to every non-empty routed
  expert block in layer 1, with per expert/block aggregate metrics. No
  fallback, decode/HCS, production HQQ path, selected-expert retargeting,
  production output consumption, or speed benchmark.
  Closed with implementation and live all-active-block numeric proof: extended
  the existing branch-replay/debug-gated diagnostic launch/export to all
  non-empty layer-1 runtime expert blocks and extended the offline Rust
  comparator to aggregate by request/layer/expert/block. Focused
  `expert_hqq` tests passed `72/0`; `./dev build` passed. Live BF16 Nano proof
  reached READY after long calibration (`343.55s` prefill, `4.25s` decode) and
  the same three `1023` payloads returned HTTP 200. Exported all `128` experts
  per case, `384` blocks total, rows `2694/2958/2676`, and full
  input/W13/activation/output buffers. Comparator passed for all `1536`
  block-stage rows: input/activation/output exact; W13 sum/max/L2
  `8.936433336278421240e-33 / 7.523163845262640051e-37 /
  3.690801512227770158e-35`, below tolerance. No production output
  consumption, fallback, selected-expert retargeting, decode/HCS, production
  HQQ path, or speed benchmark was added.

- Opened diagnostic HQQ prefill same-block numeric export/comparator
  implementation gate
  `20260626_2003_nemotron_nano_diagnostic_hqq_prefill_same_block_numeric_export_comparator`.
  Recorded `1952` first: Gemma remains `5619.6 / 92.43 / 155.69`; selected
  path is targeted full-buffer diagnostic export plus offline Rust comparator;
  and `1952` made no source/runtime changes. Scope is only to export the same
  expert-0 block inputs and HQQ diagnostic W13/activation/W2 full buffers under
  the existing branch-replay trace/debug gate, add a `./dev`-reachable Rust
  comparator reusing existing BF16/reference helper logic where possible, and
  keep production output untouched with no runtime semantic change. Required
  fail-closed tests: missing buffers, wrong layer/expert, shape mismatch,
  dtype/layout mismatch, and a valid small fixture. Required validation:
  `./dev test-kernels expert_hqq` and `./dev build`; if those pass, rerun the
  existing BF16 Nano branch-replay proof with the same three `1023` payloads
  and run the offline comparator. No fallback, selected-expert retargeting,
  decode/HCS, output consumption, HQQ production path, or speed benchmark.
  Closed with implementation and live numeric proof: added diagnostic-only
  full-buffer export for the same layer-1 expert-0 block's gathered input,
  HQQ W13, HQQ activation, and HQQ W2 temporary output under the existing
  branch-replay trace path. Added `./dev expert-hqq-trace-compare`, a Rust
  offline comparator using the explicit all-128 `1819` KRHQ spec/cache and
  BF16-path oracle. Focused `expert_hqq` tests passed `70/0`; `./dev build`
  passed. Live BF16 Nano proof reached READY after long calibration
  (`343.43s` prefill, `4.25s` decode), then the same three `1023` payloads
  returned HTTP 200. Exported rows were `16/21/19`. Comparator passed:
  input/activation/output deltas were exactly zero; W13 sum/max/L2 were
  `2.185213692464900340e-34 / 2.887307999207243691e-37 /
  5.641618242106247438e-36`, below tolerance. No production output
  consumption, runtime semantic change, fallback, selected-expert retargeting,
  decode/HCS, HQQ production path, or speed benchmark was added.

- Opened diagnostic HQQ prefill same-block numeric oracle design gate
  `20260626_1952_nemotron_nano_diagnostic_hqq_prefill_same_block_numeric_oracle_design`.
  Recorded `1943` first: Gemma remains `5619.6 / 92.43 / 155.69`; the live
  diagnostic launch used layer 1 expert 0 with rows `16/21/19`; there is no
  exact same-block reference surface for that launch; and no source/runtime
  changes were made in `1943`. Scope is design-only: inspect the existing BF16
  expert path, existing HQQ CPU/reference helpers, and diagnostic launch export
  surface to define the smallest numeric comparison path. Do not implement,
  hardcode-retarget to selected experts, add fallback, touch decode/HCS,
  consume production output, or run speed benchmarks.
  Closed with design proposal only: the smallest path is to add targeted
  diagnostic full-buffer export for the live expert-0 block's gathered input,
  HQQ W13, HQQ activation, and HQQ W2 output, then run an offline Rust
  comparator using the existing `cfg(test)` BF16-path oracle and the
  `1347/1412` tolerances. The BF16 production path can provide final expert
  output, but current full-vector BF16 traces are selected-expert oriented and
  ungated Nano overwrites W13 preactivation in-place. No implementation, source
  changes, runtime rerun, production output consumption, fallback, decode/HCS,
  or speed benchmark was added.

- Opened diagnostic HQQ prefill single-block numeric sanity gate
  `20260626_1943_nemotron_nano_diagnostic_hqq_prefill_single_block_numeric_sanity`.
  Recorded `1908` first: Gemma remains `5619.6 / 92.43 / 155.69`; diagnostic
  GPU launch stages emitted `3/3`; the launch used layer 1 expert 0, HQQ6 g64,
  temporary diagnostic buffers only, no production output consumption, no
  runtime output path, focused built-command `expert_hqq` tests passed `65/0`,
  and `./dev build` passed. Scope is only to prove the diagnostic output is
  numerically sane against an existing validated CPU/reference or
  BF16-equivalent path if one already exists. If no exact same-block reference
  exists without broad converter/reference logic, stop and record the missing
  call-site/data requirements. No production output path, runtime semantic
  change, decode/HCS, fallback, or speed benchmark.
  Closed as blocked before numeric comparison: the live diagnostic launch is
  expert 0 for row counts `16/21/19`, while existing full-vector traces only
  export selected BF16 experts (`26/42/72/88/89/112` for cases 0/1 and
  `26/42/47/72/88/89` for case 2). No expert-0 routed-input/W13/branch-output
  full vectors exist in the current artifacts, and the diagnostic launch emits
  hashes/counts/eight samples rather than full W13/activation/W2 vectors.
  Existing BF16 oracle code needs explicit host input rows and is not a live
  same-block runtime comparator. No source changes, runtime rerun, production
  output path, fallback, decode/HCS, or speed benchmark was added.

- Opened diagnostic HQQ prefill GPU launch proof gate
  `20260626_1908_nemotron_nano_diagnostic_hqq_prefill_gpu_launch_proof`.
  Recorded `1846` first: Gemma remains `5619.6 / 92.43 / 155.69`; the
  all-128 `1819` diagnostic spec registered successfully in the existing BF16
  Nano branch-replay path; the same three `1023` payloads returned HTTP 200
  and emitted `3/3` hook success reports; checked metadata covered `128`
  experts, `256` W13/W2 records, `1,117,519,872` payload bytes, rows
  `2694/2958/2676`, and strides `2688/1856/1856/2688`; no source changes,
  fallback, runtime hook change, output consumption/comparison, HQQ GPU launch,
  decode/HCS, or speed benchmark was added. Scope is only to inspect the
  existing hook/kernel adapter path and define the minimum single-layer,
  single-block diagnostic GPU launch using the all-128 `1819` cache/spec. Stop
  and record the exact blocker if a launch requires broader architecture or
  output-path changes.
  Closed with live diagnostic launch proof: added a request-scoped diagnostic
  stage under the existing branch-replay trace path. It launches one runtime
  expert block using the all-128 KRHQ cache/spec into temporary buffers only
  and emits BF16 hashes/counts. `./dev build` passed and focused
  `./dev test-kernels expert_hqq` passed `65/0`. The live BF16 Nano proof
  reached READY after long calibration (`343.48s` prefill, `4.31s` decode)
  and heatmap ranking; the same three `1023` payloads returned HTTP 200.
  Metadata contract stages emitted `3/3`; diagnostic GPU launch stages emitted
  `3/3` with `available=true`, `gpu_kernels_launched=true`, layer 1 expert 0,
  row counts `16/21/19`, HQQ6 g64, and strides `2688/1856/1856/2688`.
  Output was not consumed by production, no output comparison was added, the
  test-only adapter was not called, and no fallback, decode/HCS, runtime
  output path, or speed benchmark was added.

- Opened all-128 KRHQ cache runtime diagnostic proof gate
  `20260626_1846_nemotron_nano_all128_krhq_runtime_diagnostic_proof`.
  Recorded `1819` first: Gemma remains `5619.6 / 92.43 / 155.69`, focused
  built-command `expert_hqq` tests passed `65/0`, `./dev build` passed, Rust
  `./dev expert-hqq-cache-generate` passed, the full Nano layer-1 all-128 HQQ6
  g64 KRHQ cache/spec was generated with `256` W13/W2 records and
  `1,117,542,464` cache bytes, and no live proof was rerun. Scope is only to
  run the existing BF16 Nano branch-replay path with the new diagnostic spec,
  wait to READY, send the same three `1023` payloads, and compare emitted
  metadata against `1537`/`1432` plus the `1819` manifest/spec. No source
  changes, fallback, runtime hook changes, output consumption, HQQ GPU launch,
  decode/HCS, or speed benchmark.
  Closed with live proof: READY was reached after long calibration
  (`343.53s` prefill, `4.21s` decode) and heatmap ranking. The same three
  `1023` payloads returned HTTP 200 and emitted `3/3` metadata hook stages.
  Registration succeeded: all hook reports were `available=true` with
  `128` checked experts, `256` W13/W2 records, `1,117,519,872` payload bytes,
  HQQ6 g64, runtime rows `2694/2958/2676` matching `prompt_len * 6`, and
  strides `2688/1856/1856/2688`. This matches the `1819` all-128 manifest/spec
  and is the expected broader live runtime contract compared with the earlier
  selected-block `1537`/`1432` artifacts. No source changes, fallback, runtime
  hook change, output consumption/comparison, HQQ GPU launch, decode/HCS, or
  speed benchmark was added.

- Opened all-128 KRHQ cache generator implementation gate
  `20260626_1819_nemotron_nano_layer1_all_expert_krhq_cache_generator_implementation`.
  Recorded `1811` first: Gemma remains `5619.6 / 92.43 / 155.69`; all
  `256/256` required Nano layer-1 W13/W2 source tensors are present; and the
  blocker is no built/gated all-128 generator entrypoint. Scope is only a
  built `./dev`-reachable generator that validates an explicit manifest and
  feeds the existing Rust `write_expert_hqq_cache_from_safetensors` / readback
  path. No Python converter, runtime hook changes, HQQ GPU launch, fallback,
  output path work, decode/HCS, speed benchmark, or live proof rerun in this
  implementation gate.
  Closed with implementation: added strict Rust manifest validation, a Rust
  generator binary, and `./dev expert-hqq-cache-generate <manifest.json>`.
  Focused built-command `expert_hqq` tests pass `65/0`; `./dev build` passed.
  Generated the full Nano layer-1 all-128 HQQ6 g64 KRHQ cache and diagnostic
  spec as benchmark artifacts (`256` W13/W2 records,
  `1,117,542,464` cache bytes). The first manifest failed closed on config
  hash mismatch before cache write; the corrected manifest matched the current
  in-repo hash contract and readback/spec validation passed. No Python cache
  converter, runtime hook change, HQQ GPU launch, fallback, output path work,
  decode/HCS, speed benchmark, or live proof rerun was added.

- Opened layer-1 all-expert KRHQ descriptor coverage gate
  `20260626_1811_nemotron_nano_layer1_all_expert_krhq_cache_generation`.
  Recorded `1749` first: Gemma remains `5619.6 / 92.43 / 155.69`; the
  explicit `0836` diagnostic spec registered successfully; hook stages emitted
  `3/3`; and the hook failed closed on
  `missing required expert-HQQ descriptor for layer=1 expert=0 role=w13`
  because live runtime validates all 128 layer-1 experts while `0836` covers
  only seven selected experts. Scope is to inspect existing KRHQ
  generation/readback tooling and determine the smallest data-only route to a
  full layer-1 all-128 Nano HQQ6 g64 diagnostic cache/spec artifact. No source
  changes, runtime hook semantic changes, fallback, output paths, GPU HQQ
  launch, decode/HCS, or speed benchmark.
  Closed as blocked before cache generation: all `256/256` all-expert layer-1
  W13/W2 safetensors source tensors exist, but the current validated KRHQ
  builder is only reachable through Rust API/test plumbing and the real `0836`
  proof derives seven experts from prior selected-expert artifacts. There is
  no existing built/gated entrypoint that accepts an all-128 explicit spec
  manifest and writes/readback-validates a KRHQ cache. No usable cache/spec,
  source change, runtime hook change, fallback, output path, GPU HQQ launch,
  decode/HCS, or speed work was added.

- Opened explicit diagnostic KRHQ cache spec runtime proof gate
  `20260626_1749_nemotron_nano_explicit_diagnostic_krhq_cache_spec_runtime_proof`.
  Recorded `1730` first: Gemma remains `5619.6 / 92.43 / 155.69`;
  explicit diagnostic spec input is implemented, focused built-command
  `expert_hqq` tests pass `62/0`, `./dev build` passed, and no live proof was
  rerun. Scope is to generate an `0836` benchmark artifact spec only from
  existing validated metadata if possible, then run the existing BF16 Nano
  branch-replay path with the explicit spec, wait through calibration, send the
  same three `1023` payloads, and record whether registration succeeds and the
  hook succeeds or fails closed. No source changes, auto-discovery, fallback,
  output consumption, HQQ GPU launch, decode/HCS, or speed benchmark.
  Closed with live proof: generated the `0836` diagnostic spec from existing
  descriptor metadata only, runtime registration succeeded, and all three
  `1023` payloads emitted the metadata hook. The hook failed closed on
  `missing required expert-HQQ descriptor for layer=1 expert=0 role=w13`,
  because live layer-1 runtime validation covers all `128` experts while the
  `0836` KRHQ cache contains only the seven selected experts. No source
  changes, fallback, output consumption, HQQ GPU launch, decode/HCS, or speed
  benchmark was added.

- Opened explicit diagnostic KRHQ cache spec registration gate
  `20260626_1730_nemotron_nano_explicit_diagnostic_krhq_cache_spec_registration`.
  Recorded `1722` first: the selected contract is one explicit
  diagnostic-only spec path surface,
  `CFG_EXPERT_HQQ_DIAGNOSTIC_CACHE_SPEC` /
  `--expert-hqq-diagnostic-cache-spec`, pointing to JSON that contains both
  the KRHQ cache path and non-empty W13/W2 descriptor requirements.
  Registration must occur after `WeightStore::load_from_hf/load_from_gguf`
  and before `Arc` wrapping in `KrasisEngine.load`, using FNV-1a over raw
  model `config.json` bytes for the KRHQ hash, while reusing existing
  `GpuDecodeGraph`/`PrefillEngine` propagation. Scope is only strict spec
  validation, diagnostic-only registration, focused fail-closed tests, and
  `./dev build`; no auto-discovery, hardcoded `0836` path, fallback, output
  consumption, GPU HQQ launch, decode/HCS, speed work, or live proof rerun.
  Closed with implementation: added strict JSON diagnostic spec parsing,
  explicit cache/spec validation, `WeightStore` registration before `Arc`
  wrapping in `KrasisEngine.load`, and Python config/CLI/model-load pass-through
  for `CFG_EXPERT_HQQ_DIAGNOSTIC_CACHE_SPEC` /
  `--expert-hqq-diagnostic-cache-spec`. Focused built-command tests now pass
  `62/0`; `./dev build` passes. The `./dev test-kernels expert_hqq` wrapper
  now maps to the actual `expert_hqq` Rust filter and includes cargo on PATH,
  so the built command no longer reports zero focused tests. No live
  branch-replay proof, auto-discovery, hardcoded `0836` path, fallback, output
  consumption, GPU HQQ launch, decode/HCS, or speed work was added.

- Opened KRHQ cache registration contract design gate
  `20260626_1722_nemotron_nano_krhq_cache_registration_contract_design`.
  Recorded `1715` first: Gemma remains `5619.6 / 92.43 / 155.69`; the live
  branch-replay path reached READY in `1655`, emitted `3/3` metadata-only hook
  stages, and failed closed with `expert-HQQ runtime prefill diagnostic cache
  is not registered`; the `1715` inspection made no source changes and stopped
  because live registration of the `0836` cache requires an explicit
  cache-path contract rather than hardcoding or auto-discovery. Scope is
  design-only: map the smallest explicit way to provide a KRHQ cache path to
  `WeightStore`, covering config/API surface, validation rules, failure modes,
  and affected call sites. No source implementation, runtime rerun, config
  edit, fallback, output consumption, HQQ GPU launch, decode/HCS, or speed
  work. Closed design-only with proposal: add one explicit diagnostic spec
  path surface, `CFG_EXPERT_HQQ_DIAGNOSTIC_CACHE_SPEC` /
  `--expert-hqq-diagnostic-cache-spec`. The spec supplies both the KRHQ cache
  path and non-empty W13/W2 descriptor requirements; Rust should register it
  after `WeightStore::load_from_hf/load_from_gguf` and before the store is
  wrapped in `Arc`, using FNV-1a over raw model `config.json` bytes for the
  KRHQ hash contract. Existing `GpuDecodeGraph`/`PrefillEngine` propagation is
  reused. A path-only design was rejected because it would weaken the
  descriptor requirement guard; hardcoding and auto-discovery remain rejected.
  No source changes or runtime rerun were made.

- Opened runtime diagnostic KRHQ cache registration gate
  `20260626_1715_nemotron_nano_runtime_diagnostic_krhq_cache_registration`.
  Recorded `1655` first: Gemma remains `5619.6 / 92.43 / 155.69`; the
  existing branch-replay path reached READY after long calibration; the three
  exact `1023` reference-test payloads all returned HTTP 200; all three
  emitted the metadata-only hook stage; the hook failed closed with
  `expert-HQQ runtime prefill diagnostic cache is not registered`; and no
  GPU HQQ launch, test-only adapter call, output use/comparison, production
  behavior change, or source change was added. Scope is only to inspect the
  explicit model/weight-load/config path needed to register the `0836` KRHQ
  cache as a diagnostic-only handle in `WeightStore`/`PrefillEngine`, then
  rerun the same payloads if that path stays narrow. No auto-discovery,
  fallback, output consumption, decode/HCS, speed work, GPU HQQ launch, or
  broad loading/config changes. Closed blocked before implementation:
  `WeightStore::register_expert_hqq_cache_from_path` exists and registered
  KRHQ metadata can already propagate into `PrefillEngine`, but the normal
  server/model load path has no existing explicit expert-KRHQ diagnostic cache
  path or descriptor-registration input. Adding one would be a new
  user-facing config/env/CLI trigger or broader loader change, while hardcoding
  or auto-discovering the `0836` benchmark artifact would violate the gate. No
  source changes, runtime rerun, GPU HQQ launch, output use, fallback,
  decode/HCS, or speed work was added.

- Opened runtime diagnostic hook retry gate
  `20260626_1655_nemotron_nano_runtime_diagnostic_contract_hook_retry`.
  Recorded `1636` first: Gemma remains `5619.6 / 92.43 / 155.69`; long
  prefill calibration completed in `343.38s`; decode completed in `4.29s`;
  Rust timing attributed MoE `227,583.1ms` and attention `115,589.6ms`; no
  source changes were made; and READY requires waiting through the long
  calibration rather than treating the probe line as a stuck kernel/sync path.
  Scope is only to rerun the existing branch-replay trace path with built
  commands and diagnostic logging, send the existing `1023` reference-test
  payloads once READY appears, and capture whether the metadata-only runtime
  hook emits a contract report or fail-closed error. No source changes,
  triggers, config knobs, fallbacks, output consumption, HQQ GPU launch,
  decode/HCS, or speed benchmarking.
  Closed with live proof: the existing branch-replay path reached READY after
  long calibration, and the three exact `1023` reference-test payloads all
  returned HTTP 200. The metadata-only hook fired for all three requests and
  failed closed with `expert-HQQ runtime prefill diagnostic cache is not
  registered`. It reported no GPU HQQ launch, no test-only adapter call, no
  output comparison, and no production behavior change. Because the live
  runtime config lacks a registered KRHQ diagnostic cache handle, selected
  KRHQ block/stride/buffer and role/nbits/group/layout comparison against
  `1537`/`1432` stopped at the absent-cache boundary. No source changes or
  runtime behavior changes were made.

- Opened long-calibration blocker diagnosis gate
  `20260626_1636_nemotron_nano_long_calibration_blocker_diagnosis`.
  Recorded `1620` first: Gemma remains `5619.6 / 92.43 / 155.69`; the real
  runtime diagnostic-hook proof sent no request, fired no hook, and made no
  source changes; the exact blocker line was `Long calibration: probing
  39,920 prompt tokens + 32 decode tokens`. Scope is only to inspect
  calibration/probe logging and runtime startup, then gather clear timing/data
  to determine whether long calibration is slow progress, missing progress
  logging, or stuck in a specific kernel/sync path. No calibration bypass,
  reduced probe, new trigger/config, proof-path change, runtime output change,
  HQQ dispatch, decode/HCS, fallback, or speed work.
  Closed with data: the same BF16 Nano branch-replay startup path was rerun
  through `./dev run` in tmux with only existing diagnostics enabled. The
  39,920-token long calibration completed rather than hanging: prefill
  `343.38s` (`116.3 tok/s`), decode `4.29s` (`7.5 tok/s`), then VRAM
  calibration completed and heatmap building began. Existing Rust timing
  attributed the long prefill to MoE `227,583.1ms` (`66.3%`) and attention
  `115,589.6ms` (`33.7%`). Diagnosis: genuinely slow long calibration plus
  missing in-flight progress logging inside `rust_prefill_tokens`, not a
  confirmed stuck kernel/sync path. No source changes or runtime behavior
  changes were made.

- Opened real runtime execution proof gate
  `20260626_1620_nemotron_nano_runtime_diagnostic_contract_hook_real_execution_proof`.
  Recorded `1602` first: the metadata-only runtime diagnostic hook exists
  under the existing `KRASIS_REFERENCE_BRANCH_REPLAY_FULL` trace path and adds
  no GPU HQQ launch, output comparison, runtime output consumption,
  trigger/config, auto-selection, fallback, decode/HCS, or speed path. Scope
  is only to run the existing branch-replay trace path with built commands and
  prove the hook writes a contract report or fail-closed error on a real Nano
  request.
  Closed blocked before request: the existing BF16 Nano branch-replay trace
  startup path reached short calibration but did not reach `KRASIS SERVER
  READY`; it remained on long calibration (`39,920` prompt tokens + `32`
  decode tokens). No request was sent, so no runtime contract report/error was
  emitted and no `1537`/`1432` metadata comparison was possible. No bypass,
  new trigger/config, GPU HQQ launch, fallback, decode/HCS, output path, or
  speed benchmark was added.

- Opened request-scoped runtime diagnostic contract hook gate
  `20260626_1602_nemotron_nano_runtime_diagnostic_contract_hook`. Recorded
  `1537` first: production-safe CPU-only KRHQ runtime prefill diagnostic
  contract validation exists; focused expert-HQQ tests passed `57/0`; gated
  real Nano no-GPU proof passed `1/0`; and no config knob, env runtime
  trigger, prefill hook, GPU execution, output comparison, fallback,
  decode/HCS, protected-config edit, or speed work was added. Scope: inspect
  the actual selected-expert runtime point in `gpu_prefill.rs` and wire only
  the production-safe CPU contract validator against real runtime-shaped
  block/stride/index metadata if that can be triggered narrowly without new
  user-facing config/env or broad prefill plumbing. No GPU kernels, test-only
  adapter calls, output comparisons, auto-selection, fallback, decode/HCS, or
  speed work.
  Closed with a metadata-only runtime contract hook under the existing
  `KRASIS_REFERENCE_BRANCH_REPLAY_FULL` selected-sequential trace path. Added
  a `PrefillEngine` CPU wrapper around the `1537` validator and a runtime
  metadata stage that validates compact selected-expert blocks plus logical
  strides/buffer lengths. The hook writes only a contract report or
  fail-closed error; it does not launch GPU kernels, call the test-only
  adapter, execute the BF16 oracle, compare outputs, consume model output,
  add a new config/env trigger, auto-select HQQ, fall back to Marlin, touch
  decode/HCS, or do speed work. `./dev build` and lightweight validation
  passed; `./dev test-kernels expert_hqq` was checked and still ran `0` tests
  due the existing filter mapping.

- Opened production-safe BF16-path oracle/runtime-shaped diagnostic API
  surface gate
  `20260626_1537_nemotron_nano_bf16_oracle_runtime_diagnostic_api_surface`.
  Recorded `1508` first: `PrefillEngine` carries an optional explicit `KRHQ`
  diagnostic cache handle and exposes the availability validator; focused
  expert-HQQ tests passed `53/0`; gated real Nano payload proof passed `1/0`;
  seven layer-1 selected experts and 14 W13/W2 HQQ6 g64 records were found;
  and no GPU execution, output comparison, runtime prefill consumer, config
  knob, auto-selection, fallback, decode/HCS, protected-config edit, or speed
  work was added. Scope: inspect the `cfg(test)` adapter/oracle and extract
  only non-test CPU-only runtime diagnostic contract pieces for descriptor
  validation, block/stride/index validation, and BF16-path oracle metadata.
  GPU launch harness and proofs remain `cfg(test)`. No config knobs, env
  runtime triggers, prefill hooks, GPU execution, decode/HCS, fallback, or
  speed work.
  Closed with a production-safe CPU-only diagnostic contract surface in
  `src/weights/expert_hqq.rs`. The new contract validates registered `KRHQ`
  descriptor availability, W13/W2 role pairing, `nbits`/group/layout metadata,
  sorted absolute runtime blocks, row ranges, input/W13/activation/output
  strides, buffer lengths, and BF16-path oracle metadata without dequantizing,
  launching kernels, comparing outputs, or feeding model output. Focused
  expert-HQQ tests passed `57/0`; the gated real Nano no-GPU proof passed
  `1/0` using the `0836` KRHQ cache and `1023` full routed/branch capture.
  Real proof coverage: `3` prompt cases, `18` plan entries, `385` claimed
  rows inside `6,984` runtime sorted rows, `6,599` padding rows, HQQ6 g64
  row-major uint6 layout. `./dev build`, artifact assertions,
  syntax/py-compile, protected-config status, source-scope audit, scatter
  guard, diff checks, and cleanup passed. GPU launch harness and proofs remain
  `cfg(test)`; no config knob, env runtime trigger, prefill hook, GPU
  execution, output comparison, decode/HCS, fallback, protected-config edit,
  broad architecture change, or speed work was added.

- Opened production-safe KRHQ metadata/cache availability gate
  `20260626_1508_nemotron_nano_runtime_krhq_metadata_cache_availability`.
  Recorded `1458` first: runtime shadow validation stopped at the correct
  boundary with no hook added, focused expert-HQQ tests passed `47/0`, Gemma
  remains `5619.6 / 92.43 / 155.69`, and the blocker is exact:
  `PrefillEngine` has no registered `KRHQ` cache handle while the validated
  runtime-shaped GPU adapter, BF16-path oracle, and GPU harness remain
  `cfg(test)` / `has_prefill_kernels` only. Scope: inspect model/weight
  loading and `PrefillEngine` construction, then add only a disabled-by-default
  diagnostic registry/check if isolated enough to prove runtime can locate
  correct `KRHQ` W13/W2 payloads for Nano layer-1 experts. No GPU kernels,
  output comparison, config knob, auto-selection, fallback, decode/HCS,
  runtime prefill output consumption, protected-config edit, or speed work.
  Closed with production-safe diagnostic metadata plumbing only. `WeightStore`
  explicit `KRHQ` registration metadata is propagated into `GpuDecodeGraph`
  and `PrefillEngine` as an optional diagnostic cache handle, and
  `PrefillEngine::validate_expert_hqq_runtime_diagnostic_availability`
  verifies W13/W2 payload availability without launching kernels or consuming
  outputs. The validator fails closed on absent cache, wrong model shape,
  wrong layer/expert, duplicate requirements, nbits/group/layout mismatch,
  shape/payload mismatch, and missing W13/W2 role pairs. Validation passed:
  `./dev build`, focused expert-HQQ tests `53/0`, the gated real Nano payload
  availability proof `1/0`, artifact assertions, syntax/py-compile,
  protected-config status, source-scope audit, scatter guard, diff checks, and
  cleanup. The real proof validated 7 selected layer-1 experts
  (`26,42,47,72,88,89,112`) and 14 W13/W2 HQQ6 g64 payload records from the
  `0836` KRHQ cache. No GPU execution, output comparison, config knob,
  auto-selection, fallback, decode/HCS, runtime prefill output
  consumption, protected-config edit, broad architecture change, or speed work
  was added.

- Opened request-scoped runtime shadow expert-HQQ GPU prefill validation gate
  `20260626_1458_nemotron_nano_runtime_shadow_expert_hqq_gpu_prefill_validation`.
  Recorded `1432` first: runtime-shaped GPU prefill buffer contract validation
  passed with focused expert-HQQ tests `47/0`, synthetic and real gated proofs
  passing, `385` claimed rows inside `6,984` runtime sorted rows, zero GPU
  final/W2 and activation delta vs the BF16-path oracle, W13 below tolerance,
  runtime padding untouched, and Gemma speeds
  `5619.6 / 92.43 / 155.69`. Scope: inspect `gpu_prefill.rs`
  selected-expert execution points and add only a request-scoped/off-by-default
  shadow diagnostic hook if it stays narrow. The hook must never feed HQQ
  output into model output. No config knob, auto-selection, decode/HCS, Marlin
  fallback, route-around, protected-config edit, broad architecture change, or
  speed benchmarking.
  Closed at a stop boundary with no runtime source hook added. Runtime
  selected-expert buffers and BF16 trace hooks exist, but `PrefillEngine` has
  no registered `KRHQ` cache handle and the validated runtime-shaped GPU
  adapter, BF16-path oracle, and GPU harness remain test-only. Adding the
  shadow hook would require promoting those APIs plus runtime cache
  registration/propagation, which is broader than this gate. Validation
  passed: `./dev build`, focused expert-HQQ tests `47/0`, artifact assertions,
  syntax/py-compile, protected-config status, source-scope audit, scatter
  guard, diff checks, and cleanup. No runtime prefill consumer, config knob,
  auto-selection, decode/HCS, Marlin fallback, route-around, guard bypass,
  protected-config edit, broad architecture change, or speed benchmark was
  added.

- Opened runtime-shaped GPU prefill buffer contract validation gate
  `20260626_1432_nemotron_nano_runtime_shaped_gpu_prefill_buffer_contract_validation`.
  Recorded `1412` first: full-block real Nano GPU replay passed with focused
  expert-HQQ tests `44/0`, gated proof `1/0`, `385/385` rows, zero GPU
  final/W2 delta vs the BF16-path oracle, W13 below the `1e-30` tolerance, and
  no runtime/config/decode/HCS/fallback/speed work. Scope remains
  test-only/offline: inspect runtime selected-expert buffer conventions and
  validate a runtime-shaped adapter over the existing GPU prototype without
  adding a runtime prefill consumer.
  Closed with only `cfg(test)` / env-gated code in `src/weights/expert_hqq.rs`.
  Focused expert-HQQ tests passed (`47 passed, 0 failed`), the synthetic
  runtime-shaped GPU proof passed (`1 passed, 0 failed`), the real Nano
  runtime-shaped proof passed (`1 passed, 0 failed`), and `./dev build`
  passed. The real proof covered `385` claimed rows over `6,984` absolute
  runtime sorted rows with `6,599` padding rows and `18` plan entries. GPU
  final/W2 and activation vs BF16-path oracle were exact zero; W13 sum/max
  was `4.271920958330893e-35 / 4.547693769743725e-37`, below `1e-30`; runtime
  W13/activation/output padding remained untouched. CPU f32/reference and
  captured-BF16 branch metrics remain diagnostics only. No runtime prefill
  consumer, config knob, auto-selection, decode/HCS, Marlin fallback,
  route-around, guard bypass, protected-config edit, broad architecture
  change, or speed benchmark was added.

- Opened full-block real Nano GPU prototype replay validation gate
  `20260626_1412_nemotron_nano_full_block_gpu_prototype_replay_validation`.
  Recorded `1347` first: BF16-path oracle is now the formal GPU prototype
  correctness contract, focused expert-HQQ tests passed (`43 passed, 0
  failed`), the gated real proof passed (`1 passed, 0 failed`), GPU final vs
  BF16-path oracle had zero delta, W13 differed only by subnormal noise
  (`8.46355932592047e-37` sum), and CPU f32 diagnostic W2 sum was
  `9.390317355753723`. Gemma speeds remain
  `5619.6 / 92.43 / 155.69`. Scope: offline/test-only validation using the
  existing `0836` real `KRHQ` cache and `1023` full routed-input/branch-output
  capture, expanded from `18` final selected slots to all `385` captured
  selected-expert rows, with GPU intermediates/final output checked against
  the BF16-path oracle using the `1347` tolerances. No runtime prefill
  consumer, config knob, auto-selection, decode/HCS, Marlin fallback,
  route-around, guard bypass, protected-config edit, broad architecture
  change, or speed benchmarking.
  Closed with only `cfg(test)` / env-gated code in `src/weights/expert_hqq.rs`.
  Focused expert-HQQ tests passed (`44 passed, 0 failed`), the gated
  full-block proof passed (`1 passed, 0 failed`), and `./dev build` passed.
  The full-block proof covered `385/385` routed and branch rows across `18`
  contiguous expert blocks and `1,034,880` output values. GPU final output vs
  BF16-path oracle remained exact (`0` sum abs); W13 subnormal sum/max was
  `4.271920958330893e-35 / 4.547693769743725e-37`, below the `1e-30`
  tolerance; activation and W2 deltas were zero, with first GPU oracle
  contract violation stage `none`. CPU f32/reference and captured-BF16 branch
  metrics were preserved as diagnostics. No runtime prefill consumer, config
  knob, auto-selection, decode/HCS, Marlin fallback, route-around, guard
  bypass, protected-config edit, broad architecture change, or speed benchmark
  was added.

- Opened GPU BF16-path oracle correctness contract gate
  `20260626_1347_nemotron_nano_gpu_bf16_oracle_correctness_contract`.
  Recorded `1319` first: test-only GPU BF16-path diagnostics passed, focused
  expert-HQQ tests passed (`43 passed, 0 failed`), the gated real alignment
  proof passed (`1 passed, 0 failed`), GPU final output matched the BF16-path
  oracle exactly (`0` sum abs), GPU/oracle W13 differed only by subnormal
  noise (`8.46355932592047e-37` sum), and CPU f32 reference vs BF16-path
  oracle W2 sum was `9.390317355753723`. Gemma speeds remain
  `5619.6 / 92.43 / 155.69`. Current Nemotron token results carried: BF16
  Moby/War/Les all `1044 1044 1044`; plain amax INT4 Moby
  `1321 1272 78526`, War `1044 1108 1078`, Les `1044 1262 1384`;
  search-RMSE INT4 Moby `1321 13540 1314`, War `1044 1083 3843`, Les
  `1044 1262 1384`. Scope: formalize the test-only GPU prototype correctness
  contract against the BF16-path oracle, keep f32 CPU/reference as diagnostic
  context, and preserve captured BF16 branch metrics. No runtime prefill
  consumer, config knob, auto-selection, decode/HCS, Marlin fallback,
  route-around, guard bypass, protected-config edit, broad architecture
  change, or speed benchmarking.
  Closed with only `cfg(test)` / env-gated validation changes in
  `src/weights/expert_hqq.rs`. The gated real proof now formalizes
  BF16-path oracle agreement as the correctness contract: W13 sum/max
  tolerance `1e-30`, activation/W2 exact-zero tolerances, captured-BF16 branch
  metric delta tolerance `1e-9`, and f32 CPU/reference output retained only as
  diagnostic context. Focused expert-HQQ tests passed (`43 passed, 0 failed`);
  the gated real oracle-contract proof passed (`1 passed, 0 failed`); `./dev
  build` passed. Real result: selected slots `18/18`, GPU final vs BF16-path
  oracle sum abs `0`, W13 subnormal sum/max
  `8.46355932592047e-37 / 4.70197740328915e-38`, activation/W2 deltas `0`,
  first GPU oracle contract violation stage `none`, and CPU f32 diagnostic W2
  sum `9.390317355753722950e0`. Captured BF16 branch metrics remain
  sum/max/L2 `64.231438333168626 / 0.0091552734375 /
  0.380071850011764`, with deltas vs `1259` below `1e-9`. No runtime prefill
  consumer, config knob, auto-selection, decode/HCS, Marlin fallback,
  route-around, guard bypass, protected-config edit, broad architecture
  change, or speed benchmark was added.

- Opened test-only GPU BF16-path numeric alignment diagnostics gate
  `20260626_1319_nemotron_nano_gpu_bf16_path_numeric_alignment_diagnostics`.
  Recorded `1259` first: real Nano GPU prototype replay passed structurally,
  focused expert-HQQ tests passed (`42 passed, 0 failed`), the gated real GPU
  proof passed (`1 passed, 0 failed`), selected-slot coverage was `18/18`,
  and runtime integration remains blocked. Gemma speeds remain
  `5619.6 / 92.43 / 155.69`. Current Nemotron token results carried: BF16
  Moby/War/Les all `1044 1044 1044`; plain amax INT4 Moby
  `1321 1272 78526`, War `1044 1108 1078`, Les `1044 1262 1384`;
  search-RMSE INT4 Moby `1321 13540 1314`, War `1044 1083 3843`, Les
  `1044 1262 1384`. Numeric boundary carried from `1259`: GPU-vs-CPU
  sum/max `9.390317355754 / 0.002254664898`; GPU-vs-captured-BF16
  sum/max/L2 `64.231438333169 / 0.009155273438 / 0.380071850012`. Scope:
  instrument only the `cfg(test)` GPU prototype to expose W13 preactivation,
  activation output, and final W2 output; build a CPU BF16-path oracle that
  mirrors GPU BF16 input/storage boundaries; attribute the remaining delta
  before any runtime work. No runtime prefill consumer, config knob,
  auto-selection, decode/HCS, Marlin fallback, route-around, guard bypass,
  protected-config edit, broad architecture change, or speed benchmarking.
  Closed with only `cfg(test)` / env-gated diagnostics in
  `src/weights/expert_hqq.rs`. The GPU prototype exposes W13 preactivation,
  activation output, and final W2 output; a CPU BF16-path oracle mirrors the
  GPU path's BF16 routed-input and intermediate/output storage boundaries.
  Focused expert-HQQ tests now pass (`43 passed, 0 failed`) and the gated real
  alignment proof passed (`1 passed, 0 failed`). GPU vs BF16-path oracle
  matches final output exactly (`0` sum abs); activation and W2 deltas are
  zero, and W13 differs only by subnormal noise (`8.46355932592047e-37` sum).
  The prior GPU-vs-CPU/reference delta is explained by CPU f32 reference vs
  BF16-path oracle at W13 GEMM/output-cast boundaries:
  `reference_oracle_w2_sum_abs=9.390317355753723`, matching `1259` within
  `2.77e-13`. GPU-vs-captured-BF16 branch metrics reproduce `1259` within
  roundoff: sum/max/L2
  `64.231438333168626 / 0.0091552734375 / 0.380071850011764`. Decision:
  runtime integration remains blocked pending an explicit BF16-path numeric
  contract. Validation passed: `./dev build`, focused tests, gated proof,
  artifacts, syntax/py-compile, whitespace, protected configs, source-scope,
  scatter guard, and cleanup.

- Opened real Nano GPU expert-HQQ prototype replay validation gate
  `20260626_1259_nemotron_nano_real_gpu_prototype_replay_validation`.
  Recorded `1236` first: the isolated `cfg(test)` GPU prototype exists,
  focused expert-HQQ tests passed (`41 passed, 0 failed`), the gated synthetic
  GPU proof passed (`1 passed, 0 failed`), and synthetic exact-HQQ6 GPU output
  matched CPU dispatch exactly with sum/max deltas `0.000000000000`. Gemma
  speeds remain `5619.6 / 92.43 / 155.69`. Current Nemotron token results
  carried: BF16 Moby/War/Les all `1044 1044 1044`; plain amax INT4 Moby
  `1321 1272 78526`, War `1044 1108 1078`, Les `1044 1262 1384`;
  search-RMSE INT4 Moby `1321 13540 1314`, War `1044 1083 3843`, Les
  `1044 1262 1384`. Scope: use the real `0836` HQQ6 g64 `KRHQ` cache and
  `1023` routed-input/branch-output capture to drive the `cfg(test)` GPU
  prototype over real selected layer-1 Nano slots, then compare GPU output
  against CPU test dispatch, offline KRHQ reference, and captured BF16 branch
  output using the existing `1023/1217` metric boundaries. No runtime prefill
  consumer, config knob, auto-selection, decode/HCS, Marlin fallback,
  route-around, guard bypass, protected-config edit, broad architecture
  change, or speed benchmarking.
  Closed at a real GPU numeric boundary. Added only an env-gated `cfg(test)`
  real proof in `src/weights/expert_hqq.rs`, using the existing `0836` HQQ6
  g64 `KRHQ` cache and `1023` full routed-input/branch capture. Descriptor,
  plan, row mapping, role pairing, `nbits=6`, `group_size=64`, layout, and
  selected-slot coverage passed for `18/18` slots. The gated real GPU proof
  passed (`1 passed, 0 failed`), focused expert-HQQ tests now pass
  (`42 passed, 0 failed`), and `./dev build` passed. GPU output is close but
  not exact to CPU dispatch/reference: sum abs `9.390317355754`, max abs
  `0.002254664898`. GPU-vs-captured-BF16 branch metrics are sum abs
  `64.231438333169`, max abs `0.009155273438`, L2 `0.380071850012`; deltas
  vs `1023` are `0.003041217992` / `0.000220641494` / `0.000971257408`.
  Decision: next boundary is real GPU BF16-path numeric behavior before any
  runtime prefill consumer. No runtime prefill consumer, config knob,
  auto-selection, decode/HCS, Marlin fallback, route-around, guard bypass,
  protected-config edit, broad architecture change, or speed benchmark was
  added.

- Opened test-only GPU expert-HQQ prefill dispatch prototype gate
  `20260626_1236_nemotron_nano_test_only_gpu_expert_hqq_prefill_dispatch_prototype`.
  Recorded `1217` first: real Nano KRHQ test-only dispatch replay passed,
  focused expert-HQQ tests passed (`39 passed, 0 failed`), dispatch output
  matched the offline reference exactly, and dispatch-vs-BF16 metrics matched
  `1023` (sum abs `64.228397115177`, max abs `0.008934631944`, L2
  `0.379100592604`). Gemma speeds remain `5619.6 / 92.43 / 155.69`.
  Current Nemotron token results carried: BF16 Moby/War/Les all
  `1044 1044 1044`; plain amax INT4 Moby `1321 1272 78526`, War
  `1044 1108 1078`, Les `1044 1262 1384`; search-RMSE INT4 Moby
  `1321 13540 1314`, War `1044 1083 3843`, Les `1044 1262 1384`. Scope:
  inspect CUDA launch/test patterns and HQQ/Marlin kernel interfaces, define
  the smallest test-only GPU boundary for selected-row HQQ W13/W2 execution
  from a validated plan, and only if isolated add a synthetic gated GPU/unit
  test comparing GPU output to the existing CPU dispatch/reference output. No
  runtime prefill consumer, user-facing config, auto-selection, decode/HCS,
  fallback, route-around, guard bypass, protected-config edit, broad
  architecture change, or speed work.
  Closed with an isolated `cfg(test)` GPU prototype in
  `src/weights/expert_hqq.rs`. Added
  `ExpertHqqPrefillGpuPrototypeOutput`,
  `ExpertHqqCache::execute_prefill_test_gpu_prototype`, and
  `execute_prefill_test_gpu_prototype_from_registered_cache`. The prototype
  loads only existing HQQ4/HQQ6 prefill GEMM plus ReLU2/SiLU kernels from PTX,
  validates the KRHQ plan and metadata before CUDA, and writes selected-row
  outputs into the sorted-row buffer. Synthetic exact-HQQ6 ungated GPU proof
  passed (`1 passed, 0 failed`) and matched CPU test dispatch exactly
  (`0` sum/max delta). Focused expert-HQQ tests now pass (`41 passed,
  0 failed`), and `./dev build` passed. No runtime prefill consumer,
  user-facing config, auto-selection, decode/HCS, fallback, route-around,
  guard bypass, protected-config edit, broad architecture change, or speed
  work was added.

- Opened real Nano KRHQ test-only dispatch replay validation gate
  `20260626_1217_nemotron_nano_real_krhq_test_only_dispatch_replay_validation`.
  Recorded `1153` first: the `cfg(test)` weights-module dispatch consumer
  exists, focused expert-HQQ tests passed (`38 passed, 0 failed`), and
  synthetic HQQ6 ungated, HQQ4 ungated, and HQQ6 gated dispatch outputs match
  the offline reference executor. Gemma speeds remain
  `5619.6 / 92.43 / 155.69`. Current Nemotron token results carried: BF16
  Moby/War/Les all `1044 1044 1044`; plain amax INT4 Moby
  `1321 1272 78526`, War `1044 1108 1078`, Les `1044 1262 1384`;
  search-RMSE INT4 Moby `1321 13540 1314`, War `1044 1083 3843`, Les
  `1044 1262 1384`. Scope: use the existing real `0836` HQQ6 g64 `KRHQ`
  cache plus the `1023` full routed-input and branch-output capture to drive
  the test-only dispatch consumer over real selected layer-1 Nano slots, then
  compare dispatch output against the offline KRHQ reference executor and
  captured BF16 branch output. No user-facing config, auto-selection, runtime
  prefill consumer, decode/HCS, fallback to Marlin, route-around, guard
  bypass, protected-config edit, broad architecture change, or speed work.
  Closed with a real-data, env-gated test-only dispatch replay proof. The
  test uses the existing `0836` HQQ6 g64 `KRHQ` cache and the existing `1023`
  full routed-input/branch-output capture. Dispatch output matched the offline
  KRHQ reference executor exactly over `48384` values, with sum/max delta
  `0`. Dispatch-vs-captured-BF16 reproduced the `1023` HQQ branch metrics
  exactly: sum abs `64.228397115177`, max abs `0.008934631944`, L2
  `0.379100592604`, all deltas vs `1023` equal `0`. Descriptor/plan lookup,
  W13/W2 role pairing, row mapping, `nbits=6`, `group_size=64`, layout, and
  selected-slot coverage `18/18` were validated. Focused expert-HQQ tests now
  pass (`39 passed, 0 failed`), the gated real dispatch proof passed
  (`1 passed, 0 failed`), and `./dev build` passed. No user-facing config,
  auto-selection, runtime prefill consumer, decode/HCS, fallback to Marlin,
  route-around, guard bypass, protected-config edit, broad architecture
  change, or speed work was added.

- Opened real Nemotron test-only expert-HQQ prefill selected-expert dispatch
  integration gate
  `20260626_1153_nemotron_nano_test_only_expert_hqq_prefill_dispatch_integration`.
  Recorded `1023` first: full routed-input capture and offline branch replay
  passed, focused expert-HQQ tests passed (`31 passed, 0 failed`), gated
  full-branch proof passed, and BF16 replay was closer than HQQ on all
  selected slots (`18/18`). Gemma speeds remain `5619.6 / 92.43 / 155.69`.
  Final HQQ6 g64 branch-vs-BF16 captured delta carried: sum abs
  `64.228397115177`, max abs `0.008934631944`, L2 `0.379100592604`. Current
  Nemotron token results carried: BF16 Moby/War/Les all `1044 1044 1044`;
  plain amax INT4 Moby `1321 1272 78526`, War `1044 1108 1078`, Les
  `1044 1262 1384`; search-RMSE INT4 Moby `1321 13540 1314`, War
  `1044 1083 3843`, Les `1044 1262 1384`. Scope: define the minimal generic
  dispatch boundary from a validated `ExpertHqqPrefillDispatchPlan` plus
  `KRHQ` W13/W2 payloads into a test-only execution path, and add only a
  gated/test-only dispatch consumer if it stays isolated and can be compared
  against the offline KRHQ reference executor on synthetic tensors. No
  user-facing runtime config, auto-selection, decode/HCS, fallback to Marlin,
  route-around, guard bypass, protected-config edit, broad architecture
  change, or speed work.
  Closed with a `cfg(test)` weights-module dispatch consumer only. Added
  `ExpertHqqPrefillTestDispatchOutput`,
  `ExpertHqqCache::execute_prefill_test_dispatch`, and
  `execute_prefill_test_dispatch_from_registered_cache`. Synthetic dispatch
  output matches the existing offline reference executor for HQQ6 ungated,
  HQQ4 ungated, and HQQ6 gated cases. Fail-closed tests cover missing
  registered metadata, unsupported nbits/layout/group metadata, shape mismatch,
  row mismatch, role mismatch, absent W13/W2, and no Marlin fallback. Focused
  expert-HQQ tests now pass (`38 passed, 0 failed`) and `./dev build` passed.
  Validation passed: artifact assertions, syntax/py-compile, whitespace,
  protected configs, source-scope, scatter guard, blocked-work checks, and
  cleanup. No user-facing runtime config, auto-selection, `gpu_prefill`
  consumer, decode/HCS, fallback to Marlin, route-around, guard bypass,
  protected-config edit, broad architecture change, or speed work was added.

- Opened real Nemotron full routed-input capture and offline full
  expert-branch replay gate
  `20260626_1023_nemotron_nano_full_routed_input_offline_branch_replay`.
  Recorded `1004` first: real Nano HQQ6 g64 `KRHQ` reference execution matched
  prior W13/W2 component proofs, focused expert-HQQ tests passed (`30 passed,
  0 failed`), gated real Nano proof passed, and exact full W13 routed-input
  vectors were still missing. Gemma speeds remain `5619.6 / 92.43 / 155.69`.
  Current Nemotron token results carried: BF16 Moby/War/Les all
  `1044 1044 1044`; plain amax INT4 Moby `1321 1272 78526`, War
  `1044 1108 1078`, Les `1044 1262 1384`; search-RMSE INT4 Moby
  `1321 13540 1314`, War `1044 1083 3843`, Les `1044 1262 1384`. Scope: add
  only a request-scoped/off-by-default routed-input diagnostic if existing
  artifacts still lack exact layer-1 selected-expert inputs, then use those
  inputs with the existing `KRHQ` reference executor and BF16 safetensors to
  validate full W13 activation plus W2 projection against BF16 branch outputs.
  No user-facing runtime config, GPU dispatch consumer, decode/HCS, fallback
  to Marlin, route-around, guard bypass, protected-config edit, broad
  architecture change, or speed work.
  Closed with full routed-input capture and offline full expert-branch replay
  validated. Added only `KRASIS_REFERENCE_BRANCH_REPLAY_FULL`, an
  off-by-default/request-scoped BF16 trace for layer-1 selected routed inputs
  and branch outputs, with stream synchronization before full-vector downloads.
  Captured `385` routed vectors and `385` branch vectors; selected slot
  coverage was `18/18` for both. BF16 CPU replay with BF16 rounding matched
  captured branch output with sum abs `5.976875055581` over `48384` values and
  max abs `0.00390625`; HQQ6 g64 branch-vs-BF16 captured sum abs was
  `64.228397115177`, max abs `0.008934631944`, L2 `0.379100592604`. BF16
  replay was closer than HQQ on every selected slot (`18/18`). Focused
  expert-HQQ tests now pass (`31 passed, 0 failed`) and the gated real
  full-branch proof passed (`1 passed, 0 failed`). Validation passed:
  `./dev build`, BF16 READY trace requests, artifact assertions,
  syntax/py-compile, whitespace, protected configs, source-scope, scatter
  guard, blocked-work checks, and cleanup. No user-facing runtime config, GPU
  dispatch consumer, decode/HCS, fallback to Marlin, route-around, guard
  bypass, protected-config edit, broad architecture change, or speed work was
  added.

- Opened real Nemotron expert-HQQ reference-execution validation gate
  `20260626_1004_nemotron_nano_real_expert_hqq_reference_execution_validation`.
  Recorded `0938` first: a weights-module-only offline reference executor
  exists for validated `KRHQ` W13/W2 dispatch plans, focused expert-HQQ tests
  passed (`29 passed, 0 failed`), and there is still no model runtime,
  user-facing runtime config, GPU prefill dispatch consumer, decode/HCS,
  fallback to Marlin, route-around, guard bypass, protected-config edit, broad
  architecture change, or speed work. Gemma speeds remain
  `5619.6 / 92.43 / 155.69`. Current Nemotron token results carried: BF16
  Moby/War/Les all `1044 1044 1044`; plain amax INT4 Moby
  `1321 1272 78526`, War `1044 1108 1078`, Les `1044 1262 1384`;
  search-RMSE INT4 Moby `1321 13540 1314`, War `1044 1083 3843`, Les
  `1044 1262 1384`. Scope: use the existing real `0836` HQQ6 g64 `KRHQ` cache
  and `0629/0641` selected-slot artifacts to drive the offline reference
  executor over real Nano layer-1 selected experts. No runtime config, GPU
  prefill dispatch, decode/HCS, fallback to Marlin, route-around, guard
  bypass, protected-config edit, broad architecture change, or speed work.
  Closed with real Nano offline `KRHQ` reference-execution validation against
  the existing `0836` HQQ6 g64 cache. The executor validated `14` tensor
  records, `18` selected plan entries, `108` W13 rows, and `18` W2 selected
  rows. W13 reproduced the prior HQQ6 g64 component proof within `5.1959e-8`
  total abs-error delta (`0.086280518677` vs `0.086280570636`), W13 dot max
  delta was `1.9820e-8`, ungated `relu2` activation delta was `0`, W2 readback
  reproduced the prior proof within `1.6529972e-5`, and executor projection
  self-delta was `0`. Existing artifacts lack exact full routed-input vectors,
  so this validates component replay from the real cache rather than a full
  branch-output runtime replay. Focused tests now pass (`30 passed, 0 failed`)
  and the gated real proof passed (`1 passed, 0 failed`). Validation passed:
  `./dev build`, artifacts, syntax/py-compile, whitespace, protected configs,
  source-scope, scatter guard, blocked-work checks, and cleanup. No runtime
  config, GPU prefill dispatch consumer, decode/HCS, fallback to Marlin,
  route-around, guard bypass, protected-config edit, broad architecture change,
  or speed work was added.

- Opened real Nemotron expert-HQQ prefill selected-expert reference execution
  gate
  `20260626_0938_nemotron_nano_expert_hqq_prefill_selected_expert_reference_execution`.
  Recorded `0921` first: metadata-only prefill dispatch contract planning
  exists for registered `KRHQ` W13/W2 descriptors, focused expert-HQQ tests
  passed (`20 passed, 0 failed`), and there is still no CUDA dispatch consumer,
  user-facing runtime config, decode/HCS, fallback to Marlin, route-around,
  guard bypass, protected-config edit, broad architecture change, or speed
  work. Gemma speeds remain `5619.6 / 92.43 / 155.69`. Current Nemotron token
  results carried: BF16 Moby/War/Les all `1044 1044 1044`; plain amax INT4
  Moby `1321 1272 78526`, War `1044 1108 1078`, Les `1044 1262 1384`;
  search-RMSE INT4 Moby `1321 13540 1314`, War `1044 1083 3843`, Les
  `1044 1262 1384`. Scope: add only a synthetic/offline reference execution
  path over a validated `ExpertHqqPrefillDispatchPlan`, with dequantized
  `KRHQ` W13/W2 payloads and selected-expert W13 activation plus W2 projection
  semantics. No model runtime, config, GPU prefill dispatch, decode/HCS,
  fallback to Marlin, route-around, guard bypass, protected-config edit, broad
  architecture change, or speed work.
  Closed with a weights-module-only synthetic/offline reference executor.
  Added `ExpertHqqPrefillReferenceOutput`,
  `ExpertHqqCache::execute_prefill_reference`, and
  `execute_prefill_reference_from_registered_cache`. The executor consumes a
  validated dispatch plan, dequantizes `KRHQ` HQQ4/HQQ6 W13/W2 payloads,
  applies gated `silu(gate) * up` or ungated `relu(preact)^2`, and projects
  through W2. Focused expert-HQQ tests now pass (`29 passed, 0 failed`) with
  HQQ4/HQQ6 happy paths and fail-closed coverage for missing plan entries,
  row-range mismatch, missing row coverage, wrong W13/W2 role pairing,
  unsupported nbits/group/layout metadata, and no Marlin fallback. Validation
  passed: final `./dev build`, focused tests, artifact assertions,
  syntax/py-compile, whitespace checks, protected-config status,
  source-scope audit, scatter guard, blocked-work checks, and cleanup. No model
  runtime, user-facing config, GPU prefill dispatch consumer, decode/HCS,
  fallback to Marlin, route-around, guard bypass, protected-config edit, broad
  architecture change, or speed work was added.

- Opened real Nemotron expert-HQQ prefill selected-expert dispatch contract
  gate
  `20260626_0921_nemotron_nano_expert_hqq_prefill_selected_expert_dispatch_contract`.
  Recorded `0901` first: metadata-only `KRHQ` registration exists through
  explicit path-based `WeightStore` APIs, focused expert-HQQ tests passed
  (`16 passed, 0 failed`), and there is still no auto-discovery, runtime
  config, prefill dispatch, decode/HCS, fallback, route-around,
  protected-config edit, broad architecture change, or speed work. Gemma speeds
  remain `5619.6 / 92.43 / 155.69`. Current Nemotron token results carried:
  BF16 Moby/War/Les all `1044 1044 1044`; plain amax INT4 Moby
  `1321 1272 78526`, War `1044 1108 1078`, Les `1044 1262 1384`;
  search-RMSE INT4 Moby `1321 13540 1314`, War `1044 1083 3843`, Les
  `1044 1262 1384`. Scope: inspect the existing prefill expert dispatch path
  and `KRHQ` descriptor lookup APIs, define the minimal generic dispatch
  contract for registered W13/W2 metadata, and add only a small synthetic
  unit/offline contract test if it stays isolated. No user-facing runtime
  config, decode/HCS, fallback to Marlin, route-around, guard bypass,
  protected-config edit, broad architecture change, or speed work.
  Closed with metadata-only prefill selected-expert dispatch contract plumbing.
  Added `ExpertHqqPrefillWork`, `ExpertHqqPrefillDispatchEntry`,
  `ExpertHqqPrefillDispatchPlan`, `ExpertHqqCache::prefill_dispatch_plan`, and
  `prefill_dispatch_plan_from_registered_cache`. The contract mirrors current
  prefill active-expert work (`expert_idx`, sorted row `offset/count`) and
  validates registered W13/W2 descriptor pairs, gated vs ungated W13 shape, W2
  shape, axis/layout, nbits-derived layout, payload lengths, duplicate selected
  experts, nonzero rows, and absent/incomplete metadata before dispatch.
  Focused expert-HQQ tests now pass (`20 passed, 0 failed`). Validation passed:
  `./dev build`, focused tests, artifact assertions, syntax/py-compile,
  whitespace checks, protected-config status, source-scope audit, scatter
  guard, blocked-work checks, and cleanup. No user-facing runtime config,
  model-load caller, prefill CUDA dispatch consumer, decode/HCS, fallback to
  Marlin, route-around, guard bypass, protected-config edit, broad architecture
  change, or speed work was added. Runtime expert-HQQ remains unwired.

- Opened real Nemotron expert-HQQ model-load registration metadata gate
  `20260626_0901_nemotron_nano_expert_hqq_model_load_registration_metadata`.
  Recorded `0836` first: real Nano selected-expert `KRHQ` HQQ6 g64
  build/readback validation passed for layer-1 experts
  `26,42,47,72,88,89,112`, focused expert-HQQ tests passed
  (`11 passed, 0 failed`), the gated real proof passed (`1 passed, 0 failed`),
  and runtime expert-HQQ remains intentionally unwired. Gemma speeds remain
  `5619.6 / 92.43 / 155.69`. Current Nemotron token results carried: BF16
  Moby/War/Les all `1044 1044 1044`; plain amax INT4 Moby
  `1321 1272 78526`, War `1044 1108 1078`, Les `1044 1262 1384`;
  search-RMSE INT4 Moby `1321 13540 1314`, War `1044 1083 3843`, Les
  `1044 1262 1384`. Scope: inspect `WeightStore`, the `KRHQ` loader, and
  expert Marlin registration paths, then add only explicit fail-closed
  registration metadata from a provided `KRHQ` path if it stays isolated.
  Tests must cover matching metadata registration, layer/expert/projection
  lookup, duplicate/missing descriptor failure, model-shape mismatch, and no
  fallback to Marlin. No cache auto-discovery, runtime config, prefill
  dispatch, decode/HCS, fallback, route-around, protected-config edit, broad
  architecture change, or speed work.
  Closed with explicit fail-closed model-load registration metadata only.
  Added `ExpertHqqTensorKey`, `ExpertHqqCache` descriptor lookup/requirement
  validation, and `WeightStore::register_expert_hqq_cache_from_path(path,
  config_hash, required)`. The registration derives expected model shape from
  `WeightStore` plus an explicit config hash, loads only the provided `KRHQ`
  path, requires a non-empty descriptor set, validates each requested
  role/layer/expert projection, and attaches metadata only after all checks
  pass. Added `WeightStore::require_expert_hqq_tensor` for explicit metadata
  lookup. Focused expert-HQQ tests now pass (`16 passed, 0 failed`), covering
  matching registration, lookup, missing descriptor failure, duplicate
  descriptor cache failure, model-shape mismatch, empty implicit requirement
  rejection, and no Marlin fallback on failed registration. Validation passed:
  `./dev build`, focused tests, artifact assertions, syntax/py-compile,
  whitespace checks, protected-config status, source-scope audit, scatter
  guard, blocked-work checks, and cleanup. No cache auto-discovery, runtime
  config, prefill dispatch, decode/HCS, fallback, route-around,
  protected-config edit, broad architecture change, or speed work was added.
  Next boundary: explicit model-load caller or prefill selected-expert dispatch
  in a later gate.

- Opened real Nemotron selected-expert `KRHQ` cache build/readback validation
  gate `20260626_0836_nemotron_nano_real_expert_hqq_cache_readback_validation`.
  Recorded `0813` first: offline safetensors-to-`KRHQ` builder exists, focused
  expert-HQQ tests passed (`10 passed, 0 failed`), runtime expert-HQQ remains
  intentionally unwired, and Gemma speeds remain `5619.6 / 92.43 / 155.69`.
  Current Nemotron token results carried: BF16 Moby/War/Les all
  `1044 1044 1044`; plain amax INT4 Moby `1321 1272 78526`, War
  `1044 1108 1078`, Les `1044 1262 1384`; search-RMSE INT4 Moby
  `1321 13540 1314`, War `1044 1083 3843`, Les `1044 1262 1384`. Scope: use
  the new builder on real Nano layer-1 selected experts from the `0629/0641`
  artifacts for W13+W2, HQQ6 g64 first, optionally cheap HQQ variants for
  comparison. Write a real benchmark cache under `benchmarks/` or a temp
  benchmark path, read it back through `KRHQ`, assert descriptors/payload byte
  counts/roles/shape metadata, and replay the same selected W13/W2 dot products
  from readback payloads to confirm previous offline HQQ6 gains. If readback
  replay diverges from prior proof, stop at the builder/readback mismatch. No
  runtime config, model-load integration, prefill dispatch, decode/HCS,
  fallback, route-around, protected-config edit, broad architecture change, or
  speed work.
  Closed with real Nano selected-expert `KRHQ` HQQ6 g64 cache/readback
  validation passing. Built a `61,115,664` byte benchmark cache for seven
  selected layer-1 experts (`26,42,47,72,88,89,112`) containing `14` records
  (`7` W13 + `7` W2). Readback validated role, layer, expert, `nbits=6`,
  `group_size=64`, shape metadata, and packed/scales/zeros byte counts. Replay
  from readback payloads reproduced prior W13 HQQ6 g64 exactly
  (`0.086280570636`, `0.247560523989` of amax) and W2 within `1.653e-5`
  total error (`47.840301673001`, `0.198534661059` of amax). The temporary
  real-proof test is env-gated by `KRASIS_REAL_NANO_KRHQ_PROOF=1`; the normal
  focused expert-HQQ test set now passes (`11 passed, 0 failed`). Validation
  passed: `./dev build`, real gated proof, focused tests, artifact assertions,
  syntax/py-compile, whitespace checks, protected-config status, source-scope
  audit, scatter guard, blocked-work checks, and cleanup. No runtime config,
  model-load integration, prefill dispatch, decode/HCS, fallback, route-around,
  protected-config edit, broad architecture change, or speed work was added.
  Next boundary: model-load registration metadata or prefill selected-expert
  dispatch, in a separate gate.

- Opened offline expert-HQQ W13/W2 cache builder-from-safetensors gate
  `20260626_0813_nemotron_nano_expert_hqq_cache_builder_safetensors`. Recorded
  `0747` first: isolated `KRHQ` writer/loader plumbing is in place, focused
  expert-HQQ tests passed (`7 passed, 0 failed`), runtime expert-HQQ remains
  unwired, and no runtime config/model-load integration/prefill dispatch/
  decode/HCS/speed path exists. Gemma speeds carried:
  `5619.6 / 92.43 / 155.69`. HQQ6 g64 offline fidelity gains carried: W13
  `0.247560524` of amax and W2 `0.198534730` of amax; HQQ4 remained
  insufficient. Scope: inspect existing HQQ quantizer code and the `KRHQ`
  writer input contract, then add only a generic offline builder from explicit
  safetensors paths/keys and layer/expert/projection selection if it can
  quantize W13/W2 into `KRHQ` HQQ4/HQQ6 payloads and validate fail-closed
  readback. Tests should use tiny synthetic safetensors or in-memory tensors for
  W13/W2 role/shape/key selection, payload byte counts, and metadata mismatch
  failures. No runtime config, model-load integration, prefill dispatch,
  decode/HCS, fallback, route-around, protected-config edit, broad architecture
  change, or speed work.
  Closed with offline safetensors-builder plumbing only. Added explicit
  safetensors tensor specs for W13/W2 role, layer, expert, `nbits`, group size,
  path, and key; the builder accepts only 2D BF16/F16/F32 tensors, quantizes
  them into `KRHQ` HQQ4/HQQ6 payloads, writes the cache, and validates
  fail-closed readback. HQQ4 reuses the existing Rust HQQ4 search quantizer via
  a crate-local helper, while HQQ6 uses isolated row-major axis-1 affine grouped
  quantization and uint6 packing for offline cache payloads. Focused Rust tests
  now pass (`10 passed, 0 failed`), covering tiny synthetic safetensors W13/W2
  HQQ4/HQQ6 round-trip, key selection, role/shape mismatch, payload byte counts,
  metadata/header mismatch, and duplicate descriptors. Validation passed:
  `./dev build`, focused Rust tests, artifact assertions, syntax/py-compile,
  whitespace checks, protected-config status, source-scope audit, scatter guard,
  blocked-work checks, and cleanup. No runtime config, model-load integration,
  prefill dispatch, decode/HCS, fallback, route-around, protected-config edit,
  broad architecture change, or speed work was added.

- Opened expert-HQQ W13/W2 cache writer/loader gate
  `20260626_0747_nemotron_nano_expert_hqq_cache_writer_loader`. Recorded `0717`
  first: `KRHQ` v1 metadata/read-write plumbing and optional
  `WeightStore.expert_hqq_cache` metadata are in place, focused fail-closed
  tests passed (`4 passed, 0 failed`), and runtime expert-HQQ is still unwired.
  Carried HQQ6 g64 offline fidelity gains: W13 `0.247560524` of amax and W2
  `0.198534730` of amax; HQQ4 remained insufficient. Scope: inspect existing
  HQQ quantizer code and cache writer patterns, then implement only isolated
  generic offline writer/reader plumbing if it can take explicit tensor plus
  metadata input, write/read `KRHQ`, and validate fail-closed. Tests should
  cover tiny synthetic W13/W2 HQQ4/HQQ6 round-trip, payload byte counts,
  role/shape mismatch, nbits/group/axis mismatch, duplicate descriptors. No
  runtime config, model-load integration, prefill dispatch, decode/HCS,
  fallback, route-around, protected-config edit, broad architecture change, or
  speed work.
  Closed with isolated offline writer/loader plumbing only. Added explicit
  `ExpertHqqTensorInput`, `ExpertHqqCache::from_inputs`,
  `write_expert_hqq_cache_from_inputs`, and `load_expert_hqq_cache` to write
  `KRHQ` caches from caller-provided W13/W2 HQQ packed/scales/zeros payloads and
  immediately read them back with expected-header fail-closed validation.
  Focused expert-HQQ Rust tests now pass (`7 passed, 0 failed`), covering tiny
  synthetic W13/W2 HQQ4/HQQ6 round-trip, payload byte counts, role/shape
  mismatch, nbits/group/axis mismatch, duplicate descriptors, header mismatch,
  and corrupt metadata. Validation passed: `./dev build`, focused Rust tests,
  artifact assertions, syntax/py-compile, whitespace checks, protected-config
  status, source-scope audit, scatter guard, blocked-work checks, and cleanup.
  No runtime config, model-load integration, prefill dispatch, decode/HCS,
  fallback, route-around, protected-config edit, broad architecture change, or
  speed work was added.

- Opened expert-HQQ W13/W2 cache/header and `WeightStore` descriptor plumbing
  gate `20260626_0717_nemotron_nano_expert_hqq_cache_descriptor_plumbing`.
  Recorded `0641` first: offline HQQ6 g64 materially improved selected layer 1
  W13 (`0.247560524` of amax) and W2 (`0.198534730` of amax), HQQ4 was not
  enough, W2 search-RMSE was unavailable, and runtime expert-HQQ support is
  still absent. Gemma speeds carried: `5619.6 / 92.43 / 155.69`. Current
  Nemotron tokens carried: BF16 Moby/War/Les all `1044 1044 1044`; plain amax
  INT4 Moby `1321 1272 78526`, War `1044 1108 1078`, Les `1044 1262 1384`;
  search-RMSE INT4 Moby `1321 13540 1314`, War `1044 1083 3843`, Les
  `1044 1262 1384`. Scope: descriptor/header read-write plumbing and tests
  only if generic. No runtime config, dispatch path, decode/HCS, fallback,
  route-around, protected-config edit, broad architecture change, or speed work.
  Closed with descriptor/header/read-write plumbing only. Added
  `src/weights/expert_hqq.rs`, a standalone `KRHQ` v1 expert-HQQ metadata cache
  contract for W13/W2 tensors with fail-closed validation of model/config
  header, `nbits`, group size, axis/layout, tensor role, layer/expert/projection
  indexing, role-specific shapes, dtypes, and payload byte counts. Added
  optional `WeightStore.expert_hqq_cache` metadata initialized to `None` across
  existing load paths; no runtime registration or dispatch consumes it yet.
  Focused Rust tests passed (`4 passed, 0 failed`): W13/W2 round-trip, header
  mismatch rejection, corrupt tensor metadata rejection, and duplicate
  descriptor rejection. Validation passed: `./dev build`, focused expert-HQQ
  Rust unit tests, artifact assertions, syntax/py-compile, whitespace checks,
  protected-config status, source-scope audit, scatter guard, blocked-work
  checks, and cleanup. No runtime config, dispatch, decode/HCS, kernel path,
  fallback, route-around, protected-config edit, broad architecture change, or
  speed work was added.

- Opened offline expert-HQQ W2 fidelity proof gate
  `20260626_0641_nemotron_nano_offline_expert_hqq_w2_fidelity_proof`.
  Recorded `0629` first: offline HQQ6 g64 materially reduced selected W13 dot
  error (`hqq6_g64/amax=0.247560524`), HQQ4 did not materially improve W13, W2
  remained untested, and runtime expert-HQQ wiring still does not exist. Gemma
  speeds carried: `5619.6 / 92.43 / 155.69`. Current Nemotron tokens carried:
  BF16 Moby/War/Les all `1044 1044 1044`; plain amax INT4 Moby
  `1321 1272 78526`, War `1044 1108 1078`, Les `1044 1262 1384`;
  search-RMSE INT4 Moby `1321 13540 1314`, War `1044 1083 3843`, Les
  `1044 1262 1384`. Scope: offline W2 fidelity proof only. Use existing source,
  artifacts, safetensors, and cache bytes first to compare BF16 W2
  weights/output against amax, search-RMSE if available, HQQ4 g64/g128, and HQQ6
  g64/g128 for the same layer 1 selected experts/prompts. Do not add HQQ cache
  format, runtime config, kernel path, fallback, route-around, protected-config
  edit, broad architecture change, or speed work.
  Closed with one request-scoped/off-by-default W2-input trace diagnostic only;
  no runtime HQQ cache/config/kernel path was added. BF16 and INT4 both reached
  READY before deterministic trace requests. Offline proof used exact BF16 W2
  input plus layer 1 W2 `down_proj` safetensors for `18` selected slots across
  experts `26,42,47,72,88,89,112`. W2 search-RMSE was unavailable because the
  existing calibration samples contain `up_proj` only. W2 output abs-error sums:
  amax g64 `240.967000008`, HQQ4 g64 `179.110133171`, HQQ4 g128
  `198.624023914`, HQQ6 g64 `47.840318203`, HQQ6 g128 `52.510519505`. Decision:
  HQQ6 g64 materially improves W2 fidelity too (`0.198534730` of amax), while
  HQQ4 is not enough. Minimal next boundary is expert-HQQ W13+W2 cache/header
  metadata, `WeightStore` descriptors, and prefill selected-expert dispatch
  before decode/HCS. Validation passed and cleanup is clean.

- Opened offline expert-HQQ W13/W2 fidelity proof gate
  `20260626_0629_nemotron_nano_offline_expert_hqq_w13_fidelity_proof`.
  Recorded `0619` first: HQQ support is currently attention-side only, while
  expert W13/W2 still use the Marlin INT cache path and cannot be tested through
  a safe runtime HQQ config yet. Gemma speeds carried:
  `5619.6 / 92.43 / 155.69`. Current Nemotron tokens carried: BF16 Moby/War/Les
  all `1044 1044 1044`; plain amax INT4 Moby `1321 1272 78526`, War
  `1044 1108 1078`, Les `1044 1262 1384`; search-RMSE INT4 Moby
  `1321 13540 1314`, War `1044 1083 3843`, Les `1044 1262 1384`. Scope:
  offline proof only. Use existing HQQ source/tests, prior W13 artifacts,
  safetensors, and cache bytes to compare existing HQQ k4v4/k6v6
  quantize/dequantize replay against BF16, amax, and search-RMSE selected W13
  dots. Do not add runtime HQQ config, cache format, kernel path, fallback,
  route-around, protected-config edit, broad architecture change, or speed work.
  Closed with no source change and no runtime HQQ wiring. Offline selected W13
  proof covered `108` layer 1 row cases across experts
  `26,42,47,72,88,89,112`. HQQ4 did not materially improve error:
  `hqq4_g128/amax=1.057955285`, `hqq4_g64/amax=0.868006364`. HQQ6 did:
  `hqq6_g128/amax=0.284312253`, `hqq6_g64/amax=0.247560524`; best HQQ6 g64 is
  `0.547096861` of search-RMSE. Decision: offline HQQ6 is a credible W13
  fidelity mitigation signal, but W2 was not tested and runtime support is not
  wired. Minimal next boundary is expert-HQQ W13 cache/header/metadata,
  `WeightStore` descriptors, and prefill selected-expert dispatch. Validation
  passed and cleanup is clean.

- Opened HQQ INT4 correctness mitigation gate
  `20260626_0619_nemotron_nano_hqq_int4_correctness_mitigation`.
  Recorded `0535` first: existing non-HQQ `gpu_expert_int4_calib=search_rmse`
  reduced selected W13 offline dot error (`search_rmse/amax=0.452498531`) but
  did not fix target deterministic token divergence. Fresh BF16 baseline:
  Moby/War/Les all `1044 1044 1044`. Plain amax INT4: Moby
  `1321 1272 78526`, War `1044 1108 1078`, Les `1044 1262 1384`.
  Search-RMSE INT4: Moby `1321 13540 1314`, War `1044 1083 3843`, Les
  `1044 1262 1384`. Current boundary: plain non-HQQ INT4 expert quantization
  fidelity remains the correctness blocker. Gemma speeds carried:
  `5619.6 / 92.43 / 155.69`. Scope: inspect existing HQQ/kNvN source/config
  support before changing anything; prove whether HQQ can replace only expert
  W13/W2 quantized cache path generically without kernel/layout hardcoding. If
  an existing `tests/` HQQ config is wired, run the smallest deterministic
  BF16-vs-HQQ comparison on the same three prompts; otherwise stop and record
  what is missing. Speed work remains blocked; no fallback, route-around, guard
  bypass, protected-config edit, or broad architecture change.
  Closed with no production source change and no runtime HQQ comparison. Source
  inspection proved existing HQQ support is attention-only; routed experts still
  use Marlin INT4/INT8 W13/W2 cache buffers through `WeightStore`, prefill
  `MarlinWeight`, and resident decode/HCS packed/scales metadata. No existing
  Nemotron HQQ config exists, and existing HQQ k4v4/k6v6 configs are
  attention-HQQ configs rather than expert-only W13/W2 replacement tests.
  Decision: stop at the missing expert-HQQ contract: cache
  format/namespace/header, `WeightStore` expert HQQ metadata, prefill
  selected-expert dispatch, resident decode/HCS staging/GEMV, and a fail-closed
  tests-only config knob. Validation passed and cleanup is clean.

- Opened INT4 W13 quantization fidelity mitigation gate
  `20260626_0535_nemotron_nano_int4_w13_quantization_fidelity_mitigation`.
  Recorded `0513` first: the layer 1 W13 cache contract is correct and the
  remaining boundary is normal plain amax INT4 W13 fidelity versus BF16, not
  wrong weight source, tensor orientation, permutation, scale generation,
  group-size propagation, GEMM/reduction, routing, or sampling. `0513`
  confirmed full selected-expert W13 cached scales matched BF16 amax
  `545664/545664`, cached nibbles matched `34922496/34922496`, same-cache dot
  replay matched within `4.47034836e-07`, while BF16-vs-INT4 selected-dim
  error remained (`mean=0.016407737988333334`, `max=0.0585808828`). Current
  Gemma speeds carried: `5619.6 / 92.43 / 155.69`. HQQ/k4v4 and speed work
  remain blocked. Scope: inspect existing quantization modes/source/configs to
  prove whether Krasis already supports a generic non-HQQ INT4 expert W13
  calibration/scale mode such as calibrated amax/search/RMSE that can reduce
  layer 1 W13 error without kernel changes or Nano hardcoding. If yes, create a
  new test config under `tests/` only and run deterministic BF16/INT4
  comparison; if no mode applies, stop and record plain amax INT4 as the
  current correctness blocker.
  Closed with no production source change. Existing generic non-HQQ mode
  `gpu_expert_int4_calib=search_rmse` is wired through config/cache generation
  and can be tested without kernel changes or Nano hardcoding. Added only
  `tests/nemotron-nano-4-4-k6v6-a16-search-rmse.conf` and benchmark artifacts
  with explicit calibration samples. Offline selected W13 dot error improved
  (`search_rmse/amax=0.452498531`), but deterministic token correctness did not:
  Moby token0 stayed `1321` instead of BF16 `1044`; War token1 changed from
  `1108` to `1083` but still missed `1044`; Les token1 stayed `1262`. Decision:
  existing search-RMSE reduces the local W13 probe error but does not mitigate
  the target divergence, so plain non-HQQ INT4 expert quantization fidelity
  remains the current Nano correctness blocker. Validation passed and cleanup
  is clean.

- Opened prefill layer 1 W13 quantized-cache fidelity gate
  `20260626_0513_nemotron_nano_int4_prefill_layer1_w13_cache_fidelity`.
  Recorded the `0437` result first: routed input matched BF16 for `18/18`
  selected final-token expert slots, BF16 W13 reference rows joined to the
  prior trace for `18/18`, BF16-vs-INT4 W13 preactivation hashes matched
  `0/18`, INT4 prefill W13 used effective `group_size=64`,
  `num_groups_w1=42`, `q_type=U4B8`, no pointer/scale slots were missing,
  same-cache normal row-major replay tracked GPU W13 output
  (`max_l2=0.0244934056`), and alternate layout was rejected
  (`min_alt_l2_over_normal_l2=599.688`). Current boundary: quantized W13
  cache/dequant fidelity versus BF16 W13 reference, not wrong prefill
  group-size propagation, layout/permutation, missing scales, Marlin launch
  metadata, GEMM/reduction, routed input, routing, or sampling. Scope: use
  existing `0437` artifacts/source/cache bytes before adding diagnostics. For
  Moby final prefill token and one selected expert, map BF16 W13 weights plus
  routed input to the exact INT4 cached/dequantized W13 values and replayed dot
  product. Determine whether the W13 mismatch is normal INT4 amax quantization
  error or a wrong weight source/tensor orientation/permutation/scale-generation
  contract, then confirm across all three prompts and selected experts. If
  evidence is insufficient, add only a request-scoped/off-by-default
  cache-fidelity diagnostic for selected layer 1 W13 entries. No HQQ/k4v4,
  speed work, sampling change, fallback, route-around, guard bypass,
  protected-config edit, or broad architecture change.
  Closed with no source change and no new diagnostic. Existing `0437`
  artifacts, cache bytes, and BF16 safetensors were sufficient. For the seven
  unique selected layer 1 experts, full W13 cached scales matched the BF16 amax
  quantizer `545664/545664`, and cached nibbles matched
  `34922496/34922496`. For all `18/18` selected prompt/expert slots, selected
  dims had matching scale and nibble contracts. Moby expert 89 dim 0 exact
  top-input entries matched scale/u4/dequant values one-for-one. Same-cache dot
  replay matched the existing normal replay within `4.47034836e-07`, while
  BF16-vs-INT4 selected-dim error remained (`mean=0.016407737988333334`,
  `max=0.0585808828`). Decision: normal amax INT4 quantization fidelity
  boundary for Nano W13, not wrong weight source, tensor orientation,
  permutation, scale generation, group-size propagation, GEMM/reduction,
  routing, or sampling. HQQ/k4v4 and speed work remain blocked. Validation
  passed: `./dev build`, artifact assertions, `bash -n dev`, py-compile,
  whitespace checks, protected-config status, source-scope audit confirming no
  current-gate source diagnostic, scatter guard, blocked-work checks, and
  cleanup.

- Opened prefill layer 1 W13/dequant preactivation producer gate
  `20260626_0437_nemotron_nano_int4_prefill_layer1_w13_dequant_preactivation_producer`.
  Recorded the `0353` result first: layer 1 `pre_mlp_hidden`, router/top-k
  ids, top-k weights, and routed input matched BF16 across Moby, War, and Les
  for `18/18` selected final-token expert slots, while the first mismatch was
  layer 1 W13 preactivation/dequant output (`0/18` slots matched);
  activation/W2 input, W2 output, combined branch output, and residual handoff
  are downstream; all compared summaries were finite with zero NaNs. Carried
  `0147`: decode graph group size is propagated from loaded
  `WeightStore.group_size`; W13 replay uses effective `group_size=64`, has no
  missing/nonfinite scales, and max GPU/CPU replay diff is `6.568e-37`, but
  quality still fails. Carried `0247`: Moby's token0 flip is upstream of
  LM-head projection, with matching input token hash/final position but
  divergent final residual, final norm, and `final_hidden_before_lm_head`.
  Current Gemma speeds carried: `5619.6 / 92.43 / 155.69`. HQQ/k4v4 and speed
  work remain blocked. Scope: use existing `0353` artifacts/source first; for
  Moby final prefill token and one selected layer 1 expert, map BF16 vs INT4
  through matched routed input, expert/top-k/weight, BF16 W13 reference, INT4
  packed/scales metadata, prefill Marlin W13 launch, dequant/GEMM/reduction
  output, and observed W13 preactivation. Determine whether the mismatch is
  expected quantization error, wrong tensor/layout/permutation/scale indexing,
  or prefill runtime using wrong effective group-size/cache metadata, then
  confirm across all three prompts and selected experts. If evidence is
  missing, add only an off-by-default request-scoped diagnostic around prefill
  layer 1 final-token selected-expert W13/dequant. If source proves a generic
  propagation/layout bug, implement the narrow fix; otherwise stop at the
  boundary. No sampling change, fallback, route-around, guard bypass,
  protected-config edit, HQQ/k4v4, speed work, or broad architecture change.
  Closed with one request-scoped/off-by-default diagnostic scoping change only:
  the selected-expert W13 diagnostic now restricts rows to the requested final
  token rows instead of default watched rows. BF16 and INT4 both reached READY,
  then deterministic post-READY requests completed for Moby, War, and Les.
  Across `18/18` selected expert slots, routed input and prior BF16 W13 joins
  matched, while BF16-vs-INT4 W13 preactivation hashes matched `0/18`. INT4
  W13 launch/dequant metadata used effective `group_size=64`,
  `num_groups_w1=42`, `q_type=U4B8`, and no pointer/scale slots were missing.
  Same-cache normal row-major INT4 replay tracked GPU W13 output
  (`max_l2=0.0244934056`), while the alternate layout was rejected
  (`min_alt_l2_over_normal_l2=599.688`). Boundary: quantized W13 cache/dequant
  fidelity versus BF16 W13 reference, not prefill runtime group-size
  propagation, layout/permutation, missing scales, Marlin launch metadata,
  GEMM/reduction, routed input, routing, or sampling. HQQ/k4v4 and speed work
  remain blocked. Validation passed: `./dev build`, INT4/BF16 startup to READY,
  deterministic post-READY requests, artifact assertions, `bash -n dev`,
  py-compile, whitespace checks, protected-config status, source-scope audit,
  scatter guard, blocked-work checks, and cleanup.

- Opened prefill layer 1 branch/MoE output producer gate
  `20260626_0353_nemotron_nano_int4_prefill_layer1_branch_moe_output_producer`.
  Recorded the `0317` result first: existing prefill stage traces showed the
  first common BF16/INT4 final-token hidden mismatch for Moby, War, and Les at
  layer 1 `layer1_handoff_branch_last_bits`, after matching layer 1
  `pre_mlp_hidden_last` and `layer1_handoff_residual_last_bits`; all compared
  rows were finite with zero NaNs, and the next boundary is prefill layer 1
  branch/MoE output before residual handoff add. Carried `0247`: Moby's token0
  flip is upstream of LM-head projection, with matching input token hash/final
  position but divergent final residual, final norm, and
  `final_hidden_before_lm_head`; both runtimes use the same traced LM-head
  path. Carried `0147`: decode graph group size is propagated from loaded
  `WeightStore.group_size`; W13 replay uses effective `group_size=64`, has no
  missing/nonfinite scales, and max GPU/CPU replay diff is `6.568e-37`, but
  quality still fails. Current Gemma speeds carried: `5619.6 / 92.43 /
  155.69`. HQQ/k4v4 and speed work remain blocked. Scope: use existing traces
  and source first; for Moby final prefill token, compare BF16 vs INT4 through
  `pre_mlp_hidden`, router/gate logits, selected experts/top-k/weights, routed
  input, W13 output, activation/W2 input, W2 output, routed scaling, shared
  path if present, combined branch output, and residual handoff, then confirm
  on War and Les. If traces lack internals, add only an off-by-default
  request-scoped diagnostic for prefill layer 1 final-token MoE/branch
  components. No sampling change, fallback, route-around, guard bypass,
  protected-config edit, HQQ/k4v4, speed work, or broad architecture change.
  Closed with one request-scoped/off-by-default BF16 trace addition only.
  Existing traces lacked BF16 W13 preactivation and activation/W2-input
  snapshots for the selected final-token experts, so the diagnostic was scoped
  to request tracing and did not alter kernel or default runtime behavior.
  BF16 and INT4 both reached READY, then deterministic post-READY requests
  completed for Moby, War, and Les. Across all three prompts, layer 1
  `pre_mlp_hidden`, router/top-k ids, top-k weights, and routed input matched
  for `18/18` selected expert slots. The first mismatch is layer 1 W13
  preactivation/dequant output (`0/18` slots matched). Activation, W2 output,
  combined branch output, and residual handoff are downstream. All compared
  summaries were finite with zero NaNs. Current boundary: prefill layer 1 W13
  preactivation/dequant producer for final-token selected experts. Validation
  passed: `./dev build`, INT4/BF16 startup to READY, post-READY deterministic
  requests, artifact assertions, `bash -n dev`, py-compile, whitespace checks,
  protected-config status, source-scope audit, scatter-guard audit,
  blocked-work checks, and cleanup.

- Opened prefill final-hidden producer gate
  `20260626_0317_nemotron_nano_int4_prefill_final_hidden_producer`.
  Recorded the `0247` result first: Moby's token0 flip is upstream of the LM
  head; input token hash and final prefill position match, but final residual,
  layer-51 final norm, and `final_hidden_before_lm_head` already differ before
  projection, while both runtimes use the same traced LM-head path
  (`bf16_cublas_gemm_ex` / `bf16_weight_pointer`). Recorded the retained
  `0147` fix/result first: decode graph group size is propagated from loaded
  `WeightStore.group_size`; W13 replay uses effective `group_size=64`, has no
  missing/nonfinite scales, and max GPU/CPU replay diff is `6.568e-37`, but
  quality still fails. Carried Gemma speeds: `5619.6 / 92.43 / 155.69`.
  HQQ/k4v4 and speed work remain blocked. Scope: use existing `0247`
  traces/source first, walk Moby backward from `final_hidden_before_lm_head`
  through final norm input/output, final layer residual output, final layer
  components, and any last-layer attention/MoE/shared/routed path available in
  existing traces. Determine the first producer where BF16 and INT4 final-token
  hidden diverge, then confirm the same boundary on War and Les as controls. If
  traces are insufficient, add only an off-by-default request-scoped prefill
  final-token/final-layer component diagnostic scoped to the same prompts and
  final prefill position. No HQQ/k4v4, speed work, sampling change, fallback,
  route-around, guard bypass, protected-config edit, or broad architecture
  change.
  Closed with no source changes and no new diagnostic. Existing request-scoped
  prefill stage traces were sufficient. Fresh BF16 and INT4 runs both reached
  READY and deterministic post-READY requests confirmed the first common
  final-token hidden mismatch across all three prompts is layer 1
  `layer1_handoff_branch_last_bits`, after matching layer 1
  `pre_mlp_hidden_last` and `layer1_handoff_residual_last_bits`. The layer 51
  input residual is already divergent for Moby, War, and Les, so final norm,
  `final_hidden_before_lm_head`, and LM-head logits are downstream. All
  compared summaries remain finite with zero NaNs. Fresh tokens: BF16
  `1044/1044/1044`; INT4 `1321/1044/1044`. Next boundary: prefill layer 1
  branch/MoE output producer before residual handoff add. Validation passed:
  `./dev build`, INT4/BF16 startup to READY, post-READY deterministic
  requests, artifact assertions, `bash -n dev`, py-compile, whitespace checks,
  protected-config status, source-scope audit, scatter-guard audit,
  blocked-work checks, and cleanup.

- Opened Moby prefill-final-logits producer gate
  `20260626_0247_nemotron_nano_int4_moby_prefill_final_logits_producer`.
  Recorded the `0147` fix/result first: the decode graph now propagates loaded
  `WeightStore.group_size` into resident W13 launch, W13 replay uses effective
  `group_size=64`, has no missing/nonfinite scales, and max GPU/CPU replay
  diff is `6.568e-37`. Recorded the `0222` quality boundary first: Moby's
  token0 mismatch is already in prefill final logits/top-k after the fix
  (`BF16 top1=1044`, `INT4 top1=1321`, opposite token rank 2 in each top-10),
  while War and Les Miserables remain token0-matching controls and diverge at
  token1. Carried Gemma speeds: `5619.6 / 92.43 / 155.69`. HQQ/k4v4 and speed
  work remain blocked. Scope: for Moby first, use existing artifacts/source to
  trace BF16 vs INT4 through input token IDs/chat template, final prefill
  position, final residual/hidden, final norm output, LM head input, LM head
  weight path, and final logits/top-k; determine whether the token0 flip is
  already in final hidden/norm or introduced at LM-head projection. Confirm the
  relevant boundary on War and Les as controls. If tensors/summaries are
  missing, add only an off-by-default request-scoped prefill final-token
  component diagnostic. No speed work, HQQ/k4v4, sampling change, fallback,
  route-around, guard bypass, protected-config edit, or broad architecture
  change.
  Closed with no source changes and no new diagnostic. Existing request-scoped
  `debug_reference_trace`/`debug_prompt_trace` exposed the needed tensors. INT4
  and BF16 both reached READY, then deterministic post-READY requests confirmed
  the boundary. Moby input token hash and final prefill position match
  (`prompt_len=449`, final position `448`), but final residual, layer-51
  final norm, and `final_hidden_before_lm_head` already differ before the LM
  head. The LM-head path is the same traced path on both sides
  (`bf16_cublas_gemm_ex`, `bf16_weight_pointer`) and logits are finite/no-NaN,
  so the token0 flip is not introduced by sampling, decode, or a different
  LM-head weight path. Moby BF16 selects `1044`; INT4 selects `1321`, with the
  opposite token at rank 2 in both top-10 lists. War and Les Miserables remain
  token0-matching controls (`1044`/`1044`) even though final hidden/norm hashes
  differ. Next boundary: prefill final hidden producer / final-layer path
  upstream of LM head. Validation passed: `./dev build`, INT4/BF16 startup to
  READY, post-READY deterministic requests, artifact assertions, `bash -n dev`,
  py-compile, whitespace checks, protected-config status, source-scope audit,
  scatter-guard audit, blocked-work checks, and cleanup.

- Opened post-`0147` plain Nano INT4 quality boundary gate
  `20260626_0222_nemotron_nano_int4_post0147_prefill_logits_boundary`.
  Recorded the `0147` fix/result first: the effective W13 cache group-size
  propagation bug was confirmed and fixed generically by updating the Rust
  decode graph from loaded `WeightStore.group_size`; post-fix W13 replay uses
  `group_size=64`, `scale_group_count=42`, zero missing/nonfinite scales, and
  max GPU/CPU replay diff `6.568e-37`. Quality still fails after the fix:
  BF16 remains `1044 1044 1044` for all three deterministic prompts, while
  INT4 now returns `1321 13540`, `1044 6078`, and `1044 1262`; Moby now
  diverges at output token0, and War/Les Miserables diverge at token1. Carried
  Gemma speeds: `5619.6 / 92.43 / 155.69`; carried `0913` INT4 speed baseline
  remains prefill `353.6 / 344.3 / 343.2 / 342.6 / 341.9 / 312.0 tok/s`,
  internal decode `95.42 / 95.32 / 94.70 tok/s`, and network
  `166.23 / 120.56 / 103.19 tok/s`. HQQ/k4v4 and speed work remain blocked.
  Scope: use existing `0147` BF16/INT4 artifacts first to locate the new
  earliest divergence; prove whether Moby's token0 mismatch is already present
  in prefill final logits/top-k after the W13 group-size fix. If existing
  artifacts lack prefill final logits/top-k, add only an off-by-default
  request-scoped diagnostic for prefill final logits/top-k on the same three
  prompts and validate with `./dev build` plus post-READY deterministic
  BF16/INT4 requests. No HQQ/k4v4, speed optimization, sampling change,
  route-around, guard bypass, protected-config edit, fallback, or broad
  architecture change.
  Closed with no source changes and no new diagnostic. Existing `0147`
  artifacts already contained `first_token_top_k`, and fresh post-READY
  BF16/INT4 deterministic requests confirmed the same boundary. Moby's token0
  mismatch is already in prefill final logits/top-k: BF16 top1/selects `1044`
  while INT4 top1/selects `1321`; each runtime has the other's token at rank 2.
  War and Les Miserables are controls: both still select `1044` at token0 in
  BF16 and INT4, and their earliest divergence remains token1. Next boundary:
  prefill final logits producer for Moby after the W13 group-size fix; trace
  BF16/INT4 prefill component divergence ending at LM-head logits. Validation
  passed: `./dev build`, INT4 startup to READY, BF16 startup to READY,
  post-READY deterministic BF16/INT4 requests, artifact assertions,
  `bash -n dev`, py-compile via `./dev python`, whitespace checks,
  protected-config status, source-scope audit, scatter-guard audit,
  blocked-work checks, and cleanup.

- Opened plain Nano INT4 decode effective group-size propagation / resident
  W13 Marlin scale-index contract gate
  `20260626_0147_nemotron_nano_int4_decode_w13_effective_group_size_propagation`.
  Recorded the `0129` result first: the prior `0028` `1239/1440` scale
  mismatches are not bad-cache proof because the diagnostic BF16
  expected-scale lookup used requested `g128`, while the production cache
  writer/header for Nano used effective `g64`; BF16 `g64` expected scales
  match direct cached W13 scale slots for `1440/1440` sampled rows. The next
  boundary is effective group-size propagation into the decode graph /
  resident W13 Marlin scale-index contract. Carried Gemma speeds explicitly:
  `5619.6 / 92.43 / 155.69`; carried BF16 baseline remains forced token
  `1321` / `" and"`, startup `9.0/8.8 tok/s`, prefill best `273.3 tok/s`,
  decode best `86.15 tok/s`, network best `141.93 tok/s`; carried plain INT4
  `0913` speed baseline remains prefill
  `353.6 / 344.3 / 343.2 / 342.6 / 341.9 / 312.0 tok/s`, internal decode
  `95.42 / 95.32 / 94.70 tok/s`, and network
  `166.23 / 120.56 / 103.19 tok/s`. HQQ/k4v4 and speed work remain blocked.
  Scope: use source first to trace actual group size from cache
  generation/header/load into `WeightStore`, Python model setup, Rust decode
  graph config, resident W13 pointer/stride metadata, and
  `marlin_gemv_int4_v2_batched` launch. Prove whether production decode is
  indexing the `g64` cache with `g128` scale math. If confirmed, implement the
  narrow generic fix by propagating the effective cache group size into
  decode/runtime scale lookup and diagnostics. No Nano hardcode, kernel change
  unless source proves the kernel contract is wrong, fallback, route-around,
  guard bypass, protected-config edit, sampling change, HQQ/k4v4, speed
  optimization, or broad architecture change.
  Closed with the narrow generic fix retained. Source tracing confirmed the
  production bug: cache generation/header/load and `WeightStore` used effective
  `g64`, but Python could preconfigure the Rust decode graph with requested
  `g128`; resident W13 launch then passed `graph.group_size` into
  `marlin_gemv_int4_v2_batched`, whose scale lookup uses `k / group_size`.
  The fix updates an already-configured Rust decode graph from loaded
  `WeightStore.group_size` before resident expert registration and decode
  launch. No Nano hardcode, kernel behavior change, fallback, route-around,
  guard bypass, protected-config edit, sampling change, HQQ/k4v4, speed work,
  or broad architecture change was added. Post-fix INT4 and BF16 both reached
  READY, and `18/18` resident W13 replay probes now report `group_size=64`,
  `scale_group_count=42`, zero missing/nonfinite scales, and max GPU/CPU
  replay diff `6.568e-37`. Quality still fails: BF16 remains
  `1044 1044 1044` on all three deterministic prompts, while INT4 after the
  fix returns `1321 13540`, `1044 6078`, and `1044 1262`; token divergence
  changed but remains. Validation passed: `./dev build`, INT4 startup to
  READY, BF16 startup to READY, post-READY deterministic BF16/INT4 requests,
  artifact assertions, `bash -n dev`, py-compile via `./dev python`,
  whitespace checks, protected-config status, source-scope audit,
  scatter-guard audit, blocked-work checks, and cleanup.

- Opened plain Nano INT4 decode layer-1 W13 scale
  layout/generation/index proof gate
  `20260626_0129_nemotron_nano_int4_decode_layer1_w13_scale_layout_proof`.
  This is not a fix gate. Recorded the `0028` result first: the previous gate
  narrowed the boundary to decode layer1 W13 scale layout/generation/index,
  with selected experts `18`, sampled weights `1440`, INT4 scale mismatches
  versus BF16 group-derived `amax/7` scales `1239/1440`, cached signed nibble
  mismatches `261/1440`, material INT4 GPU-vs-same-cache CPU replay
  mismatches `0`, and BF16 GPU-vs-BF16 CPU dot mismatches `0`. Token0 still
  matches and token1 diverges across all three deterministic prompts. Caution:
  do not treat `1239/1440` scale mismatches as root until the diagnostic BF16
  expected-scale lookup is proven identical to the production W13 quantizer
  and decode Marlin scale lookup. Carried Gemma speeds explicitly:
  `5619.6 / 92.43 / 155.69`; carried BF16 baseline remains forced token
  `1321` / `" and"`, startup `9.0/8.8 tok/s`, prefill best `273.3 tok/s`,
  decode best `86.15 tok/s`, network best `141.93 tok/s`; carried plain INT4
  `0913` speed baseline remains prefill
  `353.6 / 344.3 / 343.2 / 342.6 / 341.9 / 312.0 tok/s`, internal decode
  `95.42 / 95.32 / 94.70 tok/s`, and network
  `166.23 / 120.56 / 103.19 tok/s`. HQQ/k4v4 and speed work remain blocked.
  Scope: use source/artifacts first; inspect W13 quantize/cache writer scale
  layout and permutation, cache reader/runtime pointer stride, resident decode
  `marlin_gemv_int4_v2_batched` scale index formula, and existing `0028`
  sample rows. Trace one exact scale from BF16 group amax through cache
  generation, stored scale slot, cache reload, decode lookup, and diagnostic
  lookup for one prompt/expert/group, then confirm across all three prompts
  and selected experts. If evidence is missing, add only an off-by-default
  request-scoped scale-layout diagnostic outside kernels. No fix, fallback,
  route-around, guard bypass, protected-config edit, sampling change, HQQ/k4v4,
  speed optimization, or broad architecture change.
  Closed with no source change and no new diagnostic. Existing source/artifacts
  were sufficient to prove the `0028` BF16 expected-scale lookup was not
  identical to the production W13 quantizer when the cache builder adjusts the
  requested group size. The cache writer stores effective `g64` for Nano
  (`intermediate=1856` is not divisible by requested `g128`), and direct cache
  inspection confirms BF16 `g64` group `amax/7` rounded to BF16 matches the
  stored W13 scale slot for `1440/1440` sampled rows. The `0028` diagnostic
  expected-scale side matches requested `g128` recomputation for `1440/1440`
  rows, while the debug/decode layout reports `group_size=128`,
  `scale_group_count=21`, and `scales_len_u16=77952 = 42 * 1856`. Therefore
  the `1239/1440` scale mismatches are not bad-cache proof; the next boundary
  is effective group-size propagation into the decode graph / resident W13
  Marlin scale-index contract. Validation passed: `./dev build`, artifact
  assertions, `bash -n dev`, py-compile via `./dev python`, whitespace checks,
  protected-config status, source-scope audit, scatter-guard audit,
  blocked-work checks, and cleanup. No HQQ/k4v4, speed work, fallback,
  route-around, guard bypass, protected-config edit, sampling change, fix, or
  broad architecture change.

- Opened plain Nano INT4 decode layer-1 W13 cache
  packing/quantization/dequant contract gate
  `20260626_0028_nemotron_nano_int4_decode_layer1_w13_cache_contract`.
  Recorded the `0913` runtime pass/speed baseline first: plain INT4 reaches
  READY, real post-READY request returned HTTP `200` with timing and no
  NaN/scatter path; speed baseline prefill
  `353.6 / 344.3 / 343.2 / 342.6 / 341.9 / 312.0 tok/s`, internal decode
  `95.42 / 95.32 / 94.70 tok/s`, network
  `166.23 / 120.56 / 103.19 tok/s`, HCS `2944/2944`, min decode free VRAM
  `10746 MB`, startup warmup `1.3s`, long prefill min free `14464 MB`, and
  VRAM margin `606 MB`. Recorded the `0947` quality failure, `1019`
  logits-before-sampling boundary, `1052` layer1 routed MoE boundary, `2212`
  routed compute boundary, `2312` W13-before-`relu2` boundary, and `2347`
  result: INT4 same-layout CPU replay matches GPU W13 GEMV/reduce `18/18`
  (`max_abs_diff=7.354e-37`) while BF16-vs-INT4 W13 preactivation differs
  `18/18`, so the current boundary is W13 cache
  packing/quantization/dequant contract versus BF16 W13 reference weights.
  Carried BF16 baseline remains forced token `1321` / `" and"`, startup
  `9.0/8.8 tok/s`, prefill best `273.3 tok/s`, decode best `86.15 tok/s`,
  network best `141.93 tok/s`; carried Nemotron `9.0/8.7 tok/s`; Gemma
  `5619.6 / 92.43 / 155.69`. Scope: use existing `2347` artifacts/source
  first; for one prompt and one selected layer1 expert map BF16 W13 reference
  weights to tensor orientation, expert ID mapping, packed row/column index,
  scale group/index/value, nibble decode, dequant value, CPU replay
  contribution, and GPU reduced W13 output; then confirm across all prompts
  and selected experts. Add only an off-by-default request-scoped/cache
  inspection diagnostic if evidence is missing. HQQ/k4v4 and speed work remain
  blocked; no kernel behavior change, sampling change, route-around, guard
  bypass, protected-config edit, fallback, forced request before readiness, or
  broad architecture change.
  Closed with one request-scoped/off-by-default cache-inspection diagnostic in
  `src/gpu_decode.rs`; no CUDA kernel behavior, sampling, routing, HCS, guard,
  default startup, fallback, or speed path changed. BF16 and INT4 both reached
  READY and `3/3` deterministic post-READY reference-test requests returned
  HTTP `200`. Across all three prompts and `18` decode-step-0 layer1 selected
  experts, top-k/expert/weight and routed input remain matched, INT4 GPU W13
  output materially matches same-cache CPU replay, and BF16 GPU W13 output
  matches BF16 CPU dot. The sampled INT4 scale slot disagrees with the BF16
  group-derived `amax/7` scale for `1239/1440` weights, and cached signed
  nibbles differ from BF16-derived quant values for `261/1440` weights. This
  rejects ordinary expected quantization error as the first explanation and
  narrows the current boundary to the W13 scale layout/generation/index
  contract for decode layer1 selected experts. Token0 still matches; token1
  still diverges in all three deterministic requests. Validation passed:
  `./dev build`, BF16/INT4 post-READY requests, artifact assertions,
  `bash -n dev`, py-compile via `./dev python`, whitespace checks,
  protected-config status, source-scope audit, scatter-guard audit,
  blocked-work checks, and cleanup.

- Opened plain Nano INT4 decode layer-1 resident W13/dequant preactivation
  producer gate
  `20260625_2347_nemotron_nano_int4_decode_layer1_w13_dequant_preactivation`.
  Recorded the `0913` runtime pass/speed baseline first: plain INT4 reaches
  READY, real post-READY request returned HTTP `200` with timing and no
  NaN/scatter path; speed baseline prefill
  `353.6 / 344.3 / 343.2 / 342.6 / 341.9 / 312.0 tok/s`, internal decode
  `95.42 / 95.32 / 94.70 tok/s`, network
  `166.23 / 120.56 / 103.19 tok/s`, HCS `2944/2944`, min decode free VRAM
  `10746 MB`, startup warmup `1.3s`, long prefill min free `14464 MB`, and
  VRAM margin `606 MB`. Recorded the `0947` quality failure and `1019`
  boundary: BF16/INT4 both reached READY; all three deterministic real-prompt
  requests returned HTTP `200` with no scatter/NaN/error path; token0 matched
  (`1044`), then token1 diverged before sampling in decode-step-1 logits/top-k
  with INT4 HCS fully cached. Recorded the `1052` boundary:
  `layer1_moe_out_after_routed_scale_before_shared`. Recorded the `2212`
  routed compute boundary: top-k/HCS/routed W13 input match BF16, but INT4
  derived `relu2` activation/W2-input differs from BF16 post-activation W2
  input. Recorded the `2312` result: BF16 derived `relu2` exactly matches
  actual BF16 post-activation W2 input, INT4 `relu2` f32 derivation matches
  trace activation for `18/18` selected expert slots, and BF16-vs-INT4 W13
  preactivation hashes differ for `18/18` slots with no nonfinite summaries.
  Carried BF16 baseline remains forced token `1321` / `" and"`, startup
  `9.0/8.8 tok/s`, prefill best `273.3 tok/s`, decode best `86.15 tok/s`,
  network best `141.93 tok/s`; carried Nemotron `9.0/8.7 tok/s`; Gemma
  `5619.6 / 92.43 / 155.69`. Scope: use existing artifacts/source first; for
  one prompt and one selected layer1 expert map matched routed input,
  expert/top-k/weight, W13 packed/scales metadata, INT4 dequant path, GEMV
  partial/reduced W13 output, and BF16 W13 reference; then confirm all three
  prompts/selected experts. Add only a request-scoped off-by-default decode
  step 1/layer 1 resident INT4 W13/dequant diagnostic if traces are
  insufficient. No kernel behavior change, speed work, HQQ/k4v4,
  route-around, guard bypass, protected-config edit, sampling change, forced
  request before readiness, fallback, or broad architecture change.
  Closed with one request-scoped/off-by-default Rust diagnostic in
  `src/gpu_decode.rs`; no CUDA kernel behavior, routing, sampling, HCS, guard,
  or default startup behavior changed. BF16 and INT4 both reached READY and
  `3/3` deterministic post-READY `/v1/internal/reference_test` requests
  returned HTTP `200`. Across all 18 decode-step-0 selected expert slots,
  top-k IDs/expert IDs/weights matched exactly, while BF16-vs-INT4 W13
  preactivation hashes differed in all 18 slots. INT4 CPU replay using host
  W13 packed/scales metadata and Marlin inverse permutations matched the GPU
  resident W13 GEMV/reduce numerically for `18/18` slots
  (`max_abs_diff=7.354e-37`), with sampled packed/scales present and finite.
  Current boundary: W13 cache packing/quantization/dequant contract versus
  BF16 W13 reference weights for layer1 selected experts. Final validation
  passed: `./dev build`, BF16/INT4 requests, artifact assertions,
  `bash -n dev`, py-compile via `./dev python`, whitespace checks,
  protected-config status, source-scope scan, scatter-guard audit,
  blocked-work checks, and cleanup. Final state: no tmux/server/reference
  process remains; GPUs idle at `15 MB / 11 MB`.

- Opened plain Nano INT4 decode layer-1 routed expert activation boundary gate
  `20260625_2312_nemotron_nano_int4_decode_layer1_activation_boundary`.
  Recorded the `0913` runtime pass/speed baseline first: plain INT4 reaches
  READY, real post-READY request returned HTTP `200` with timing and no
  NaN/scatter path; speed baseline prefill
  `353.6 / 344.3 / 343.2 / 342.6 / 341.9 / 312.0 tok/s`, internal decode
  `95.42 / 95.32 / 94.70 tok/s`, network
  `166.23 / 120.56 / 103.19 tok/s`, HCS `2944/2944`, min decode free VRAM
  `10746 MB`, startup warmup `1.3s`, long prefill min free `14464 MB`, and
  VRAM margin `606 MB`. Recorded the `0947` quality failure and `1019`
  boundary: BF16/INT4 both reached READY; all three deterministic real-prompt
  requests returned HTTP `200` with no scatter/NaN/error path; token0 matched
  (`1044`), then token1 diverged before sampling in decode-step-1 logits/top-k
  with INT4 HCS fully cached. Recorded the `1052` boundary:
  `layer1_moe_out_after_routed_scale_before_shared`. Recorded the `2212`
  result: top-k/HCS/routed W13 input match BF16, corrected INT4 W13 summaries
  are finite/nonzero, but INT4 derived `relu2` activation/W2-input differs
  from BF16 post-activation W2-input for every selected expert. Carried BF16
  baseline remains forced token `1321` / `" and"`, startup `9.0/8.8 tok/s`,
  prefill best `273.3 tok/s`, decode best `86.15 tok/s`, network best
  `141.93 tok/s`; carried Nemotron `9.0/8.7 tok/s`; Gemma
  `5619.6 / 92.43 / 155.69`. Scope: use existing artifacts/source first;
  compare BF16 W13 preactivation, BF16 post-activation W2 input, INT4 W13
  dequant output, INT4 derived activation/W2 input, and exact ungated `relu2`
  formula/stride for one prompt, then confirm all three. Add only a
  request-scoped off-by-default diagnostic if BF16 W13 preactivation or exact
  INT4 dequant values are missing. No kernel behavior change, sampling change,
  route-around, guard bypass, protected-config edit, HQQ/k4v4, speed work,
  forced request before readiness, fallback, or broad architecture change.
  Closed with one request-scoped/off-by-default decode-step-1 layer-1 selected
  expert diagnostic in `src/gpu_decode.rs`; no CUDA kernel behavior,
  routing, sampling, HCS, or guard behavior changed. BF16 and INT4 both reached
  READY and all three deterministic post-READY `/v1/internal/reference_test`
  requests returned HTTP `200`. Across all 18 selected expert slots, top-k
  expert IDs/weights matched, token0 matched (`1044`), and token1 still
  diverged. BF16 W13 preactivation was captured separately and BF16 derived
  `relu2` exactly matched the actual post-activation W2 input
  (`0` mismatches). INT4 `relu2` f32 derivation matched the existing trace
  activation for all 18 slots. BF16-vs-INT4 preactivation hashes differed in
  all 18 slots with no nonfinite operand/activation summaries, proving the
  divergence is already in resident INT4 W13/dequant preactivation before the
  `relu2` activation transform. Current boundary: resident INT4 decode W13
  output producer/dequant GEMV for layer1 selected experts. Final validation
  passed: `./dev build`, BF16/INT4 requests, artifact assertions,
  `bash -n dev`, py-compile via `./dev python`, whitespace checks,
  protected-config status, source-scope scan, scatter-guard audit, blocked-work
  checks, and cleanup. Final state: no tmux/server/reference process remains;
  GPUs idle at `15 MB / 11 MB`.

- Opened plain Nano INT4 decode layer-1 resident routed expert computation
  gate
  `20260625_2212_nemotron_nano_int4_decode_layer1_resident_moe_compute`.
  Recorded the `0913` runtime pass/speed baseline first: plain INT4 reaches
  READY, real post-READY request returned HTTP `200` with timing and no
  NaN/scatter path; speed baseline prefill
  `353.6 / 344.3 / 343.2 / 342.6 / 341.9 / 312.0 tok/s`, internal decode
  `95.42 / 95.32 / 94.70 tok/s`, network
  `166.23 / 120.56 / 103.19 tok/s`, HCS `2944/2944`, min decode free VRAM
  `10746 MB`, startup warmup `1.3s`, long prefill min free `14464 MB`, and
  VRAM margin `606 MB`. Recorded the `0947` quality failure and `1019`
  boundary: BF16/INT4 both reached READY; all three deterministic real-prompt
  requests returned HTTP `200` with no scatter/NaN/error path; token0 matched
  (`1044`), then token1 diverged before sampling in decode-step-1 logits/top-k
  with INT4 HCS fully cached. Recorded the `1052` boundary: the first active
  forward mismatch is `layer1_moe_out_after_routed_scale_before_shared`;
  layer0 materialized residual/norm/Mamba output and layer1 router
  input/gate/top-k match, and INT4 layer1 uses resident-HCS
  `relu2_w2_batched_int4` with DMA `0` and resident source hashes matching
  device buffers. Carried BF16 baseline remains forced token `1321` / `" and"`,
  startup `9.0/8.8 tok/s`, prefill best `273.3 tok/s`, decode best
  `86.15 tok/s`, network best `141.93 tok/s`; carried Nemotron
  `9.0/8.7 tok/s`; Gemma `5619.6 / 92.43 / 155.69`. Scope: map layer1
  selected experts/top-k and compare BF16 vs INT4 through routed W13 input,
  W13 output, activation, W2 input, W2 output, routed scale, shared path, and
  combined MoE output for one prompt, then confirm all three. Existing
  artifacts/source first; add only an off-by-default decode-step-1 resident
  INT4 MoE diagnostic if traces are insufficient. No HQQ/k4v4, speed work,
  sampling change, route-around, guard bypass, protected-config edit, forced
  request before readiness, fallback, or broad architecture change.
  Closed with one request-scoped debug response addition in `src/gpu_decode.rs`
  and no kernel/routing/sampling behavior change. `./dev build` passed; BF16
  and INT4 post-READY deterministic requests returned HTTP `200` for all three
  prompt cases. Layer1 top-k IDs/weights and routed W13 input match BF16 across
  all three prompts, with INT4 resident-HCS source `resident_hcs`, DMA `0`, and
  HCS hit IDs matching top-k. The corrected INT4 W13 summaries are finite and
  nonzero for all selected experts, but the derived relu2 activation/W2-input
  summaries differ from BF16 post-activation W2-input for every selected expert
  across all three prompts; W2 output, routed scale, and combined MoE output
  are downstream mismatches. Current boundary: resident INT4 layer1 W13 output
  equivalence or relu2 activation/W2-input equivalence against BF16. Final
  validation passed: build, BF16/INT4 requests, artifact assertions,
  `bash -n dev`, py-compile via `./dev python`, whitespace checks,
  protected-config status, source-scope scan, scatter-guard audit, blocked-work
  checks, and cleanup. Final state: no tmux/server/reference process remains;
  GPUs idle at `15 MB / 11 MB`.

- Opened plain Nano INT4 decode-step-1 logits-producer gate
  `20260625_1052_nemotron_nano_int4_decode_step1_logits_producer`.
  Recorded the `0913` runtime pass and speed baseline first: plain INT4
  reaches READY, real post-READY request returned HTTP `200` with timing and
  no NaN/scatter path; speed baseline prefill
  `353.6 / 344.3 / 343.2 / 342.6 / 341.9 / 312.0 tok/s`, internal decode
  `95.42 / 95.32 / 94.70 tok/s`, network
  `166.23 / 120.56 / 103.19 tok/s`, HCS `2944/2944`, min decode free VRAM
  `10746 MB`, startup warmup `1.3s`, long prefill min free `14464 MB`, and
  VRAM margin `606 MB`. Recorded the `0947` quality failure and `1019`
  boundary: BF16/INT4 both reached READY; all three deterministic real-prompt
  requests returned HTTP `200` with no scatter/NaN/error path; token0 matched
  (`1044`), then token1 diverged before greedy sampling. BF16 token1 selected
  `1044` in all cases; INT4 token1 selected `52596`, `1785`, and `1100`.
  KV/cache handoff was visible and clean, and INT4 HCS was fully cached
  (`2944/2944`, hit rate `1.0`, cold total `0`, DMA calls `0`). Carried BF16
  baseline remains forced token `1321` / `" and"`, startup `9.0/8.8 tok/s`,
  prefill best `273.3 tok/s`, decode best `86.15 tok/s`, network best
  `141.93 tok/s`; carried Nemotron `9.0/8.7 tok/s`; Gemma
  `5619.6 / 92.43 / 155.69`. Scope: identify the first layer/component where
  INT4 decode-step-1 hidden/logits diverge from BF16 after the matched prefill
  token0, starting with one real prompt and confirming across all three.
  Existing artifacts/source first; add only off-by-default decode-step-1
  component diagnostics if traces are insufficient. No sampling change,
  route-around, guard bypass, protected-config edit, HQQ/k4v4, speed
  optimization, forced request before readiness, or broad architecture change.
  Closed with no source diagnostic added. `./dev build` passed; BF16 and INT4
  both reached READY and `3/3` deterministic post-READY
  `/v1/internal/reference_test` requests returned HTTP `200`. Token0 matched
  (`1044`) in all three cases. Token1 diverged before sampling again in this
  gate run: BF16 selected `1044` for all cases; INT4 selected `52596`,
  `4268`, and `4163`. Existing decode early traces were sufficient to localize
  the first active forward mismatch: decode input embedding, materialized
  layer0 residual, layer0 normalized hidden, layer0 Mamba output, layer0
  output, layer1 router/expert input, layer1 gate logits, and layer1 top-k
  ids/weights all match; `layer1_moe_out_after_routed_scale_before_shared`
  differs for all three prompts. A layer2+ Mamba state-handoff mismatch is also
  recorded, but it is not consumed before the layer1 routed MoE branch; the raw
  layer0 pre-norm `residual_branch` mismatch is not the active boundary because
  the materialized layer0 residual matches. INT4 layer1 uses resident-HCS
  `relu2_w2_batched_int4` with DMA count `0` and HCS hit IDs matching router
  top-k; resident source hashes match device buffers. Current boundary:
  decode layer1 routed expert computation / INT4 resident batched
  W13-activation-W2 path before shared-add and downstream logits. HQQ/k4v4 and
  speed optimization remain blocked.
  Final validation passed: `./dev build`, BF16/INT4 post-READY deterministic
  requests, artifact assertions, `bash -n dev`, py-compile via `./dev python`,
  whitespace checks, protected-config status, source-scope scan, scatter-guard
  audit, blocked-work checks, and cleanup. Final state: no tmux/server/
  reference process remains; GPUs idle at `15 MB / 11 MB`.

- Opened plain Nano INT4 first decode-token divergence gate
  `20260625_1019_nemotron_nano_int4_decode_step1_divergence`. Recorded the
  `0913` runtime pass/speed baseline first: plain INT4 reaches READY,
  post-READY request returned HTTP `200` with timing and no NaN/scatter path;
  speed baseline prefill `353.6 / 344.3 / 343.2 / 342.6 / 341.9 /
  312.0 tok/s`, internal decode `95.42 / 95.32 / 94.70 tok/s`, network
  `166.23 / 120.56 / 103.19 tok/s`, HCS `2944/2944`, min decode free VRAM
  `10746 MB`, startup warmup `1.3s`, long prefill min free `14464 MB`, and
  VRAM margin `606 MB`. Recorded the `0947` quality failure: BF16 and INT4
  both reached READY and all `3/3` deterministic post-READY raw-token requests
  returned HTTP `200` with no NaN/scatter/error path, but every case matched
  BF16 output token index `0` and diverged at output token index `1`, the
  first decode-produced token; divergent selected tokens were absent from the
  opposite runtime top-10. Carried BF16 baseline remains forced token `1321` /
  `" and"`, startup `9.0/8.8 tok/s`, prefill best `273.3 tok/s`, decode best
  `86.15 tok/s`, network best `141.93 tok/s`; carried Nemotron
  `9.0/8.7 tok/s`; Gemma `5619.6 / 92.43 / 155.69`. Scope: use existing
  artifacts first, trace one prompt then confirm across all three through
  prefill final logits/top-k, token-0 selection, KV/cache handoff,
  decode-step-1 logits/top-k/logprobs, sampler input, and INT4-only
  decode/HCS expert-cache path. No speed optimization, HQQ/k4v4,
  route-around, guard bypass, protected-config edit, fallback, forced request
  before readiness, or broad architecture change.
  Closed with the first decode-token boundary identified and no source changes.
  BF16 and INT4 both reached READY through built commands and `3/3`
  post-READY debug requests returned HTTP `200`. Token0 matched in all cases
  (`1044`). Token1 diverged before greedy sampling: BF16 selected `1044` in
  all cases, while INT4 selected `52596`, `1785`, and `1100`; token1 selected
  rank was `1` in both runtimes. KV/cache handoff was visible and clean
  (`decode_kv_position_set_to_prompt_len == prompt_len`, runtime restore and
  scratch release called, reload pending false). INT4 HCS was fully cached for
  the debug runs (`2944/2944`, hit rate `1.0`, cold total `0`, DMA calls `0`),
  with layer-1 resident source hashes matching. Repeat controls showed
  additional downstream state drift after token1, left for a later gate. HQQ/k4v4
  and speed optimization remain blocked. Final validation passed: `./dev build`,
  BF16/INT4 post-READY deterministic requests, artifact assertions,
  syntax/py-compile, whitespace checks, protected-config status, no
  current-gate source edits, scatter-guard audit, log scan, and cleanup.

- Opened plain Nano INT4 quality/correctness gate
  `20260625_0947_nemotron_nano_int4_quality_correctness_vs_bf16`. Recorded
  the `0913` runtime pass and speed baseline first: plain INT4 reaches READY,
  post-READY streaming `/v1/chat/completions` returned HTTP `200`,
  non-empty text, server timing, and no NaN/scatter return. Startup baseline:
  cache `15.8 GB in 45s`, load `10s`, warmup `1.3s`, long prefill min free
  `14464 MB`, HCS soft `2944/2944`, VRAM margin `606 MB`. Speed baseline:
  internal prefill `353.6 / 344.3 / 343.2 / 342.6 / 341.9 / 312.0 tok/s`,
  internal decode `95.42 / 95.32 / 94.70 tok/s`, network
  `166.23 / 120.56 / 103.19 tok/s`. Quality caveat carried from `0913`:
  the streamed text was not semantically clean, so this gate must compare
  plain INT4 quality/correctness before HQQ/k4v4 or speed optimization. BF16
  baseline remains forced token `1321` / `" and"`, validation startup
  `9.0/8.8 tok/s`, internal prefill best `273.3 tok/s`, internal decode best
  `86.15 tok/s`, and network best `141.93 tok/s`; carried Nemotron
  `9.0/8.7 tok/s`; Gemma `5619.6 / 92.43 / 155.69`. Plan: deterministic
  post-READY `/v1/internal/reference_test` raw-token requests on canonical
  Gutenberg prompt excerpts, comparing INT4 against BF16 Krasis token IDs,
  text, top-k/logprob containment, timing, and server logs. No speed
  optimization, HQQ/k4v4, route-around, guard bypass, protected-config edit,
  fallback, forced request before readiness, or broad architecture change.
  Closed as a quality/correctness failure. BF16 and INT4 both reached READY
  and `3/3` deterministic raw-token requests returned HTTP `200` with
  `32` generated tokens and no scatter/NaN/error return. Prompt cases used
  canonical Gutenberg excerpts with `449`, `493`, and `446` input tokens.
  Token comparison against BF16: `0/3` exact matches; every case matched the
  BF16 prefill-selected first token and diverged at output token index `1`,
  the first decode-produced token. At the first divergence, the BF16 selected
  token was absent from INT4 top-10 and the INT4 selected token was absent
  from BF16 top-10 in all three cases. BF16 raw-token text was also
	  comma-heavy/degenerate, so this gate does not establish semantic BF16
	  quality; it establishes that plain INT4 quality is not acceptable and the
	  next boundary is first decode step after matched prefill. HQQ/k4v4 and speed
	  optimization remain blocked. Final validation passed: `./dev build`,
	  artifact assertions, syntax/py-compile, whitespace checks, protected-config
	  status, no current-gate source edits, scatter-guard audit, and cleanup.

- Opened plain Nano INT4 post-fix validation gate
  `20260625_0913_nemotron_nano_int4_post_fix_request_baseline`. Recorded the
  `0759` result first: default plain INT4 startup reaches READY after the
  generic W2 scale padding fix, and replay startup reaches READY with
  `scales_raw_elems=40320`, `target_scale_missing=0`, `weight_nan=0`,
  `first_nonfinite_slice=none`, and finite row0 dims `686/32`. Carried BF16
  baseline remains forced token `1321` / `" and"`, validation startup
  `9.0/8.8 tok/s`, internal prefill best `273.3 tok/s`, internal decode best
  `86.15 tok/s`, and network best `141.93 tok/s`; carried Nemotron
  `9.0/8.7 tok/s`; Gemma `5619.6 / 92.43 / 155.69`. Scope: run the same
  `tests/nemotron-nano-4-4-k6v6-a16.conf --test-endpoints` config to READY,
  send a real endpoint request only after readiness, verify generation
  succeeds, timing is reported, and no NaN/scatter path returns; if clean,
  capture plain INT4 startup/runtime speed numbers as baseline artifacts. No
  HQQ/k4v4, route-around, guard bypass, protected-config edit, fallback,
  forced-token request before readiness, or broad architecture change.
  Closed with plain INT4 request/runtime baseline captured. Default startup
  reached READY (`15.8 GB` cache built in `45s`, load `10s`, warmup `1.3s`,
  long prefill min free `14464 MB`, heatmap `2835`, HCS soft `2944/2944` in
  `0.66s`); long prefill budget `13858 MB` leaves `606 MB` against observed
  low-water, close to the configured `600 MB`. A real streaming
  `/v1/chat/completions` request was sent only after READY and returned HTTP
  `200`, non-empty text, finish `length`, and timing (`32` prompt tokens,
  prefill `69.7 tok/s`, `47` decode tokens at `92.54 tok/s`, overhead
  `602.6 ms`). Quality caveat: response text was not semantically clean, so
  this gate validates runtime generation, timing, and no NaN/scatter return,
  not answer quality. Standard benchmark completed: internal prefill
  `353.6 / 344.3 / 343.2 / 342.6 / 341.9 / 312.0 tok/s`, internal decode
  `95.42 / 95.32 / 94.70 tok/s`, network
  `166.23 / 120.56 / 103.19 tok/s`, HCS `2944/2944`, min decode free VRAM
  `10746 MB`. No HQQ/k4v4, route-around, guard bypass, protected-config edit,
  fallback, forced request before readiness, or broad architecture change.

- Opened Nano INT4 layer-1 Marlin MoE W2 scale padding/coverage fix gate
  `20260625_0759_nemotron_nano_int4_layer1_moe_w2_scale_padding_fix`.
  Recorded BF16 baseline speeds first (prefill best `273.3 tok/s`, decode
  best `86.15 tok/s`, network best `141.93 tok/s`), carried Nemotron
  `9.0/8.7 tok/s`, Gemma `5619.6 / 92.43 / 155.69`, and the `0743` result
  that production W2 final K group `1792-1855` indexes beyond the declared
  `14` scale groups for `expert_h=2688`, with the warmup dim `32`
  live-finite caveat (`0x3d2e`). Scope is source-first: inspect cache
  generation, W2 scale packing/permutation, padded `inter`, group-size math,
  and Marlin launch K selection; choose a model-generic fix between padding
  W2 scales for launched padded K groups and restricting launch/iteration to
  valid unpadded K. No Nano hardcode, route-around, fallback, guard bypass,
  protected-config edit, forced-token request before readiness, HQQ/k4v4, or
  broad architecture change.
  Closed the gate with a narrow generic fix retained. Source inspection
  rejected truncating W2 K to the floor group boundary because `inter=1856`
  is valid model data; the fix instead generates, caches, loads, stages, and
  launches W2 scales for `ceil(K/group_size)` groups. `CACHE_VERSION_MARLIN`
  is now `7` to reject stale floor-scale caches. Validation found and fixed
  one missed runtime stride in the prefill-engine descriptor: the first replay
  after the cache/write patch still reported `scales_raw_elems=37632` and
  `target_scale_missing=64`; after the descriptor fix, replay reports
  `scales_raw_elems=40320`, `target_scale_missing=0`, `weight_nan=0`, and
  `first_nonfinite_slice=none` for row0 dims `686/32` across warmup, short,
  long, and heatmap probes. Default startup reached READY (`warmup 6.2s`,
  long prefill min free `14464 MB`, HCS soft `2944/2944`); replay startup
  also reached READY (`warmup 1.4s`). No Nano hardcode, route-around,
  fallback, guard bypass, protected-config edit, forced-token request,
  HQQ/k4v4, or broad architecture change. Final validation passed:
  `./dev build`, default and replay startup paths, artifact assertions,
  `bash -n dev`, py-compile via `./dev python`, whitespace checks,
  protected-config status, source-scope/K-slice audits, scatter-guard audit,
  and cleanup.

- Opened Nano INT4 layer-1 Marlin MoE W2 scale layout/stride/index contract
  gate `20260625_0743_nemotron_nano_int4_layer1_moe_w2_scale_layout_contract`.
  Recorded BF16 baseline speeds first (prefill best `273.3 tok/s`, decode
  best `86.15 tok/s`, network best `141.93 tok/s`), carried Nemotron
  `9.0/8.7 tok/s`, Gemma `5619.6 / 92.43 / 155.69`, and the `0712` result
  that replay is off by default/outside the production Marlin MMA loop,
  default startup is unaffected, and row0 dims `686/32` for experts `48` and
  `8` hit `target_scale_missing=64`/`weight_nan=64` at final K slice
  `1792-1855`. Caveat recorded first: warmup dim `32` is live-finite
  (`0x3d2e`) while CPU replay reports NaN, so missing replay scales are not
  a production root until replay scale indexing is proven identical to
  production Marlin dequant lookup. Scope is diagnostic/provenance only: use
  existing artifacts/source first; if missing, add only an off-by-default
  Rust-side scale-layout metadata diagnostic outside CUDA templates and the
  production MMA loop. Validate with `./dev build`, default startup first,
  then replay startup. No route-around, guard bypass, protected-config edit,
  kernel behavior patch, broad architecture change, forced-token request, or
  HQQ/k4v4 work.
  Existing artifacts/source were sufficient, so no new diagnostic/source patch
  was added. Production grouped scale lookup reconciles with replay indexing:
  W2 scales contain `37632` BF16 entries (`14` groups for `expert_h=2688`,
  K `0-1791`), but the production launch processes padded final K slice
  `1792-1855` with `inter=1856`, `thread_k=64`, and `k_tiles=29`. Production
  final group index `14` puts dim `32` at the first out-of-buffer scale slot
  and dim `686` further past it; replay reports `target_scale_missing=64` for
  all four target observations at that same group. Caveat retained: warmup
  dim `32` is live-finite (`0x3d2e`), consistent with undefined adjacent-memory
  scale reads, so this proves a W2 scale layout/stride/index contract boundary
  rather than deterministic value-level replay equality or a fix. Default
  startup first emitted zero replay lines and restored the known layer-3
  scatter boundary; replay startup second emitted four replay lines and
  restored the same boundary. No route-around, guard bypass, protected-config
  edit, HQQ/k4v4, broad architecture change, CUDA-template instrumentation,
  forced-token request, or kernel behavior patch. Final validation passed:
  `./dev build`, exact default/replay INT4 startup paths, artifact assertions,
  syntax/py-compile, whitespace checks, protected-config status,
  source-scope/K-slice marker audits, and cleanup/idle snapshots.

- Opened Nano INT4 layer-1 Marlin MoE W2 standalone replay gate
  `20260625_0712_nemotron_nano_int4_layer1_moe_w2_standalone_replay`.
  Recorded BF16 baseline speeds first (prefill best `273.3 tok/s`, decode
  best `86.15 tok/s`, network best `141.93 tok/s`), carried Nemotron
  `9.0/8.7 tok/s`, Gemma `5619.6 / 92.43 / 155.69`, and the `0429` result
  that hot-loop K-slice probes were rejected/reverted while the boundary
  remains W2 scaled dequant/MMA accumulation for row0 dims `686/32`.
  Scope is diagnostic only: design the cheapest standalone/replay path outside
  the production Marlin MMA hot loop using existing `0229/0252/0429`
  artifacts, mapped expert/sorted slot/tile, W2 input, packed weights, and
  scales metadata. Any added diagnostic must be off by default and must not
  affect default startup. No kernel behavior patch, route-around, guard
  bypass, protected-config edit, shape hardcode, forced-token request before
  readiness, HQQ/k4v4 work, or broad architecture change.
  Existing artifacts/source were used first. Added only the off-by-default
  Rust-side standalone replay diagnostic
  `KRASIS_PREFILL_DIAG_MOE_W2_REPLAY[_LAYER/_ROW/_DIMS]`; no production
  Marlin MMA hot-loop instrumentation or kernel behavior change. The initial
  full-dequant replay attempt panicked in the CPU Marlin helper, so it was
  replaced with targeted exact-dim unpack/scale replay. Default startup with
  replay disabled emitted zero replay lines, completed warmup in `0.5s`, and
  returned to the known layer-3 scatter boundary. Replay-enabled startup
  emitted four exact layer-1 row0 dim `686/32` lines: W2 input is finite,
  packed weights are readable, `target_packed_missing=0`, but all targets
  reach `target_scale_missing=64`, `weight_nan=64`, and
  `first_nonfinite_slice=28` for K slice `1792-1855`. Three live outputs are
  NaN and replay-NaN; warmup dim `32` is a live-finite caveat (`0x3d2e`) while
  CPU replay is NaN. The next boundary is the actual Marlin W2 scale
  layout/stride/index contract for the final K group outside the hot MMA loop.
  No route-around, guard bypass, protected-config edit, forced-token request,
  HQQ/k4v4, or broad architecture change.
  Final validation passed: `./dev build`, exact default/replay INT4 startup
  paths, artifact assertions, `bash -n dev`, py-compile via `./dev python`,
  `git diff --check`, protected-config status, source-scope audit, K-slice
  marker absence check, and cleanup/idle snapshots.

- Opened Nano INT4 layer-1 Marlin MoE W2 exact K-slice dequant/dot
  accumulation gate
  `20260625_0429_nemotron_nano_int4_layer1_moe_w2_kslice_dequant_dot`.
  Recorded BF16 baseline speeds first (prefill best `273.3 tok/s`, decode
  best `86.15 tok/s`, network best `141.93 tok/s`), carried Nemotron
  `9.0/8.7 tok/s`, Gemma `5619.6 / 92.43 / 155.69`, and the `0252` result
  that row0 dim `686/32` targets are already `NaN` in `acc0` before BF16
  conversion/final store. Scope is diagnostic only: use existing
  template/source and artifacts first to map target lanes to K slices,
  activation fragments, packed weight nibbles, scale/zero metadata if present,
  and partial dot accumulation; if missing, add only an off-by-default
  exact-element per-K-slice diagnostic. No kernel behavior patch, Marlin
  route-around, guard bypass, protected-config edit, shape hardcode,
  forced-token request before readiness, or HQQ/k4v4 work.
  Existing `0252` artifacts/source were used first. They narrow the boundary
  to U4B8 group-scaled dequant/MMA accumulation with zeroed accumulators,
  finite W2 input, valid sorted/expert metadata, and final conversion/store
  downstream, but cannot prove the exact first K-step NaN point. Tried two
  off-by-default K-slice hot-loop diagnostic shapes; both left warmup
  CPU-active/GPU-idle with no K-slice lines, and the env-disabled compiled
  code also regressed default startup. Reverted all current-gate K-slice
  source. After revert, `./dev build` passed and default INT4 startup returned
  to the known layer-3 scatter boundary (`warmup complete in 0.5s`; short
  calibration `total_sorted=0 < m_topk=3000`). No forced-token request,
  route-around, guard bypass, protected-config edit, HQQ/k4v4, or kernel
  behavior patch. Current boundary remains exact W2 scaled dequant/MMA dot
  accumulation; next diagnostic must avoid the Marlin MMA hot loop.
  Final validation passed: artifact assertions, `bash -n dev`, py-compile via
  `./dev python`, `git diff --check`, protected-config status, source-scope
  scan confirming K-slice markers are absent, and cleanup/idle snapshots.

- Opened Nano INT4 layer-1 Marlin MoE W2 accumulator/dequant/final-write lane
  gate
  `20260625_0252_nemotron_nano_int4_layer1_moe_w2_accumulator_dequant_writeback`.
  Recorded BF16 baseline speeds first (prefill best `273.3 tok/s`, decode
  best `86.15 tok/s`, network best `141.93 tok/s`), carried Nemotron
  `9.0/8.7 tok/s`, Gemma `5619.6 / 92.43 / 155.69`, and the `0229` result
  that row0 dims `686/32` map through valid routed expert IDs, sorted metadata,
  W2 input, packed/scales pointer health, N/K tile selection, sync/copyback,
  aliasing, and final writer ownership, leaving the selected
  `thread_n=128/thread_k=64` Marlin W2 accumulator/dequant/final BF16
  writeback lane as the current boundary. Scope is diagnostic only: inspect
  K-slice accumulation, scale/dequant inputs, any FP32 reduce/`C_tmp` use,
  BF16 conversion, and final store index for row0 dims `686` and `32`. Use
  existing source/artifacts first; if missing, add only an off-by-default
  exact-element diagnostic for accumulator/partial-reduce values, relevant
  scale/dequant metadata, and final BF16 store bits. No kernel behavior
  change, Marlin route-around, guard bypass, protected-config edit, shape
  hardcode, forced-token request before readiness, or HQQ/k4v4 work.
  Existing artifacts/source were used first. Existing `0229` evidence lacked
  pre-conversion accumulator values, so added only the off-by-default
  `KRASIS_PREFILL_DIAG_MOE_W2_LANE[_LAYER/_ROW/_DIMS]` exact-element
  diagnostic. Exact INT4 startup shows row0 dim `686` in warmup and row0 dims
  `32/686` in short calibration are already `NaN` in the FP32 accumulator
  before BF16 conversion/final store (`bf16_0_bits=0x7fff`, final `0x7fff`).
  Warmup row0 dim `32` remains a finite control (`acc0=0.042361833`, final
  `0x3d2e`). Target lanes have `slice_count=1` and `owner_count=1`, so
  multi-slice `C_tmp` reduce is not the first producer. Root is now exact
  K-slice dequant/dot accumulation before `write_result` conversion/store for
  row0 dims `686/32`. No forced-token request was sent; readiness remains
  blocked downstream at the known layer-3 scatter guard. Validation passed
  through `./dev build`, exact INT4 startup diagnostic, artifact assertions,
  syntax/py-compile, whitespace checks, protected-config status,
  source-scope audit, and cleanup.

- Opened Nano INT4 layer-1 Marlin MoE W2 kernel internals gate
  `20260625_0229_nemotron_nano_int4_layer1_moe_w2_kernel_internals`.
  Recorded BF16 baseline speeds first (prefill best `273.3 tok/s`, decode
  best `86.15 tok/s`, network best `141.93 tok/s`), carried Nemotron
  `9.0/8.7 tok/s`, Gemma `5619.6 / 92.43 / 155.69`, and the `0204` result
  that layer-1 routed Marlin MoE W2 input/output-pre/`C_tmp`, sampled
  W2 packed/scales metadata, sync/copyback, and alias checks are clean before
  W2 call, while W2 output first becomes nonfinite immediately after the call
  at row0 dim `686` in warmup and row0 dim `32` in short calibration. Scope is
  diagnostic only: map those exact elements through routed expert IDs,
  sorted-token metadata, W2 launch params, N/K tile selection, packed
  weight/scales layout, accumulator path, and output writeback. Existing
  artifacts/source must be used first; any added diagnostic must be
  off-by-default and minimal at the W2 kernel boundary or vendor path. No
  kernel behavior patch, Marlin route-around, guard bypass, protected-config
  edit, shape hardcode, forced-token request before readiness, or HQQ/k4v4
  work.
  Existing artifacts/source were used first. Existing `0204` evidence lacked
  exact sorted-slot/tile/writeback mapping, so added only the off-by-default
  `KRASIS_PREFILL_DIAG_MOE_W2_KERNEL[_LAYER/_ROW/_DIMS]` mapper around the
  existing W2 call. Exact startup maps row0 dim `686` to valid W2 metadata and
  output `0x7fff` in both warmup and short calibration (`m=300`: sorted_pos
  `892`, block `13`, expert `48`, N tile `5`, final writer `74`; `m=500`:
  sorted_pos `576`, block `9`, expert `8`, N tile `5`, final writer `11`).
  Row0 dim `32` is finite at warmup (`0x3d2e`) but NaN in short calibration
  (`0x7fff`) with valid metadata. W2 launch uses the existing
  `thread_n=128/thread_k=64` generic path, not the prior N64 candidate. Root
  boundary is inside the selected Marlin W2 template accumulator/dequant/final
  BF16 writeback lane. No forced-token request was sent because readiness was
  not reached; terminal failure remains the downstream layer-3 scatter guard.
  Validation passed through `./dev build`, exact INT4 startup diagnostic,
  artifact assertions, syntax/py-compile, whitespace checks,
  protected-config status, source-scope audit, and cleanup.

- Opened Nano INT4 layer-1 Marlin MoE W2 output producer gate
  `20260625_0204_nemotron_nano_int4_layer1_moe_w2_output_producer`.
  Recorded BF16 baseline speeds first (prefill best `273.3 tok/s`, decode
  best `86.15 tok/s`, network best `141.93 tok/s`), carried Nemotron
  `9.0/8.7 tok/s`, Gemma `5619.6 / 92.43 / 155.69`, and the `0145` result
  that layer-1 router/top-k/route compaction, W1 output, activation output,
  and shared branch are finite/healthy while routed Marlin MoE W2 output
  post-call first introduces NaNs. Scope is diagnostic only: trace W2 input
  activation, W2 packed weights/scales/metadata, launch params/status, output
  buffer initialization, kernel writeback, sync/copyback, and aliasing without
  kernel patches, Marlin route-arounds, guard bypasses, protected-config edits,
  shape hardcodes, forced-token requests before readiness, or HQQ/k4v4 work.
  Existing artifacts/source were used first. Existing `0145` diagnostics
  proved W1, activation, and shared branch were finite, but lacked W2 call
  boundary operands/aliasing, so added only the off-by-default
  `KRASIS_PREFILL_DIAG_MOE_W2[_LAYER]` report. Exact INT4 startup shows W2
  input activation finite, output and `C_tmp` clean before call, row-0 routed
  W2 pointers nonzero, sampled packed/scales metadata readable, sync ok, and
  no output/input, output/`C_tmp`, output/workspace, or output/W2 pointer
  overlap. Immediately after W2 call, W2 output contains nonfinites
  (`m=300`: `32138` finite / `118` NaN, first row `0` dim `686`; `m=500`:
  `32074` finite / `181` NaN / `1` -Inf, first row `0` dim `32`). Root is
  inside layer-1 routed Marlin MoE W2 kernel computation/writeback. No
  forced-token request was sent because readiness was not reached; terminal
  failure remains the downstream layer-3 scatter guard. Validation passed
  through `./dev build`, exact INT4 startup diagnostic, artifact assertions,
  syntax/py-compile, whitespace checks, protected-config status,
  source-scope audit, and cleanup.

- Opened Nano INT4 layer-1 MoE/branch output/write boundary gate
  `20260625_0145_nemotron_nano_int4_layer1_moe_branch_output_boundary`.
  Recorded BF16 baseline speeds first (prefill best `273.3 tok/s`, decode
  best `86.15 tok/s`, network best `141.93 tok/s`), carried Nemotron
  `9.0/8.7 tok/s`, Gemma `5619.6 / 92.43 / 155.69`, and the `0134` result
  that layer-1 pre-MoE input is finite while layer1-to-layer2 `hidden_pre` is
  partially NaN/extreme. Scope is diagnostic only: trace layer-1 router/top-k,
  route compaction, Marlin MoE W1/W2 outputs, scatter/reduce, residual/add,
  output write, and handoff buffer without routing patches, guard bypasses,
  protected-config edits, shape hardcodes, forced-token requests before
  readiness, or HQQ/k4v4 work.
  Existing artifacts/source were used first. Existing gate/top-k/sorted
  reports covered layer-1 router and route compaction, but branch output
  coverage was missing, so added only the off-by-default
  `KRASIS_PREFILL_DIAG_MOE_BRANCH[_LAYER]` report. Exact INT4 startup shows
  layer-1 gate GEMM input/weights/logits finite, top-k IDs valid, and route
  compaction healthy. W1 and activation outputs are finite in sampled rows.
  The first nonfinite producer is immediately after routed Marlin MoE W2:
  warmup `m=300` has `32137` finite / `119` NaN at `w2_output_post_call`;
  short calibration `m=500` has `32074` finite / `181` NaN and `1` -Inf at
  `w2_output_post_call`. Scatter/reduce, shared-add accumulation, final BF16
  output, and layer2 handoff propagate those W2 nonfinites. Next boundary is
  layer-1 W2 output producer; no forced-token request was sent because
  readiness was not reached.
  Validation passed through `./dev build`, exact INT4 startup diagnostic,
  artifact assertions, syntax/py-compile, whitespace checks,
  protected-config status, source-scope audit, and cleanup.

- Closed the Nano INT4 layer-1 output/hidden producer gate
  `20260625_0134_nemotron_nano_int4_layer1_output_hidden_producer`.
  Recorded BF16/Gemma carried speeds and the `0119` result first, then reused
  the existing off-by-default `KRASIS_PREFILL_DIAG_ROUTER_INPUT[_LAYER]`
  report targeted at layer 1; no new source diagnostic was added. Exact
  built-command startup under
  `tests/nemotron-nano-4-4-k6v6-a16.conf --test-endpoints` shows
  layer0-to-layer1 handoff, layer-1 input norm, pass/mixer path, residual,
  and pre-MoE router input are finite for sampled rows in both warmup and
  short calibration. Existing `0108` handoff evidence shows layer1 output
  `hidden` handed to layer2 `hidden_pre` is already partially NaN/extreme
  (`m=300`: `5257` finite / `119` NaN; `m=500`: `5196` finite / `179` NaN)
  while residual is finite. Root is inside layer1 MoE/branch output/write
  after finite pre-MoE input and before layer2 handoff. No routing patch,
  guard bypass, protected-config edit, shape hardcode, HQQ/k4v4 work, new
  source diagnostic, or forced-token request. Artifacts:
  `20260625_0134_nemotron_nano_int4_layer1_output_hidden_producer_*`.
  Validation passed through `./dev build`, exact INT4 startup diagnostic,
  artifact assertions, syntax/py-compile, whitespace checks,
  protected-config status, source-scope audit, and cleanup.

- Closed the Nano INT4 layer-2 input RMSNorm operand gate
  `20260625_0119_nemotron_nano_int4_layer2_input_rmsnorm_operands`.
  Recorded BF16/Gemma carried speeds and the `0108` result first, then added
  only the off-by-default layer-scoped diagnostic
  `KRASIS_PREFILL_DIAG_INPUT_RMSNORM[_LAYER]` because existing artifacts did
  not expose the materialized fused-add RMSNorm operand, norm weights, rstd,
  or output writeback comparison. Exact built-command startup under
  `tests/nemotron-nano-4-4-k6v6-a16.conf --test-endpoints` shows layer-2
  `hidden_pre` already contains NaNs before input RMSNorm while
  `residual_pre` and input norm weights are finite. The materialized fused
  RMSNorm input retains those NaNs, making sum-squares, mean-square, and
  `rstd` NaN; expected output is all NaN and actual output is all NaN with
  zero class/bit mismatches in the sampled rows. Root is upstream of layer-2
  input RMSNorm at the layer-1 output/hidden producer or layer-2 hidden
  handoff source. No routing patch, guard bypass, protected-config edit,
  shape hardcode, HQQ/k4v4 work, or forced-token request. Artifacts:
  `20260625_0119_nemotron_nano_int4_layer2_input_rmsnorm_operands_*`.
  Validation passed through `./dev build`, exact INT4 startup diagnostic,
  artifact assertions, syntax/py-compile, whitespace checks,
  protected-config status, source-scope audit, and cleanup.

- Closed the Nano INT4 layer-2 output/hidden producer gate
  `20260625_0108_nemotron_nano_int4_layer2_output_hidden_producer`.
  Recorded BF16/Gemma carried speeds and the `0051` result first, then reused
  the existing off-by-default `KRASIS_PREFILL_DIAG_ROUTER_INPUT[_LAYER]`
  report targeted at layer 2; no new source diagnostic was added. Exact
  built-command startup under
  `tests/nemotron-nano-4-4-k6v6-a16.conf --test-endpoints` shows layer-1
  handoff `hidden` into layer 2 is already partially NaN/extreme while the
  layer-2 residual input is clean (`m=300`: hidden `5257` finite / `119` NaN;
  `m=500`: hidden `5196` finite / `179` NaN). Layer-2 `input_norm_post` is
  the first all-NaN layer-2 `hidden` stage (`finite=0`, `nan=5376`, hash
  `0xdc49ce8a54816525`), and layer-2 Mamba/output handoff preserves it. No
  routing patch, guard bypass, protected-config edit, Nemotron hardcode,
  HQQ/k4v4 work, new source diagnostic, or forced-token request. The next
  boundary is layer-2 input-norm operands or upstream layer-1 output/hidden
  producer. Artifacts:
  `20260625_0108_nemotron_nano_int4_layer2_output_hidden_producer_*`.
  Validation passed through final `./dev build`, exact INT4 startup
  diagnostic, artifact assertions, syntax/py-compile, whitespace checks,
  protected-config status, source-scope audit, and cleanup.

- Closed the Nano INT4 layer-3 router-input provenance gate
  `20260625_0051_nemotron_nano_int4_layer3_router_input_provenance`.
  Recorded BF16/Gemma carried speeds and the `0037` result first, then added
  only the minimal off-by-default layer-3 boundary diagnostic
  `KRASIS_PREFILL_DIAG_ROUTER_INPUT[_LAYER]` because existing artifacts did
  not expose the layer-3 input/handoff producer chain. Built-command startup
  under `tests/nemotron-nano-4-4-k6v6-a16.conf --test-endpoints` shows layer-2
  previous-layer handoff `hidden` is already all NaN before layer 3 starts for
  both warmup `m=300` and short calibration `m=500` (`finite=0`, `nan=5376`,
  hash `0xdc49ce8a54816525`), and layer 3 preserves that all-NaN buffer
  through input norm, pass/mixer, pre-MoE router input, gate GEMM, top-k,
  route compaction, and the existing scatter guard. No routing patch, guard
  bypass, protected-config edit, Nemotron hardcode, HQQ/k4v4 work, or
  forced-token request. The next boundary is the layer-2 output/hidden
  producer before handoff into layer 3. Artifacts:
  `20260625_0051_nemotron_nano_int4_layer3_router_input_provenance_*`.
  Validation passed through final `./dev build`, exact INT4 startup
  diagnostic, artifact assertions, syntax/py-compile, whitespace checks,
  protected-config status, source-scope audit, and cleanup.

- Closed the Nano INT4 layer-3 gate-logit producer gate
  `20260625_0037_nemotron_nano_int4_layer3_gate_logit_producer`. Recorded the
  BF16/Gemma carried speeds and the `0023` NaN-logits result first, then added
  only the minimal off-by-default layer-3 env-gated diagnostic
  `KRASIS_PREFILL_DIAG_GATE_GEMM[_LAYER]` because existing artifacts did not
  expose router input, gate weight, cuBLAS status, or immediately-after-GEMM
  logits. Built-command startup under
  `tests/nemotron-nano-4-4-k6v6-a16.conf --test-endpoints` shows warmup and
  short calibration both have router input all NaN before gate GEMM
  (`input_finite=0`, `input_nan=5376`, sample bits `0x7fff`), finite gate
  weights (`344064` finite, `0` NaN), and `cublas_status=ok`; sampled logits
  become all NaN after GEMM. Root is now upstream at the layer-3 router-input
  producer, not gate weight/cache data, cuBLAS launch/status, output
  initialization, synchronization/copyback, top-k, route compaction, scatter
  guard, or Marlin N64 lookup. No routing patch, guard bypass, protected-config
  edit, Nemotron hardcode, HQQ/k4v4 work, or forced-token request. Artifacts:
  `20260625_0037_nemotron_nano_int4_layer3_gate_logit_producer_*`.
  Validation passed through `./dev build`, exact INT4 startup diagnostic,
  artifact assertions, syntax/py-compile, whitespace checks,
  protected-config status, source-scope audit, and final cleanup.

- Closed the Nano INT4 layer-3 router/top-k producer gate
  `20260625_0023_nemotron_nano_int4_layer3_router_topk_producer`. Recorded
  the BF16/Gemma carried speeds and the `0008` result first, then added only
  the minimal layer-3 env-gated diagnostic
  `KRASIS_PREFILL_DIAG_ROUTER_TOPK[_LAYER]` because existing logs lacked
  gate-logit and immediate post-top-k samples. Built-command startup under
  `tests/nemotron-nano-4-4-k6v6-a16.conf --test-endpoints` shows warmup and
  short calibration both have `logits_finite=0`, `logits_nan=256` before
  `sigmoid_topk_kernel`; after top-k, IDs are all `-1` and weights are the
  kernel initializer `-1e30`; later normalization masks weights to `0.166667`
  while IDs remain invalid. Root is now layer-3 gate logits already being
  `NaN` before top-k, not scoring-function selection, top-k ID writeback,
  normalization, route compaction, scatter guard, or Marlin N64 lookup. No
  routing patch, guard bypass, protected-config edit, HQQ/k4v4 work, or
  forced-token request. Artifacts:
  `20260625_0023_nemotron_nano_int4_layer3_router_topk_producer_*`.

- Closed the post-N64 Nano INT4 Marlin MoE sorted-token boundary gate
  `20260625_0008_nemotron_nano_int4_marlin_moe_total_sorted_boundary`.
  Recorded the BF16 baseline first (forced token `1321` / `" and"`, startup
  `9.0/8.8 tok/s`, internal prefill best `273.3 tok/s`, internal decode best
  `86.15 tok/s`, network best `141.93 tok/s`), carried Nemotron
  `9.0/8.7 tok/s`, Gemma `5619.6 / 92.43 / 155.69`, and the retained `2329`
  N64 result. Existing logs proved the old Marlin MoE `KERNEL NOT FOUND`
  boundary was gone but did not expose route metadata, so added a minimal
  env-gated compile/report diagnostic
  `KRASIS_PREFILL_DIAG_MOE_SORTED[_LAYER]` around the existing MoE route
  compaction path. Built-command validation showed layer-3 warmup and short
  calibration both enter compaction with `topk_ids_sample=[-1,...]`,
  normalized `topk_weights_sample=[0.166667,...]`, `counts_sum=0`,
  `expected_padded=0`, `active_experts=0`, and `total_sorted=0`. Root is now
  named as invalid layer-3 router/top-k output before count/prefix/scatter,
  not Marlin output checking, workspace sizing/alignment, sorted metadata
  copyback, synchronization, or the scatter guard. No scatter-guard bypass,
  model-specific shape hardcode, protected-config edit, HQQ/k4v4 work, or
  forced-token request. Artifacts:
  `20260625_0008_nemotron_nano_int4_marlin_moe_total_sorted_boundary_*`.

- Applied the generic Nano INT4 Marlin MoE N64 fix-validation candidate from
  `2318`: appended `(thread_k=128, thread_n=64, num_threads=128)` after the
  existing MoE candidate lists, added
  `MOE_COMMON_GET_IF_M1/M234(W_TYPE, 4, 8, 128)` registry coverage, and
  updated the Rust `fused_moe_fp32_reduce_floats` workspace mirror with
  `(128,64)`. `./dev build` passed, including the new Marlin MoE N4/K8
  sidecar template, and audits confirmed the scatter guard, protected configs,
  and non-MoE paths were not changed by this gate. The exact INT4 run
  `./dev run tests/nemotron-nano-4-4-k6v6-a16.conf --test-endpoints` rebuilt
  the cache (`15.8 GB` in `43s`) and loaded it (`10s`). The previous
  `[DIAG Marlin MoE] KERNEL NOT FOUND` boundary cleared, but startup still
  failed before readiness at layer-3 MoE sorted-route metadata:
  `total_sorted=0 < m_topk=1800` during warmup and
  `total_sorted=0 < m_topk=3000` during short calibration. No forced-token
  request was sent; plain INT4 remains not requestable and HQQ/k4v4 remains
  blocked. Artifacts:
  `20260624_2329_nemotron_nano_int4_marlin_moe_n64_fix_validation_*`.

- Added a no-patch Nano INT4 Marlin MoE N-tile fix-candidate proposal gate for
  the `2307` startup boundary. The proposal is generic, not Nemotron-specific:
  append `(thread_k=128, thread_n=64, num_threads=128)` after existing MoE
  candidates, add `MOE_COMMON_GET_IF_M1/M234(W_TYPE, 4, 8, 128)` registry
  coverage for `U4B8/U8B128`, and update the Rust
  `fused_moe_fp32_reduce_floats` workspace-sizing mirror. Expected launch
  parameters for the failing shape are `thread_m_blocks=4`,
  `thread_k_blocks=8`, `thread_n_blocks=4`, `num_threads=128`,
  `group_blocks=8`, `n_tiles=29`, `k_tiles=21`, `cache_size=84480`, and
  `cache+512=84992`, with existing runtime shared-memory validation still
  acting as the hardware gate. No source patch, scatter-guard bypass,
  protected-config edit, HQQ/k4v4 work, fallback, or model-specific hardcode
  was added. Artifacts:
  `20260624_2318_nemotron_nano_int4_marlin_moe_ntile_fix_proposal_*`.
  Validation passed through `./dev build`, artifact assertions, syntax and
  whitespace checks, protected-config status, source audit, and final cleanup.

- Closed the narrow Nano INT4 startup correctness gate at the `2256` layer-3
  Marlin MoE prefill failure boundary without a production patch. Existing
  `2256` logs and source were sufficient: the root is unsupported Marlin MoE
  large-batch N-tile coverage for Nano `prob_n=1856`, not bad routed rows and
  not a real block-parameter calculation. Current large-batch candidates only
  cover `thread_n=256` and `thread_n=128`; `1856` leaves remainder `64` for
  both, while `prob_k=2688` is divisible by the candidate `thread_k=64`.
  Auto-config keeps the `{-1,-1,-1}` sentinel, which is why the diagnostic
  prints `thread_n_blocks=0` / `thread_k_blocks=0`; Marlin returns without
  launch and the existing scatter guard correctly reports
  `total_sorted=0 < m_topk=3000`. The 4-bit path selects `U4B8`, which is
  present in the local registry, so registry mismatch is not the current first
  root. No scatter-guard bypass, protected-config edit, HQQ/k4v4 work, or
  model-specific kernel hardcode was added. Artifacts:
  `20260624_2307_nemotron_nano_int4_marlin_moe_startup_boundary_*`.
  Validation passed via `./dev build`, artifact assertions, syntax/py-compile,
  whitespace checks, protected-config status, and idle runtime checks.

- Opened the first Nano INT4 correctness gate after recording the clean BF16
  baseline. The existing `tests/nemotron-nano-4-4-a16.conf` was tried first
  and failed config validation because it still uses disabled `fp8_e4m3` KV.
  The stale config was left untouched; a test-only supported-KV config
  `tests/nemotron-nano-4-4-k6v6-a16.conf` was added under `tests/`.
  That run built the GPU INT4 Marlin cache (`15.8 GB` in `44s`) and loaded it
  (`10s`), then failed before any forced-token request during the layer-3
  INT4 Marlin MoE prefill path. The concrete boundary is missing Marlin MoE
  kernel configuration for `thread_m_blocks=4`, `thread_n_blocks=0`,
  `thread_k_blocks=0`, `num_bits=4`, `group_blocks=8`, `prob_m=3000`,
  `prob_n=1856`, `prob_k=2688`, followed by
  `total_sorted=0 < m_topk=3000` in the fused MoE scatter guard. No production
  patch, fallback, protected-config edit, or INT4 performance benchmark was
  added. Artifacts:
  `20260624_2256_nemotron_nano_int4_first_correctness_gate_*`.

- Recorded the Nano BF16 acceptance/regression baseline before INT4 for the
  retained Mamba2 gated-RMSNorm `rstd` broadcast patch. The `2056` broadcast
  fix validation and `2137` post-fix propagation pass were carried first, then
  the exact Nano BF16 built-command paths were run. Clean forced-token
  validation through `./dev run tests/nemotron-nano-bf16-experts-a16.conf
  --test-endpoints` returned token `1321` / `" and"` with startup
  `9.0/8.8 tok/s`, long prefill min free `14052 MB`, decode min free
  `24626 MB`, heatmap `1673`, and HCS soft `1230`. The benchmark run through
  `./dev benchmark tests/nemotron-nano-bf16-experts-a16.conf` completed with
  timing/tracing disabled: internal prefill best `273.3 tok/s`, internal
  decode best `86.15 tok/s`, network round-trip best `141.93 tok/s`,
  HCS `1230/2944 (41.8%)`, and min decode free VRAM `844 MB`. Non-target
  RMSNorm paths were audited unchanged, protected configs stayed untouched,
  and the BF16 baseline is recorded under
  `20260624_2204_nemotron_nano_bf16_acceptance_regression_baseline_*`.
  Carried speeds: Nemotron `9.0/8.7 tok/s`; Gemma
  `5619.6 / 92.43 / 155.69`.

- Added a post-fix Nano BF16 propagation validation gate for the retained
  Mamba2 gated-RMSNorm `rstd` broadcast patch. Existing
  `2056/2048/2004/1925/1856/1843/1814` artifacts were checked first; the only
  missing post-fix operands were row `1` branch dim `1921` and the row `1`
  layer0-to-layer1 handoff, so one exact Nano `--test-endpoints` launch sent
  two request-gated diagnostics in order. Startup matched the accepted profile
  (`9.0/8.9 tok/s`, long prefill min free `14016 MB`, decode min free
  `24626 MB`, heatmap `1633`, HCS soft `1230`). Propagation passed:
  row `1` pre-out-proj dim `439` stores `0x3ae2`, branch dim `1921` is
  `0xba75`, layer0 output and layer1 input both store `0x3b85` with hash
  `0x4d52b372606d71f6`, and the forced token remains HF `1321` / `" and"`.
  Row `41` stayed fixed (`0xbbe6`, branch `0x3ada`, handoff hash
  `0x15a2af5fde91bfd1`). Non-target RMSNorm paths were audited again and
  remain unchanged. Added artifacts under
  `20260624_2137_nemotron_nano_bf16_rstd_broadcast_propagation_validation_*`.
  Carried speeds: Nemotron `9.0/8.7 tok/s`; Gemma
  `5619.6 / 92.43 / 155.69`.

- Reopened the Nano BF16 gated-RMSNorm `rstd` fix-candidate validation using
  the corrected `2048` timing baseline and reapplied only the one-per-block
  shared-memory broadcast candidate. The production change is limited to the
  Mamba2 gated group RMSNorm path and its replay mirror: one rstd per
  token/group is computed with `sqrt.approx.ftz.f32 + div.rn.f32` and
  broadcast through shared memory. Exact Nano startup completed inside the
  corrected `1925` envelope: the validated `--test-endpoints` launch took
  `347.908s` for long calibration and reached readiness `180.274s` after long
  calibration completion, with `9.0/8.9 tok/s`, long prefill min free
  `14070 MB`, decode min free `24626 MB`, heatmap `1651`, HCS soft `1230`.
  Row `1`, dim `439` now stores HF `0x3ae2`, row `41` remains `0xbbe6`, and
  the clean forced-token response flips to HF token `1321` / `" and"`. Source
  audit confirms no normal/non-target RMSNorm path was changed; existing
  diagnostic output-detail fields still report old `rsqrtf` candidate values,
  but actual stored BF16 output and forced token validate the patched
  production path. Added artifacts under
  `20260624_2056_nemotron_nano_bf16_layer0_gated_rmsnorm_rstd_broadcast_fix_validation_*`.
  Carried speeds: Nemotron `9.0/8.7 tok/s`; Gemma
  `5619.6 / 92.43 / 155.69`.

- Added a no-patch Nano BF16 performance-forensics gate for the rejected
  `2016` layer0 Mamba2 gated-RMSNorm `rstd` candidate. Existing failed
  startup logs and accepted `1925/2004` artifacts were used first. The
  timestamped logs do not prove a candidate-specific long-calibration stall:
  accepted `1925` long calibration took `347.697s` and needed `179.916s`
  after long-calibration completion to reach readiness; the first `2016`
  candidate completed long calibration in `347.742s` and stopped at heatmap
  prompt `1/6`, while the second candidate was stopped before the accepted
  long-calibration duration had elapsed. No production patch was reapplied, no
  new runtime was launched, and no diagnostic source was added. Added
  artifacts under
  `20260624_2048_nemotron_nano_bf16_layer0_gated_rmsnorm_rstd_perf_forensics_*`.
  Carried speeds: Nemotron `9.0/8.7 tok/s`; Gemma
  `5619.6 / 92.43 / 155.69`.

- Opened the Nano BF16 layer0 Mamba2 gated-RMSNorm `rstd` precision
  primitive fix-candidate gate for row `1`, dim `439`, using the `2004`
  artifacts first and recording the source-path proposal before source edits.
  The correctness proposal was narrow and model-agnostic in shape: match the
  HF/Triton `sqrt.approx.ftz.f32 + div.rn` rstd behavior only in the Mamba2
  gated-RMSNorm path, with row `41` as the fixed control and no normal
  RMSNorm/decode changes. Two built candidate shapes failed the startup safety
  gate before the row request could be sent: direct per-thread `div.rn` and a
  one-per-token/group shared-memory broadcast both remained in exact Nano long
  calibration without readiness while GPU0 was active at ~100%. The candidate
  was rejected and reverted, then the repo-local extension was rebuilt through
  `./dev build`; no current-gate production patch remains. Added artifacts
  under
  `20260624_2016_nemotron_nano_bf16_layer0_gated_rmsnorm_rstd_fix_candidate_*`.
  Carried speeds: Nemotron `9.0/8.7 tok/s`; Gemma
  `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 row `1`, layer0 Mamba2 gated-RMSNorm `rstd/output`
  producer gate for dim `439`. Existing `1925/1856/1843/1814/...` artifacts
  were checked first and were sufficient; no runtime was launched and no
  source or production math was changed. The row `1` input/gate, mean-square,
  epsilon, and Krasis adjacent-pairwise reduction summary match the HF manual
  path through `mean_square+eps=0x4338bd27`. The first operand-level split is
  the `rstd` precision primitive: HF actual Triton `_layer_norm_fwd` returns
  `rstd=0x3d96ada8` and output `0x3ae18001`, matching the diagnostic
  `sqrt.approx.ftz.f32 + div` candidate and storing `0x3ae2`; Krasis runtime
  uses `rsqrtf`/sqrt.rn-equivalent `rstd=0x3d96ada7`, output `0x3ae17fff`,
  and stores `0x3ae1`. Row `41` remains the fixed BF16-store matched control
  at `0xbbe6`. Added artifacts under
  `20260624_2004_nemotron_nano_bf16_layer0_gated_rmsnorm_rstd_producer_*`.
  Carried speeds: Nemotron `9.0/8.7 tok/s`; Gemma
  `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 row `1`, layer0 pre-out-proj dim `439` producer gate.
  Existing `1856/1843/1814/1758/1748/1736/1712/1653/1634/1606/1553/1507`
  artifacts were checked first. Exact dim `439` Krasis operands were missing,
  so one exact Nano request used existing request-gated diagnostics for layer
  `0`, rows `1,41`, dim `439`; no production math was changed. Row `1` matches
  HF at SSD output / gated-norm input (`0xbe04`), gate (`0xbf40`), gated
  product, mean-square, manual `rstd` (`0x3d96ada7`), and manual pre-store
  output (`0x3ae17fff`, BF16 candidate `0x3ae1`). The split is the HF actual
  Triton gated-RMSNorm producer, which uses `rstd 0x3d96ada8` and outputs
  `0x3ae18001`, crossing the BF16 midpoint and storing `0x3ae2`; Krasis stays
  at `0x3ae17fff` and stores `0x3ae1`. Row `41` remains the matched control at
  stored pre-out-proj `0xbbe6`. Startup accepted with `9.1/8.9 tok/s`. Added
  artifacts under
  `20260624_1925_nemotron_nano_bf16_layer0_pre_out_proj_dim439_producer_*`.
  Carried speeds: Nemotron `9.0/8.7 tok/s`; Gemma
  `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 row `1`, dim `1921` layer-0 Mamba2 branch boundary gate.
  Existing `1843/1814/1758/1748/1736/1712/1653/1634/1606/1553/1507`
  artifacts were checked first; HF had row-scoped layer0 Mamba2 internals and
  Krasis needed one exact Nano request using existing request-gated diagnostics
  for layer `0`, rows `1,41`, dim `1921`, and full pre-out-proj bits. No
  production math was changed. Row `1`, dim `1921` matches through
  input/residual, norm/mixer input, in-proj gate, raw/post-conv x, SSD output,
  gate, and selected gated-norm/pre-out-proj store. Branch output still
  differs (HF `0xba75`, Krasis `0xba74`) because the full pre-out-proj input
  row to out-proj already differs at dim `439` (HF `0x3ae2`, Krasis `0x3ae1`);
  row `41` has zero full-row pre-out-proj diffs and remains matched. Startup
  accepted with `9.0/8.8 tok/s`. Added artifacts under
  `20260624_1856_nemotron_nano_bf16_layer0_mamba2_branch_dim1921_boundary_*`.
  Carried speeds: Nemotron `9.0/8.7 tok/s`; Gemma
  `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 row `1`, dim `1921` layer-0 output producer coverage gate
  for the remaining forced-token divergence. Existing
  `1814/1758/1748/1736/1712/1653/1634/1606/1553/1507` artifacts were checked
  first; they supplied the row `1` layer0 output split and Krasis
  lhs/rhs/rounded full bits but lacked HF row `1` residual/branch operands at
  dim `1921`. One built HF/reference capture for layer0 rows `1,41`, dim
  `1921` closed the gap. No Krasis runtime was launched and no production math
  was changed. Boundary: row `1` residual/lhs matches (`0x3ba4` both), while
  layer0 Mamba2 branch output/rhs differs by one ULP (HF `0xba75`, Krasis
  `0xba74`). BF16 add/store follows local operands: HF `0x3b856000 -> 0x3b85`,
  Krasis `0x3b858000 -> 0x3b86`. Row `41` control remains matched. Added
  artifacts under
  `20260624_1843_nemotron_nano_bf16_layer0_row1_dim1921_output_producer_*`.
  Carried speeds: Nemotron `9.0/8.7 tok/s`; current Gemma speeds
  `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 row `1` layer-1 input provenance coverage gate for the
  remaining forced-token divergence. Existing
  `1758/1748/1736/1712/1653/1634/1606/1553/1507` artifacts were checked first;
  they showed row `1` full-row layer-1 input differed while selected dims
  matched, and row `41`/last-token remained a matched control. Added only
  request-gated diagnostic exposure: HF selected `layer0_output` rows now carry
  full BF16 bits, and Krasis selected `*_output_sum_selected_rows` traces carry
  full-row lhs/rhs/rounded BF16 bits. No production math was changed. One
  built HF/reference capture and one exact Nano raw-token diagnostic request
  localized the first actual differing coordinate to row `1`, dim `1921`:
  HF layer0 output `0x3b85` vs Krasis `0x3b86`, with Krasis residual lhs
  `0x3ba4` and branch rhs `0xba74`. Krasis layer0 rounded output is preserved
  into layer1 input, so handoff/index/label paths are not the culprit. Row
  `41` full-row control remains matched. Added artifacts under
  `20260624_1814_nemotron_nano_bf16_layer1_input_provenance_*`. Carried
  speeds: Nemotron `9.0/8.7 tok/s`; Gemma `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 row `1` layer-1 output producer coverage gate for the
  remaining forced-token divergence. Existing
  `1748/1736/1712/1653/1634/1606/1553/1507` artifacts were checked first; they
  had Krasis row `1/41` layer-1 selected-row internals but lacked HF row `1`
  layer-1 internals, so one built-command archived-HF forensic capture was run
  with `--diagnose-layer1-internals`, rows `1,41`, and selected dims
  `0,1,2,3,32,63,1344,2687`. No Krasis runtime was launched. The first
  boundary is row `1` full-row layer-1 input/residual before layer-1 RMSNorm or
  MoE: HF hash `0x4d52b372606d71f6` vs Krasis
  `0xbc069341c2692f4d`. Selected dims match, including dim0 `0xbb30`, so the
  first differing coordinate is outside the selected dim set. Layer `1` is MoE
  only for this boundary; row `41`/last-token remains matched through input,
  norm, routed/shared branches, combined MoE output, and rounded output. No
  production patch, fix proposal, INT4/performance work, layer-2 `in_proj`
  work, broad architecture change, or protected-config edit was made. Added
  artifacts under
  `20260624_1758_nemotron_nano_bf16_layer1_row1_output_producer_*`. Carried
  speeds: Nemotron `9.0/8.7 tok/s`; Gemma `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 row `1` layer1-to-layer2 handoff coverage gate for the
  remaining forced-token divergence. Existing
  `1736/1712/1653/1634/1606/1553/1507` artifacts were sufficient; no runtime
  was launched and no source diagnostic was added. The apparent contradiction
  with the earlier aggregate layer-1 match is scope: HF hidden summaries use
  `_tensor_last_token_summary` and select the final row/token only, so row
  `41`/last-token still matches while selected row `1` differs. HF row `1`
  layer2 input hash is `0x53c6b4a18c4a8409`, dim0 `0xbc90`; Krasis layer2
  input hash is `0x207ea7dead2a8578`, dim0 `0xbd20`. Krasis row `1` layer1
  rounded output selected dims are preserved into layer2 input, so the
  boundary is upstream of the layer1-to-layer2 handoff and not `in_proj`,
  layer2 RMSNorm, BF16 handoff store, row/token alignment, or layer index
  mapping. Added artifacts under
  `20260624_1748_nemotron_nano_bf16_layer1_layer2_row1_handoff_*`. Carried
  speeds: Nemotron `9.0/8.7 tok/s`; Gemma `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 layer-2 `in_proj` raw `dt` producer coverage gate for the
  remaining forced-token divergence. Existing `1712/1653/1634/1606/1553/1507`
  artifacts were checked first and were sufficient; no runtime was launched.
  The first operand-level split is upstream of `in_proj`: row `1` layer-2
  input/residual differs before RMSNorm and before the Mamba2 projection (HF
  hash `0x53c6b4a18c4a8409`, dim0 `0xbc90`; Krasis hash
  `0x207ea7dead2a8578`, dim0 `0xbd20`). The normalized mixer input consumed
  by `in_proj` also differs (HF hash `0x52f6d83d34487325`, dim0 `0xbddf`;
  Krasis hash `0xeff1f7135fe6d0c2`, dim0 `0xbe81`). HF row `1` raw dt head0
  is `0xc01a`; Krasis row `1` projection dim `10240` and `dt_out[1,0]` are
  both `0xc020`, proving extract preserves the projection source slot. Existing
  row `1` matmul weight/partial details are absent, but no new diagnostic was
  needed because the `in_proj` input is already mismatched. No production
  patch, fix proposal, INT4/performance work, broad architecture change, or
  protected-config edit was made. Added artifacts under
  `20260624_1736_nemotron_nano_bf16_layer2_inproj_dt_producer_coverage_*`.
  Carried speeds: Nemotron `9.0/8.7 tok/s`; Gemma
  `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 layer-2 Krasis raw `dt` source-side coverage gate for the
  remaining forced-token divergence. Existing `1653/1634/1606` artifacts were
  checked first; the prior `1653` exact Nano attempt had reached long
  calibration but did not reach readiness, so Krasis row `1` source-side raw
  `dt` was still missing. This gate reran the exact built command
  `./dev run tests/nemotron-nano-bf16-experts-a16.conf --test-endpoints` in
  tmux and waited under the recorded bounded readiness policy. The server
  reached readiness: short/long decode `9.0/8.9 tok/s`, long probe
  `39920+32`, long prefill min free `14070 MB`, decode min free `24626 MB`,
  heatmap `1649`, HCS soft `1230`, ready free about `845 MB`. Two
  request-gated row `1/41` diagnostics captured the missing Krasis operands:
  in-proj row `1` dim `10240` is `0xc020`, `dt_out[1,0]` immediately after
  extract is `0xc020`, and the row `41` token-1 scan consumer also sees
  `0xc020`. HF row `1` raw `dt` remains `0xc01a`. Boundary is therefore the
  projection output source slot before extract, not conv packing, extract
  indexing, a later cast, bias/softplus, `A*dt`, or cumsum. No production
  patch, fix proposal, INT4/performance work, broad architecture change, or
  protected-config edit was made. Added artifacts under
  `20260624_1712_nemotron_nano_bf16_layer2_raw_dt_krasis_source_coverage_*`.
  Carried speeds: Nemotron `9.0/8.7 tok/s`; Gemma
  `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 layer-2 raw `dt` producer gate for the remaining
  forced-token divergence. Existing `1634/1606/1553/1507` artifacts were
  checked first and showed the first `dA_cumsum` split at layer 2, row `41`,
  dim `0`, token `1`: HF raw `dt` `0xc01a` (`-2.40625`) versus Krasis
  `0xc020` (`-2.5`) before bias, softplus, `A*dt`, prefix/cumsum, local scan,
  or store. A built-command HF/reference capture with selected rows `1,41`
  closed the HF source-side coverage gap: HF preconv raw `dt` for row `1`,
  head `0` is `0xc01a`, matching the HF chunk-cumsum consumer raw `dt`
  `0xc01a`. `dt_bias` matches at `-1.109375`; softplus `dt`, `A*dt`, and
  `dA_cumsum` are downstream of the raw-`dt` split. An exact Nano runtime
  attempt for matching Krasis row `1/41` selected-row coverage reached long
  calibration but did not reach readiness, so no request was sent; it was
  cleaned with `./dev kill`. No production patch, fix proposal,
  INT4/performance work, broad architecture change, or protected-config edit
  was made. Added artifacts under
  `20260624_1653_nemotron_nano_bf16_layer2_raw_dt_producer_*`. Carried
  speeds: Nemotron `9.0/8.7 tok/s`; Gemma `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 layer-2 SSD scan-internals coverage gate for the remaining
  forced-token divergence. Existing `1606/1553/1507/1434/1421/1351/1335`
  artifacts were checked first; Krasis already had selected per-token
  local-scan detail, but HF only had aggregate layer-2 `dA_cumsum` /
  chunk-scan tensors. Ran one built-command HF/reference capture with
  `KRASIS_ALLOW_ARCHIVED_HF_REFERENCE=1 ./dev generate-reference nemotron-nano`
  and `--diagnose-layer2-internals` for selected row `41` and dims
  `0,1,2,3,336,672,1344,2016,2684,2685,2686,2687`; status `0`, selected HF
  token `1321` / `" and"`. The first downstream local-scan consumer mismatch
  is decay at token `0`, while the first `dA_cumsum` producer split is raw
  `dt` at token `1`, dim `0`: HF raw `dt` `0xc01a` (`-2.40625`) vs Krasis
  `0xc020` (`-2.5`), leading immediately to split softplus `dt`, `A*dt`, and
  `dA_cumsum`. `A_val` matches every selected comparable head. No production
  source patch, fix proposal, INT4/performance work, or Krasis forced-token
  rerun was made. Added artifacts under
  `20260624_1634_nemotron_nano_bf16_layer2_ssd_scan_internal_coverage_*`.
  Carried speeds: Nemotron `9.0/8.7 tok/s`; Gemma
  `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 layer-2 Mamba mixer internal-boundary gate for the
  remaining forced-token divergence. Existing `1553/1507/1434/1421/1351/1335`
  artifacts were checked first; HF layer-2 internals were available, but the
  existing Krasis forced-token response lacked layer-2 Mamba2 internals. Ran
  one request-gated diagnostic against the exact Nano command
  `./dev run tests/nemotron-nano-bf16-experts-a16.conf --test-endpoints` with
  `debug_prefill_device_trace_layer=2` and the forced-token payload. Startup
  accepted: short/long decode `9.0/8.8 tok/s`, long probe `39920+32`, long
  prefill min free `14070 MB`, decode min free `24626 MB`, heatmap `1643`,
  HCS soft `1230`, ready free about `843 MB`. Forced token remains Krasis
  `1294` / `" in"` with HF `1321` / `" and"` at rank 3. Layer-2 input,
  norm/mixer input, in-proj aggregate, raw `x/B/C/dt`, post-conv `x/B/C`,
  `dt+bias`, softplus `dt`, and aggregate `A*dt` match. The first available
  internal producer split is SSD `dA_cumsum` / chunk-scan input: HF mean/L2
  `-27.052570343017578` / `1232.1943359375` versus Krasis
  `-27.33268165588379` / `1250.3814697265625`; SSD output is downstream
  mismatched. No production patch, INT4/performance work, or fix proposal was
  made because HF per-token layer-2 chunk-scan `dA/decay/CB/local-term`
  partial sums remain the next coverage gap. Added artifacts under
  `20260624_1606_nemotron_nano_bf16_layer2_mixer_internal_boundary_*`.
  Carried speeds: Nemotron `9.0/8.7 tok/s`; Gemma
  `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 layer-2 producer-boundary gate for the remaining forced
  token divergence after the row `29` C@B order fix. The accepted local
  corrections stayed in place, BF16-only scope was preserved, and no production
  math, INT4, or performance work was changed. Existing
  `1538/1507/1434/1421/1351/1335` artifacts were checked first; they located
  the previous boundary at HF `layer_2_output` versus Krasis
  `layer_3_input_residual_last`, but did not include HF layer-2 internals. Ran
  a forensic HF/reference capture only through the built command path with
  `KRASIS_ALLOW_ARCHIVED_HF_REFERENCE=1 ./dev generate-reference ...` and
  `--diagnose-layer2-internals` for the exact `1507` forced-token payload.
  The capture selected HF token `1321` / `" and"` and emitted `77`
  pre-generate layer-2 internal summaries. HF `layer2_input` matches Krasis
  layer-2 `layer_input_residual_last` by BF16 row hash
  `0x6d31f9291cd91ac0`; HF `layer2_norm_output` matches Krasis
  `post_input_norm_last` by BF16 row hash `0x44c780d740c0ddfe`. The first
  layer-2 internal split is HF `layer2_mixer_output` versus Krasis
  `mixer_out_last`: HF row hash `0x43d64de1c571fb83`, L2
  `0.33816179633140564`; Krasis row hash `0xd85561ac318a5a56`, L2
  `0.3384962783340771`. Layer 2 is a Mamba block, so MLP/MoE is not active
  and residual handoff is downstream. Added artifacts under
  `20260624_1553_nemotron_nano_bf16_layer2_producer_boundary_*`. Carried
  speeds: Nemotron `9.0/8.7 tok/s`; Gemma `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 remaining forced-token divergence gate after the row `29`
  token `24` C@B order fix. The accepted local corrections stayed in place,
  BF16-only scope was preserved, and no production math, INT4, or performance
  work was changed. Existing `1507/1434/1421/1351/1335` artifacts were checked
  first; they contained full Krasis forced-token snapshots and HF final logits
  plus selected layer-0 internals, but not HF full hidden summaries. Ran a
  forensic HF/reference capture only through the built command path with
  `KRASIS_ALLOW_ARCHIVED_HF_REFERENCE=1 ./dev generate-reference ...`
  and `--diagnose-hidden-summaries` for the same forced-token payload. HF still
  selects `1321` / `" and"` while Krasis selects `1294` / `" in"` with HF at
  rank 3. Embedding, HF layer-0 output to Krasis layer-1 input, and HF layer-1
  output to Krasis layer-2 input match at aggregate hidden-summary level. The
  first comparable mismatch is HF `layer_2_output` versus Krasis
  `layer_3_input_residual_last`: L2 `1.1122372150421143` versus
  `1.1131692122269443`, mean delta `5.6411497228379644e-06`, max delta
  `0.0009765625`. The `1351` to `1507` Krasis patch-effect first diff at layer
  4 `mixer_out_last` is recorded separately and is not an HF boundary. The
  `1507` control request ordering caveat is recorded; future runtime validation
  for this gate must use exact requested order. Added artifacts under
  `20260624_1538_nemotron_nano_bf16_post_cb_fix_forced_token_next_mismatch_*`.
  Carried speeds: Nemotron `9.0/8.7 tok/s`; Gemma
  `5619.6 / 92.43 / 155.69`.

- Applied the Nano BF16 row `29`, token `24`, dim `1550` `C@B`
  accumulation-order production-fix candidate after recording the source-path
  proposal first. The proposal compared forward state-order, reverse
  state-order, pairwise, and higher-precision reductions over the previously
  captured `128` `C/B/product` operands; only reverse state-order reproduced
  HF `C@B=0xbba67080` and coefficient `0xbd50`, so the patch added a
  runtime-`state_size` `mamba2_ssd_cb_dot_reverse` helper and applied it only
  to the Mamba2 SSD `C@B` accumulation path with clean/trace/replay mirrors.
  Exact Nano startup accepted under `./dev run
  tests/nemotron-nano-bf16-experts-a16.conf --test-endpoints`: short/long
  decode `9.0/8.9 tok/s`, long probe `39920+32`, long prefill min free
  `14070 MB`, decode min free `24626 MB`, heatmap `1630`, HCS soft `1230`,
  ready free `845 MB`. Row `29` tracked SSD BF16 store is fixed
  (`0x3be2` -> `0x3be3`, HF `0x3be3`) with live local scan
  `0x3b97a7c0`, while exact FP32 pre-store still differs below the BF16
  rounding boundary (`0x3be2dac0` vs HF `0x3be2da00`). Rows `5`, `7`, and
  `31` remain matched fixed controls. Clean forced-token correctness is still
  not fixed: Krasis selects `1294` / `" in"`, HF expects `1321` / `" and"`
  at rank 3. No broad architecture change, protected-config edit,
  INT4/performance work, or model-specific hardcode was added. Carried speeds:
  Nemotron `9.0/8.7 tok/s`; Gemma `5619.6 / 92.43 / 155.69`. Added artifacts
  under
  `20260624_1507_nemotron_nano_bf16_row29_token24_cb_order_fix_candidate_*`.

- Added a Nano BF16 row `29`, token `24`, dim `1550` `C@B` dot-product
  producer gate. Existing artifacts lacked per-state `C/B/product` evidence,
  so the gate added only request-gated local-scan term diagnostics selected by
  `debug_prefill_device_trace_local_scan_token`; clean SSD production math was
  not patched. Exact Nano startup accepted under `./dev run
  tests/nemotron-nano-bf16-experts-a16.conf --test-endpoints`: short/long
  decode `9.0/9.0 tok/s`, long probe `39920+32`, long prefill
  post-alloc/min free `14070/14016 MB`, decode min free `24626 MB`, heatmap
  `1632`, HCS soft `1230`, ready free `845 MB`. Row `29` token `24` captured
  all `128` state terms. The same captured `C/B/product` operands reduce to
  Krasis `C@B=0xbba66f40` in forward state order and to HF
  `C@B=0xbba67080` in reverse state order; the reverse aggregate with current
  Krasis scale flips the BF16 coefficient from `0xbd4f` to HF `0xbd50`. Rows
  `5`, `7`, and `31` remain matched fixed controls. No production fix was
  proposed; the boundary is C@B accumulation order/precision before the BF16
  coefficient cast. Added artifacts under
  `20260624_1434_nemotron_nano_bf16_row29_token24_cb_producer_*`.

- Added a Nano BF16 row `29`, dim `1550` post-exp downstream split gate from
  existing `1351/1335/1218/1127/1059/1048/0805` artifacts only. The accepted
  local corrections stayed in place, and no runtime, new tracing, production
  patch, protected-config edit, INT4 work, or performance work was done.
  Row `29` now matches HF at raw `A_log` and `A_val`; all 30 `dt` and `A*dt`
  entries also match after the exp fix. The remaining propagated split is
  local-scan token `24`: HF `C@B=0xbba67080`, Krasis `C@B=0xbba66f40`,
  producing `BF16((C@B)*decay*dt)` `0xbd50` vs `0xbd4f`, term
  `0x3bc09000` vs `0x3bbfa300`, local scan `0x3b97a7c0` vs `0x3b96bac0`,
  and pre-store `0x3be2da00` vs `0x3be1edc0`. Counterfactuals show Krasis
  scale with HF `C@B` still rounds to HF `0xbd50`, while Krasis `C@B` with HF
  scale rounds to Krasis `0xbd4f`; the next boundary is therefore the token
  `24` `C@B` dot-product producer. Rows `5`, `7`, and `31` remain fixed
  controls at tracked BF16 stores.

- Applied the Nano BF16 row `29`, dim `1550` exp-output production-fix
  candidate after recording a source-path proposal first. The proposal compared
  Krasis CUDA fast-math `__expf` with HF/PyTorch CUDA `exp` for
  `A_log -> A_val`, audited the `--use_fast_math` build path, and selected the
  minimal model-agnostic candidate: use explicit libdevice `__nv_expf` only for
  Mamba2 SSD `A_log -> A_val`, with clean and trace paths mirrored. Exact Nano
  startup accepted under `./dev run
  tests/nemotron-nano-bf16-experts-a16.conf --test-endpoints`: short/long
  decode `9.0/8.9 tok/s`, long probe `39920+32`, long prefill min free
  `14070 MB`, decode min free `24626 MB`, heatmap `1618`, HCS soft `1230`;
  observed runtime free after validation requests was GPU0 `841 MB`. Row `29`
  `A_val` now matches HF (`0xbba18860`), and rows `5`, `7`, and `31` remain
  matched at tracked BF16 stores. This is not a full correctness fix: row `29`
  SSD store still differs (`0x3be2` vs HF `0x3be3`), and the clean
  forced-token check still selects `1044` / `","` while HF expects `1321` /
  `" and"` (rank 3). Added artifacts under
  `20260624_1351_nemotron_nano_bf16_row29_exp_fix_candidate_*`.

- Added a Nano BF16 HF/reference raw `A_log` coverage-confirmation gate before
  any production exp fix proposal. No production math, INT4 path, performance
  path, or protected config was changed. The first built-command forensic run
  used `KRASIS_ALLOW_ARCHIVED_HF_REFERENCE=1 ./dev generate-reference
  nemotron-nano ... --diagnostic-only` and confirmed command scope, but the
  diagnostic-only manifest intentionally omitted layer internals. A second
  built-command full reference capture with the same raw input, eager
  attention, no cache, layer-0 internals, dims `1550,801,2458,1857`, and rows
  `29,5,7,31` serialized the missing HF/reference raw `A_log` fields. Row
  `29`, dim `1550`: HF raw `A_log=0xc0aa0000` exactly matches Krasis loaded
  `A_log=0xc0aa0000`; HF CUDA `-exp(A_log)=0xbba18860`, while Krasis
  `__expf` gives post-exp `A_val=0xbba18862`. Rows `7` and `31` match both
  raw load and exp output; row `5` matches raw load and retains the known
  nonpropagating exp-bit difference. This closes the coverage gap so the next
  fix proposal can rely on direct HF/reference load-versus-exp evidence.
  Carried speeds: Nemotron `9.0/8.7 tok/s`; Gemma
  `5619.6 / 92.43 / 155.69`. Added artifacts under
  `20260624_1335_nemotron_nano_bf16_hf_alog_coverage_*`.

- Added a Nano BF16 SSD row `29`, dim `1550` A-log load versus exp
  implementation gate. Existing artifacts lacked raw `A_log`, so the gate
  added only request-gated SSD output component diagnostic fields for selected
  rows to expose loaded `A_log`, `exp(A_log)`, and post-exp `A_val`; clean SSD
  production math was not patched. Exact Nano startup accepted under
  `./dev run tests/nemotron-nano-bf16-experts-a16.conf --test-endpoints`:
  short/long decode `9.0/8.8 tok/s`, long probe `39920+32`, long calibration
  `14070/14032 MB` prefill post-alloc/min free, decode min free `24626 MB`,
  heatmap `1655`, HCS `1230`, ready free `845 MB`. Row `29` loaded
  `A_log=0xc0aa0000` (`-5.3125`); existing HF/mamba_ssm post-exp `A_val` and
  PyTorch CUDA `-exp(A_log)` are `0xbba18860`, while Krasis `__expf` produces
  `0xbba18862`, carrying into the known `dA_target`/store split. Fixed
  controls rows `5`/`7`/`31` remain matched at tracked BF16 stores. No
  production fix was proposed; boundary is the exp implementation output
  before cumsum/store. Added artifacts under
  `20260624_1229_nemotron_nano_bf16_ssd_row29_alog_exp_boundary_*`.

- Added a Nano BF16 SSD row `29`, dim `1550` upstream `A_val` producer gate
  from existing `1127/1112/1059/1048/0805/1003` artifacts only. Both accepted
  local corrections and the SSD runtime chunk-local `dA` prefix scan stayed in
  place; no runtime, new tracing, production patch, protected-config edit,
  INT4 work, or performance work was done. Row `29` remains split because the
  input to cumsum is already different: HF post-exp `A_val=0xbba18860` while
  Krasis post-exp `A_val=0xbba18862`; raw dt, dt bias, and softplus dt match
  for all 30 row-29 tokens. The first `A*dt` split is token `0`
  (`0xbdac9333` HF versus `0xbdac9335` Krasis). Replaying the accepted prefix
  scan over HF recorded increments yields final `0xc01f3325`, while replaying
  it over Krasis `A*dt` yields `0xc01f3327`, matching the post-`1127` row-level
  target. SSD store remains downstream and rounds correctly from the differing
  pre-store values (`0x3be3` HF versus `0x3be2` Krasis). Fixed controls rows
  `5`/`7`/`31` remain matched. Existing artifacts do not include raw `A_log`
  bits, so splitting A-log load versus exp implementation is a separate
  follow-up gate before any fix proposal. Carried speeds: Nemotron
  `9.0/8.7 tok/s`; Gemma `5619.6 / 92.43 / 155.69`. Added artifacts under
  `20260624_1218_nemotron_nano_bf16_ssd_row29_aval_producer_*`.

- Applied the Nano BF16 SSD `dA_target` cumsum production-fix candidate after
  recording the source-path proposal and control evidence first. The clean
  Mamba2 SSD prefill path now builds a runtime chunk-local HF-style inclusive
  prefix scan for `dA` in shared memory, with the trace mirror, launch
  shared-memory sizing, and end-of-timestep shared-prefix synchronization guard
  kept consistent; no protected config, INT4, performance benchmark, new
  tracing stage, or hardcoded model/GPU value was added. Exact Nano startup was
  accepted under `./dev run
  tests/nemotron-nano-bf16-experts-a16.conf --test-endpoints`: short/long
  decode `9.0/8.9 tok/s`, long probe `39920+32`, long prefill
  post-alloc/min free `14070/14070 MB`, decode min free `24626 MB`, heatmap
  `1633`, HCS `1230`. Row `7`, dim `2458`, is fixed at the SSD producer:
  final `dA_target` moved from `0xc08edd57` to HF `0xc08edd58`, pre-store from
  `0x3d37807d` to HF `0x3d377d39`, and BF16 store from `0x3d38` to HF
  `0x3d37`. Rows `5` and `31` remain fixed controls. Row `29` remains split
  as predicted by the recorded upstream `A_val` caveat (`dA_target=0xc01f3327`
  vs HF `0xc01f3325`, store `0x3be2` vs HF `0x3be3`). The clean
  forced-token decision is still wrong: Krasis remains `1294` / `" in"` while
  recorded HF expects `1321` / `" and"`. Added artifacts under
  `20260624_1127_nemotron_nano_bf16_ssd_da_cumsum_fix_candidate_*`.

- Added a Nano BF16 SSD `dA_target` cumsum-producer gate from the post-`1059`
  state. Both accepted local corrections remain in place (SSD self-sample
  identity and Mamba2 gated RMSNorm adjacent-pairwise reduction), and the gate
  reduced existing `1059/1048/0805/1003` artifacts only; no runtime, source
  patch, new tracing, protected-config edit, INT4 work, or performance work was
  done. Row `7`, dim `2458`, first splits in the target cumsum at token `4`:
  HF Triton `tl.cumsum(dA, axis=1)` produces final `dA_target=0xc08edd58`,
  while Krasis scalar row-order fused `dA_target += A_val * dt` produces
  `0xc08edd57`. At consumer token `u=1`, running `dA` still matches
  (`0xbf97a761`), so the final-target difference is upstream of decay and moves
  the coefficient from HF `0x3d5c` to Krasis `0x3d5d`. Row `29` remains a same
  coefficient-boundary control; rows `5`/`31` remain fixed at propagated BF16
  store. Added artifacts under
  `20260624_1112_nemotron_nano_bf16_ssd_da_cumsum_producer_*`.

- Added a Nano BF16 SSD local-scan producer gate from the post-`1048` state.
  Both accepted local corrections remain in place (SSD self-sample identity and
  Mamba2 gated RMSNorm adjacent-pairwise reduction), and the gate reduced
  existing `1048/0805/1003/0932/0906` artifacts only; no runtime, source
  patch, new tracing, protected-config edit, INT4 work, or performance work was
  done. Row `7`, dim `2458`, splits at local-scan token `u=1`: HF has
  `dA_target=0xc08edd58`, `cb*scale=0x3d5c7fff`, and BF16 coefficient
  `0x3d5c`, while Krasis has `dA_target=0xc08edd57`, `cb*scale=0x3d5c8005`,
  and BF16 coefficient `0x3d5d`. The resulting term delta
  `+3.11434268951416e-06` accounts for the local-scan/pre-store split. Row
  `29` shows the same coefficient-rounding boundary at token `24`; row `32`
  shows a smaller local-scan accumulation boundary; fixed controls `5` and `31`
  still match at BF16 store. Added artifacts under
  `20260624_1059_nemotron_nano_bf16_ssd_local_scan_producer_*`.

- Added a Nano BF16 SSD output/store producer gate from the post-`1003` state.
  Both accepted local corrections remain in place (SSD self-sample identity and
  Mamba2 gated RMSNorm adjacent-pairwise reduction), and the gate reduced
  existing `1039/1003/0932` plus HF `0805` artifacts only; no runtime, source
  patch, new tracing, protected-config edit, INT4 work, or performance work was
  done. Existing HF/reference output already exposes actual Triton SSD
  pre-store/component evidence for rows `5`/`7`/`29`/`31`/`32`. Row `7`, dim
  `2458`, splits before the output store: HF pre-store `0x3d377d39` rounds to
  `0x3d37`, while Krasis pre-store `0x3d37807d` rounds/stores/reads back as
  `0x3d38`. `D*x` and prior chunk state match exactly; local scan differs by
  `+3.1141098588705063e-06`. Rows `29` and `32` show the same local-scan
  boundary, while fixed controls `5` and `31` still match. Added artifacts
  under `20260624_1048_nemotron_nano_bf16_ssd_output_store_producer_*`.

- Added a Nano BF16 post-`1003` next-divergence gate. Both accepted local
  corrections remain in place (SSD self-sample identity and Mamba2 gated
  RMSNorm adjacent-pairwise reduction), and the gate reduced existing
  `1003/0932/0906` artifacts only; no runtime, source patch, new tracing,
  protected-config edit, INT4 work, or performance work was done. Row `5`
  remains fixed at BF16 store and row `31` remains matched. The next earliest
  propagated divergence is row `7`, dim `2458`, layer-0
  `mamba2_gated_group_rmsnorm`: HF norm input `0x3d37` versus Krasis
  `0x3d38`. Existing traces prove the value is produced upstream by the SSD
  output store (`y=0x3d37807d` rounds/stores as BF16 `0x3d38`, then gated
  RMSNorm reads `0x3d38`); row `7` gate bits match HF (`0x4040`). Rows `29`
  and `32` show the same source boundary. Added artifacts under
  `20260624_1039_nemotron_nano_bf16_post_rmsnorm_fix_next_divergence_*`.

- Applied the Nano BF16 layer-0 gated RMSNorm reduction production-fix
  candidate after recording a source-path proposal. The clean Mamba2 gated
  RMSNorm path now uses a model-agnostic adjacent-pairwise FP32 shared-memory
  reduction over group terms, with trace/replay mirrors kept consistent; no
  protected config, INT4, or performance work was done. Exact Nano startup was
  accepted under `./dev run tests/nemotron-nano-bf16-experts-a16.conf
  --test-endpoints`: short/long decode `9.0/8.8 tok/s`, long probe
  `39920+32`, long prefill post-alloc/min free `14070/14070 MB`, decode min
  free `24626 MB`, heatmap `1665`, HCS `1230`, ready free `845 MB`. Priority
  row `5`, dim `801`, group `1` is fixed at the tracked BF16 store:
  `mean_square+eps` moved from `0x43afbcae` to HF `0x43afbcad`, and store
  moved from `0x3a0f` to HF `0x3a10`. Row `31` still matches at BF16 store;
  rows `7`/`29`/`32` remain propagated mismatches from earlier norm-input
  splits. The clean forced-token decision is still not fixed: Krasis now
  selects `1294` / `" in"` while recorded HF expects `1321` / `" and"`. Added
  artifacts under
  `20260624_1003_nemotron_nano_bf16_l0_gated_rmsnorm_reduction_fix_candidate_*`.

- Added a Nano BF16 layer-0 gated RMSNorm reduction-trace gate. The accepted
  SSD self-sample identity patch remains in place, the work stayed BF16-only,
  and the new capture is request-gated diagnostic metadata for selected
  `mamba2_gated_group_rmsnorm` rows/groups only; clean production RMSNorm math
  was not changed. Exact Nano startup was accepted under
  `./dev run tests/nemotron-nano-bf16-experts-a16.conf --test-endpoints`:
  short/long decode `9.0/8.8 tok/s`, long probe `39920+32`, long prefill
  post-alloc/min free `14070/14070 MB`, decode min free `24626 MB`, heatmap
  `1652`, HCS `1230`, ready free `845 MB`. Row `5`, dim `801`, group `1`
  captured `512` squared gated terms, `511` reduction steps, and one summary.
  The captured CUDA tree reproduces Krasis `mean_square+eps=0x43afbcae`;
  existing HF/`mamba_ssm` aggregate remains `0x43afbcad`. Offline reductions
  from the captured terms produce CUDA tree `0x43afbcae`,
  sequential/reverse FP32 `0x43afbcaa`, and double-sum-then-FP32
  `0x43afbcad`. Added artifacts under
  `20260624_0932_nemotron_nano_bf16_l0_gated_rmsnorm_reduction_trace_*`.

- Added a Nano BF16 layer-0 gated RMSNorm row-5 mean-square gate. The accepted
  SSD self-sample identity patch remains in place, and the gate reduced
  existing `0842/0906/0805` artifacts only; no runtime, source edit, new
  tracing, production patch, protected-config edit, or INT4/performance work
  was done. Row `5`, dim `801` maps to group `1` (`512..1023`) with
  `group_size=512`, and selected operands match HF (`norm_input=0xbd62`,
  `gate=0xbf2a`, `silu=0xbe671531`, `gated=0x3c4c00b5`,
  `weight=0x3f530000`). The first split is the group mean-square aggregate:
  HF actual `mamba_ssm` Triton producer has `mean_square+eps=0x43afbcad`;
  Krasis clean CUDA has `0x43afbcae`. Epsilon does not change either bit
  pattern. The resulting pre-store straddles the BF16 midpoint:
  HF actual `0x3a0f8001` stores `0x3a10`, while Krasis `0x3a0f7fff` stores
  `0x3a0f`. No production fix candidate is proposed from this gate. Added
  artifacts under
  `20260624_0921_nemotron_nano_bf16_l0_gated_rmsnorm_row5_mean_square_*`.

- Added a Nano BF16 post-SSD-fix next-divergence gate from the accepted `0842`
  state. The self-sample identity patch remains in place, and the gate reduced
  existing post-fix Krasis row responses against archived HF row evidence only;
  no runtime, source edit, new tracing, production patch, protected-config
  edit, or INT4/performance work was done. The clean row39+row40 forced-token
  path still diverges (`HF 1321` / `" and"`, Krasis `1044` / `","`). Row
  `31`, dim `1857` now matches HF at the propagated layer-0 pre-out-proj BF16
  store (`0xb9a0`). The next earliest proven propagated row-level split is row
  `5`, dim `801`, layer-0 Mamba2 gated RMSNorm: norm input/gate/SiLU/gated
  product match HF, then `mean_square+eps` differs (`0x43afbcad` HF vs
  `0x43afbcae` Krasis), producing `rstd` `0x3d5a7b19` vs `0x3d5a7b16`, FP32
  pre-store `0x3a0f8001` vs `0x3a0f7fff`, and BF16 store `0x3a10` vs
  `0x3a0f`. Added artifacts under
  `20260624_0906_nemotron_nano_bf16_post_ssd_fix_next_divergence_*`.

- Applied and validated the Nano BF16 SSD self-sample decay production fix
  candidate. The clean SSD local-scan path and its trace mirror now use the
  mathematical identity `decay=1.0f` for the final self-sample `u == t`, while
  all earlier samples keep the existing `exp(fmin(dA_target - dA_running, 0))`
  expression. No new tracing was added. Exact Nano startup was accepted under
  `./dev run tests/nemotron-nano-bf16-experts-a16.conf --test-endpoints`:
  short/long decode `9.0/8.7 tok/s`, long probe `39920+32`, long prefill
  `14070/14070 MB`, decode min free `24626 MB`, heatmap `1634`, HCS `1230`.
  Row `31`, dim `1857` moved from live scan/y/coefficient
  `0xbf8d7035` / `0xbfc80235` / `0x4158` to
  `0xbf8e0935` / `0xbfc89b35` / `0x4159`; row `5` and rows `7`/`29`/`32`
  remained matching controls. The full forced-token divergence is still not
  fixed: clean row39+row40 forced decision still selects Krasis `1044` /
  `","` while recorded HF expects `1321` / `" and"`. Added artifacts under
  `20260624_0842_nemotron_nano_bf16_ssd_self_sample_decay_fix_*`.

- Added a Nano BF16 SSD `dA` cumsum/rounding producer gate after accepted
  `0804`, `0107`, `0040`, and `10:39` artifacts. Reduced existing
  artifacts/source only; no runtime or new tracing was needed. Priority row
  `31`, dim `1857` subtracts two independently rounded `dA` cumsums at the
  final local-scan sample: live `dA_target=0xc1deb086` while duplicate running
  after final `A*dt` is `0xc1deb084`, so `target-running=-3.814697e-06`
  survives `fmin(...,0)` and yields decay `0x3f7fffbf`, coefficient `0x4158`
  versus separate-shadow `0x4159`. Row `5`, row `29`, and row `32` have exact
  zero target/running difference; row `7` has a positive one-ULP difference
  that clamps to zero, so all controls keep decay `1.0` and matching
  coefficients. Recorded a minimal production fix candidate only: for the
  local-scan final sample `u == t`, use the mathematical identity
  `decay=1.0f`; otherwise keep the existing expression. No production patch,
  clean SSD kernel change, per-iteration trace, rejected hook reintroduction,
  protected config edit, or INT4 work. Validation used `./dev build`,
  artifact/source assertions, `bash -n dev`, py-compile through
  `./dev python`, `git diff --check`, protected config check, source-gating
  audit, and cleanup. Gemma baseline was not rerun and remains
  `5619.6 / 92.43 / 155.69`.

- Added an accepted Nano BF16 SSD same-kernel final-sample decay producer retry
  after the `0118` timing audit. Set the acceptance rule before launch,
  reintroduced only one minimal request-gated final-sample decay operand row in
  the same-kernel trace diagnostic path, and launched the exact Nano command
  `./dev run tests/nemotron-nano-bf16-experts-a16.conf --test-endpoints` on
  port `18021`. Startup matched the accepted Nano profile: short/long decode
  `9.0/8.8 tok/s`, long probe `39920+32`, long calibration `289.414s`, long
  prefill `14070/14070 MB`, decode min free `24626 MB`, heatmap `1651`, HCS
  `1230`, ready free `845 MB`. Priority row `31`, dim `1857` now has direct
  same-kernel decay evidence: `dA_target=0xc1deb086`, final running
  `dA=0xc1deb084`, `target-after=-3.814697e-06`, decay arg `0xb6800000`, and
  decay `0x3f7fffbf` versus separate-shadow `1.0`, producing coefficient
  `0x4158` vs shadow `0x4159`. Row `5` and rows `7`/`29`/`32` remain clean
  controls. Diagnostic only: no production patch, clean SSD kernel change,
  per-iteration trace, rejected checksum/arithmetic hook reintroduction,
  protected config edit, or INT4 work. Gemma baseline was not rerun and remains
  `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 SSD same-kernel final-sample decay producer gate after
  accepted `0107`, `0040`, and `10:39` artifacts. Existing accepted evidence
  localizes row `31`, dim `1857` to same-kernel final-sample decay/scale
  (`0.999996127`) versus separate-shadow decay `1.0`, but direct same-kernel
  final-sample `dA_target`, running `dA`, decay input, and decay bits were
  missing. Attempted one minimal request-gated decay operand row in the
  same-kernel trace variant only, launched the exact Nano command
  `./dev run tests/nemotron-nano-bf16-experts-a16.conf --test-endpoints` on
  port `18021`, and stopped before any selected request while long calibration
  had not completed. Posthoc timing audit compared actual `2343`, `0040`, and
  `0118` dev-run timestamps: accepted long calibration completed in
  `289.874s` and `289.593s`, while `0118` was stopped after a recorded `367s`
  without a completion/ready line. Because no predeclared matched Nano
  rejection threshold existed, `0118` is now classified as inconclusive
  duration-only evidence against the decay hook, not proven hook perturbation.
  The unaccepted decay hook was removed and rebuilt. Diagnostic only: no
  accepted new operand evidence, no production patch, per-iteration trace,
  clean SSD kernel change, selected request, rejected checksum/arithmetic hook
  reintroduction, protected config edit, or INT4 work. Gemma baseline was not
  rerun and remains `5619.6 / 92.43 / 155.69`. Added timing audit artifact
  `20260624_0750_nemotron_nano_bf16_0118_timing_audit.tsv`.

- Added a Nano BF16 SSD dt softplus/scale boundary gate after accepted `0040`,
  `10:39`, `23:51`, and `00:01` artifacts. Reduced existing raw JSON/source
  only; no new metadata or runtime was needed. Priority row `31`, dim `1857`
  matches same-kernel versus separate-shadow on raw dt (`0x4050`), softplus
  (`0x41450004`), `C@B` (`0x3f8cab8e`), `x`, and indices. Same-kernel scale is
  lower (`0x4144ffd2`, inferred decay `0.999996127`) and falls below the
  coefficient midpoint scale threshold, so pre-coeff `0x41587fd6` rounds to
  `0x4158`; separate-shadow final-sample decay is `1.0`, scale is
  `0x41450004`, and the coefficient rounds to `0x4159`. Row `5` and rows
  `7`/`29`/`32` match on dt/softplus/scale/coefficient. Diagnostic only: no
  production patch, replay promotion, per-iteration trace, rejected hook
  reintroduction, extra forced row, protected config edit, or INT4 work. Gemma
  baseline was not rerun and remains `5619.6 / 92.43 / 155.69`.

- Added the accepted Nano BF16 SSD same-kernel final coefficient operands
  retry after the `0009` launch/config audit, using the exact accepted Nano
  launch shape
  `./dev run tests/nemotron-nano-bf16-experts-a16.conf --test-endpoints` on
  port `18021`. Added only a selected-row/dim final-sample operand summary to
  the request-gated same-kernel SSD trace variant, not the clean production SSD
  kernel. Startup matched the accepted `2343` profile (`9.1/8.7 tok/s`, long
  probe `39920`, long prefill `14070/14070 MB`, decode min free `24626 MB`).
  Row `31`, dim `1857` matches separate-shadow on indices, `x`, raw dt, and
  `C@B`, but same-kernel scale produces pre-BF16 coefficient `0x41587fd6`,
  just below the `0x4158/0x4159` midpoint, so same-kernel stores `0x4158`
  while separate-shadow stores `0x4159`. Row `5` and rows `7`/`29`/`32`
  remain clean controls. Diagnostic only: no production patch, replay
  promotion, clean SSD kernel change, rejected hook reintroduction,
  per-iteration trace, extra forced row, protected config edit, or INT4 work.
  Gemma baseline was not rerun and remains `5619.6 / 92.43 / 155.69`.

- Audited the `0009` same-kernel final coefficient operands rejection before
  opening a new gate. Accepted `2343` used
  `./dev run tests/nemotron-nano-bf16-experts-a16.conf --test-endpoints`
  (Nano BF16 experts, port `18021`, long calibration probe `39920` tokens);
  rejected `0009` used
  `./dev run tests/nemotron-super-bf16kv-a16.conf --test-endpoints` (Super
  BF16KV/INT4 experts, port `18018`, long calibration probe `8325` tokens).
  The `0009` calibration shift is therefore an invalid config comparison, not
  evidence against the removed same-kernel operand hook. Added launch/config
  audit artifact
  `20260624_0033_nemotron_nano_bf16_0009_rejection_launch_config_audit.tsv`.

- Added a Nano BF16 SSD same-kernel final coefficient operands gate after the
  accepted `0001`, `23:51`, `23:43`, and `10:39` artifacts. Existing accepted
  evidence still has row `31`, dim `1857` split at coefficient formation
  (`0x4158` same-kernel inferred vs `0x4159` separate-shadow), with row `5`
  and rows `7`/`29`/`32` clean. Attempted a compact request-gated final-sample
  operand summary in the same-kernel trace variant only, but later audit showed
  the rejected runtime used the Super BF16KV/INT4 config rather than the
  accepted Nano BF16 config. Removed the unaccepted hook and rebuilt.
  Diagnostic only: no accepted new operand
  evidence, no production patch, replay promotion, retained metadata, rejected
  hook reintroduction, extra forced row, protected config edit, or INT4 work.
  Gemma baseline was not rerun and remains `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 SSD final-sample coefficient formation gate after the
  accepted `23:51`, `23:43`, and `10:39` artifacts. Revalidated guardrails and
  reduced existing raw traces/source only; no new metadata or runtime was
  added. Priority row `31`, dim `1857` keeps matching source token, flat
  index, dt index, B/C base, and derived pre-final accumulator, but the
  same-kernel final term `0xbf811800` implies BF16 coefficient `0x4158`
  (`13.5`) while separate-shadow final term `0xbf81b100` uses coefficient
  `0x4159` (`13.5625`). The BF16 midpoint is `13.53125`; separate-shadow
  `C@B` and a CPU softplus proximity estimate land just above it at
  `13.531263091`. Row `5` and secondary rows `7`/`29`/`32` remain clean at
  coefficient level. Diagnostic only: same-kernel per-sample
  `x/dt/C@B/decay/scale` operands are still missing from accepted artifacts,
  so the next boundary is same-kernel coefficient operands at `u=31`; no
  production patch, replay promotion, new metadata, rejected hook
  reintroduction, extra forced row, protected config edit, or INT4 work.
  Gemma baseline was not rerun and remains `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 SSD sampled final local-scan term producer gate after the
  accepted `23:27`, `23:43`, and `10:39` artifacts. Revalidated guardrails and
  reduced existing raw traces/source only; no new metadata or runtime was
  added. Priority row `31`, dim `1857` has matching source token, flat index,
  dt index, and B/C base between same-kernel context and separate-shadow
  context. The derived pre-final accumulator matches (`0xbdc58350`), but the
  final sampled term differs: same-kernel `0xbf811800` vs separate-shadow
  `0xbf81b100`. With shadow `x_bits=0xbd99`, the same-kernel term implies
  BF16 coefficient `0x4158`, while separate-shadow emits `0x4159`. Row `5`
  and secondary rows `7`/`29`/`32` remain clean at final-term/coefficient
  level. Diagnostic only: no production patch, replay promotion, new metadata,
  rejected hook reintroduction, extra forced row, protected config edit, or
  INT4 work. Gemma baseline was not rerun and remains
  `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 SSD same-kernel vs separate-shadow input/checksum boundary
  gate after the accepted `23:43` same-kernel context result. Revalidated
  `2343` and `1039`, carried the accepted Nemotron/Gemma baselines, and
  compared same-kernel context/checksum rows against separate-shadow
  pointer/index/sample rows for row `31`, dim `1857`, row `5`, dim `801`, and
  secondary rows `7`/`29`/`32`. Existing accepted evidence localizes the
  priority split to the sampled final local-scan input term: same-kernel
  `0xbf811800` vs separate-shadow `0xbf81b100`; first/mid samples match, and
  row `5` plus secondary targets have matching sampled terms. A
  request-gated separate-shadow checksum-summary metadata shape was attempted
  but rejected before any selected request because startup stayed in long
  calibration for several minutes; the hook was removed and the final build
  passed. Diagnostic only: no production patch, replay promotion,
  same-kernel arithmetic hook reintroduction, extra forced row, protected
  config edit, or INT4 work. Gemma baseline was not rerun and remains
  `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 SSD codegen/instruction-sequence boundary gate after the
  `23:43` same-kernel context result. Revalidated `2343`, carried the accepted
  Nemotron/Gemma baselines, inspected generated PTX/source for clean production
  SSD, same-kernel trace, and separate shadow/post-kernel replay, and wrote
  source/PTX comparison artifacts. Existing accepted evidence remains: row
  `31`, dim `1857` live production and same-kernel duplicate local scan both
  `0xbf8d7035`, while prior separate shadow/post-kernel remains `0xbf8e0935`;
  row `5`, dim `801` stays clean at `0x3c168000`. A request-gated
  same-kernel final-summary arithmetic-detail shape was attempted but rejected
  before any internal request because startup stayed in long calibration for
  several minutes. The unaccepted arithmetic-detail hook was removed and the
  build was restored. Diagnostic only: no production patch, replay promotion,
  extra forced row, protected config edit, or INT4 work. Gemma baseline was not
  rerun and remains `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 SSD same-kernel duplicate-summary gate after the `10:39`
  live-loop input-context result. Revalidated `1039`, carried the latest
  Nemotron/Gemma baselines, rejected the first same-kernel instrumentation
  shape because it perturbed startup/heatmap, then split the SSD launch so
  normal startup uses the clean production kernel and selected
  `/v1/internal/reference_test` traces use a request-gated trace variant. The
  accepted startup completed normally: short/long decode `9.0/8.7 tok/s`, long
  prefill post-alloc/min free `14070/14070 MB`, decode min free `24626 MB`,
  heatmap `1612`, HCS `1230`, ready free `845 MB`. Priority row `31`, dim
  `1857`: live production local scan `0xbf8d7035`, same-kernel duplicate local
  scan `0xbf8d7035`, duplicate `y` `0xbfc80235`, and production store
  `0xbfc8`; prior separate-kernel shadow local scan from `1039` was
  `0xbf8e0935`. Row `5`, dim `801` stayed clean and secondary rows
  `7`/`29`/`32` had same-kernel duplicate matching live production. Diagnostic
  only: no production output change, replay promotion, per-iteration tracing,
  extra forced row, protected config edit, or INT4 work. Gemma baseline was not
  rerun and remains `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 production SSD live-loop input-context gate after the
  `09:46` local-scan accumulation result. Revalidated `0946`, carried the
  latest Nemotron/Gemma baselines, and added a request-gated post-production
  shadow/replay diagnostic kernel after SSD that reads exact production input
  buffer pointers and selected row/dim flat indices without adding
  per-iteration tracing inside `mamba2_ssd_sequential_kernel`. A first quiet
  server without `--test-endpoints` returned the expected `404` before model
  work; the accepted server used `--test-endpoints`. Fresh startup stayed
  normal: short/long decode `4.4/4.3 tok/s`, long prefill post-alloc/min free
  `14070/14070 MB`, decode min free `24626 MB`, heatmap `1632`, HCS `1230`,
  ready free `845 MB`. Priority row `31`, dim `1857`: production local scan is
  `0xbf8d7035`, post-kernel local scan is `0xbf8e0935`, and shadow replay
  reading production buffers/indices also returns `0xbf8e0935`; production,
  shadow out, and shadow x flat indices match at `128833`. Row `5`, dim `801`
  stays clean (`0x3c168000` production/post/shadow), and secondary rows
  `7`/`29`/`32` also match production-vs-shadow local scan. Diagnostic only:
  no production patch, replay promotion, per-iteration loop trace, extra forced
  row, protected config edit, or INT4 work. Gemma baseline was not rerun and
  remains `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 SSD local-scan accumulation gate after the `09:16`
  production SSD actual-`y` source result. Revalidated `0916`, carried the
  latest Nemotron/Gemma baselines, inspected existing traces/source, rejected
  two direct production-loop trace attempts because startup/heatmap was
  perturbed, then used the accepted request-gated post-kernel accumulation
  candidate trace plus the existing production final-scalar metadata. Fresh
  startup stayed normal: short/long decode `4.4/4.3 tok/s`, long prefill
  post-alloc/min free `14070/14070 MB`, decode min free `24626 MB`, heatmap
  `1642`, HCS `1230`, ready free `845 MB`. Priority row `31`, dim `1857`:
  production local scan remains `0xbf8d7035`, while post-kernel default,
  explicit `mul.rn+add.rn`, `fma.rn`, and Kahan candidates all reproduce the
  HF/post-kernel value `0xbf8e0935`; FP32-CB-scale gives `0xbf8dbc0f`, also
  not production. Row `5`, dim `801` stays clean at `0x3c168000`, and
  secondary rows `7`/`29`/`32` match production-vs-post-kernel at local scan
  for these target dims. Diagnostic only: no production patch, replay
  promotion, extra forced row, protected config edit, or INT4 work. Gemma
  baseline was not rerun and remains `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 production SSD actual-`y` source gate after the `08:44`
  SSD/norm-input store-boundary result. Revalidated `0844`, carried the
  latest Nemotron/Gemma baselines, added request-gated production SSD component
  metadata at the selected row/dim store path, rebuilt, and ran five internal
  `/v1/internal/reference_test` requests for priority row `31` dim `1857`,
  clean contrast row `5` dim `801`, and secondary rows `7` dim `2458`, `29`
  dim `1550`, and `32` dim `884` under the same row39+row40 forced-slot
  reproduction setup. Fresh startup stayed normal: short/long decode
  `4.4/4.3 tok/s`, long prefill post-alloc/min free `14070/14070 MB`, decode
  min free `24626 MB`, heatmap `1657`, HCS `1230`, ready free `845 MB`.
  Priority row `31`, dim `1857`: flat index and `D*x` are clean, but the
  production in-kernel local scan accumulator is `0xbf8d7035` while the
  HF/post-kernel recompute path is `0xbf8e0935`, producing actual `y`
  `0xbfc80235` instead of `0xbfc89b35`. Row `5` remains a clean contrast
  (`0xbd626000 -> 0xbd62`), while rows `7`/`29`/`32` do not show the same
  production-vs-recompute source issue. Diagnostic only: no production patch,
  replay promotion, extra forced row, protected config edit, or INT4 work.
  Gemma baseline was not rerun and remains `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 SSD/norm-input store-boundary gate after the `08:35`
  norm-input producer localization. Revalidated `0835`, carried the latest
  Nemotron/Gemma baselines, added request-gated production-store/readback
  metadata around `mamba2_ssd`, rebuilt, and ran two internal
  `/v1/internal/reference_test` requests for row `31` dim `1857` and row `5`
  dim `801` under the same row39+row40 forced-slot reproduction setup. Fresh
  startup stayed normal: short/long decode `4.4/4.4 tok/s`, long prefill
  post-alloc/min free `14070/14070 MB`, decode min free `24626 MB`, heatmap
  `1649`, HCS `1230`, ready free `845 MB`. Priority row `31`, dim `1857` is
  not a BF16 conversion/store failure or later overwrite: the actual SSD
  production kernel computes `0xbfc80235 -> 0xbfc8`, immediately stores
  `0xbfc8`, and later SSD/gated-norm readbacks stay `0xbfc8`. The earlier
  `0xbfc89b35 -> 0xbfc9` was from post-kernel diagnostic recomputation, not
  the real store source. Row `5` remains a clean contrast
  (`0xbd626000 -> 0xbd62`, store/readback/input all `0xbd62`). Diagnostic
  only: no production patch, replay promotion, extra forced row, protected
  config edit, or INT4 work. Gemma baseline was not rerun and remains
  `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 norm-input producer localization gate after the `08:05`
  all-row replay mismatch localization. Revalidated `0805`, carried the
  latest Nemotron/Gemma baselines, and reduced existing HF/Krasis target-row
  traces only; no server run or new metadata was needed. Priority row `31`,
  dim `1857` now localizes to the layer-0 SSD/norm-input BF16 store:
  `D*x` matches, recomputed SSD pre-store bits match (`0xbfc89b35` both), and
  both sides produce BF16 candidate `0xbfc9`, but Krasis production stores
  `0xbfc8` into SSD/norm input while HF stores `0xbfc9`. Row `5`, dim `801`
  is the contrast case: norm-input producer matches through store (`0xbd62`
  both), so its remaining split is downstream at `mean_square+eps`. Secondary
  rows `7` and `29` differ earlier in local scan/pre-store, and row `32`
  differs in combined pre-store bits. Diagnostic only: no production patch,
  replay promotion, SSD/gated-RMSNorm/out-proj/router/correction-loader
  change, protected-config edit, extra forced row, or INT4 work. Gemma
  baseline was not rerun and remains `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 all-row replay mismatch localization gate after the
  `23:30` all-row replay validation. Revalidated `2330`, carried the latest
  Nemotron/Gemma baselines, generated archived HF target-row evidence for rows
  `0`, `5`, `7`, `29`, `31`, and `32`, and ran one
  `/v1/internal/reference_test` request with the existing all-row
  `sqrt.approx.ftz.f32 + div.rn.f32` replay plus only row39+row40 forced-slot
  reproduction setup. Fresh startup stayed normal: short/long decode
  `4.4/4.3 tok/s`, long prefill post-alloc/min free `14070/14070 MB`, decode
  min free `24626 MB`, heatmap `1632`, HCS `1230`, ready free `845 MB`.
  Remaining affected first pre-out-proj dims are now localized: row `5` dim
  `801` first differs at `mean_square+eps`; rows `7` dim `2458`, `29` dim
  `1550`, `31` dim `1857`, and `32` dim `884` first differ at the BF16 norm
  input feeding gated RMSNorm. Priority row `31` has matching replay `rstd`
  (`0x3afd988c`) but differs at norm input (`0xbfc9` vs `0xbfc8`) and gated
  product (`0xbe24d770` vs `0xbe24057d`), so the remaining mismatch is
  upstream of replayed `rstd`. Diagnostic only: no production patch, replay
  promotion, SSD/conv/out-proj/router/correction-loader change,
  protected-config edit, extra forced row, or INT4 work. Gemma baseline was not
  rerun and remains `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 gated-RMSNorm all-row internal replay validation gate after
  the `22:39` row-1 replay result. Revalidated `2239`, carried the latest
  Nemotron/Gemma baselines, generated archived HF layer-0 rows `0..41` evidence
  with cache disabled, and ran one `/v1/internal/reference_test` request with
  explicit `sqrt.approx.ftz.f32 + div.rn.f32` replay entries for all layer-0
  rows while keeping only row39+row40 forced-slot reproduction setup. Accepted
  retry startup stayed normal: short/long decode `4.4/4.4 tok/s`, long prefill
  post-alloc/min free `14070/14070 MB`, decode min free `24626 MB`, heatmap
  `1668`, HCS `1230`, ready free `845 MB`. Replay applied to all `42` rows
  with `0` skipped and `production_behavior_changed=false`. Token matched HF:
  both selected `1321` / `" and"`; Krasis `logp(1321)-logp(1044)=+0.097341`.
  Full-row exactness did not generalize: pre-out-proj mismatches remain on rows
  `5`, `7`, `29`, `31`, `32` (`37` total bit mismatches), and out-proj
  mismatches remain on rows `5`, `7`, `31`, `32` (`353` total), with row `31`
  the major outlier. Diagnostic validation only: no production gated RMSNorm
  patch, SSD/conv/out-proj/router/correction-loader change, protected-config
  edit, extra forced row, or INT4 work. Gemma baseline was not rerun and remains
  `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 gated-RMSNorm internal replay validation gate after the
  `22:10` CUDA arithmetic-candidate result. Revalidated `2210`, carried the
  latest Nemotron/Gemma baselines, and added an internal
  `/v1/internal/reference_test`-only replay path for layer-0 gated RMSNorm
  using `sqrt.approx.ftz.f32 + div.rn.f32`. Fresh accepted startup stayed
  normal: short/long decode `4.4/4.3 tok/s`, long prefill post-alloc/min free
  `14070/14070 MB`, decode min free `24626 MB`, heatmap `1652`, HCS `1230`.
  Row `0` remained a clean control. For row `1`, dim `439`, baseline Krasis
  stored `0x3ae1` while HF stored `0x3ae2`; the internal replay stores
  `0x3ae2`. Full row-1 pre-out-proj now matches HF exactly
  (`0xb6a1b1fac2cc0b21`, `0` mismatches), downstream row-1 out-proj now
  matches HF exactly (`0xd7b46918f987463a`, `0` mismatches), and the forced
  token flips from baseline Krasis `1044` / `","` to HF `1321` / `" and"`.
  Diagnostic validation only: no production gated RMSNorm behavior patch,
  SSD/conv/out-proj/router/correction-loader change, protected-config edit,
  extra forced row, or INT4 work. Gemma baseline was not rerun and remains
  `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 gated-RMSNorm CUDA arithmetic-candidate gate after the
  `21:35` rstd-boundary result. Revalidated `2135`, carried the latest
  Nemotron/Gemma baselines, and added request-gated inline PTX candidate
  metadata only around layer-0 gated RMSNorm. Fresh startup stayed normal:
  short/long decode `4.5/4.3 tok/s`, long prefill post-alloc/min free
  `14070/14070 MB`, decode min free `24626 MB`, heatmap `1662`, HCS `1230`,
  ready free about `845 MB`. Row `1`, dim `439` keeps matching
  `mean_square + eps` bits `0x4338bd27`. HF fused `rmsnorm_fn` uses
  `rstd=0x3d96ada8`; Krasis production `rsqrtf`, C `1.0f/sqrtf`,
  double-promoted, inline `sqrt.rn.f32+div.rn.f32`,
  `sqrt.rn.f32+rcp.rn.f32`, and `rsqrt.approx.ftz.f32` all give
  `0x3d96ada7`. Inline `sqrt.approx.ftz.f32+div.rn.f32` and
  `sqrt.approx.ftz.f32+div.approx.ftz.f32` give `0x3d96ada8`, matching HF for
  the target. Diagnostic only: no gated RMSNorm behavior patch, production
  change, correction-loader change, SSD/conv/out-proj/router change,
  protected-config edit, extra forced row, or INT4 work. Gemma baseline was
  not rerun and remains `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 gated-RMSNorm `rstd` boundary gate after the `20:40` ULP
  arithmetic result. Revalidated `2040`, carried the latest Nemotron/Gemma
  baselines, and added request-gated candidate metadata only around layer-0
  gated RMSNorm `rstd`. Fresh startup stayed normal: short/long decode
  `4.5/4.4 tok/s`, long prefill post-alloc/min free `14070/14016 MB`, decode
  min free `24626 MB`, heatmap `1681`, HCS `1230`. Row `1`, dim `439` has
  identical HF/Krasis `mean_square + eps` bits `0x4338bd27`. HF fused
  `rmsnorm_fn` actual replay uses `rstd=0x3d96ada8`, matching HF
  `1.0 / torch.sqrt(...)`; HF `torch.rsqrt(...)`, Krasis CUDA `rsqrtf`,
  Krasis CUDA `1.0f / sqrtf`, and Krasis double-promoted candidates all give
  `0x3d96ada7`. Diagnostic only: no gated RMSNorm behavior patch, BF16
  conversion patch, SSD/conv/out-proj/router/correction-loader change,
  protected-config edit, extra forced row, production-path change, or INT4
  work. Gemma baseline was not rerun and remains `5619.6 / 92.43 / 155.69`.

- Added a Nano BF16 gated-RMSNorm ULP arithmetic gate after the `20:24` HF
  actual store-provenance result. Revalidated `2024`, carried the latest
  Nemotron/Gemma baselines, and added request-gated metadata only around
  layer-0 gated RMSNorm arithmetic/provenance. Fresh rebuilt startup stayed
  normal: short/long decode `4.5/4.4 tok/s`, long prefill post-alloc/min free
  `14070/14070 MB`, decode min free `24626 MB`, heatmap `1608`, HCS `1230`.
  Row `1`, dim `439` matches HF manual arithmetic and Krasis through norm
  input, gate, SiLU, gated product, mean-square, epsilon,
  mean-square+epsilon, norm weight, and manual FP32 pre-store bits
  `0x3ae17fff`; HF fused `rmsnorm_fn` actual replay differs at `rstd`
  (`0x3d96ada8` vs Krasis/manual `0x3d96ada7`) and actual FP32 output
  (`0x3ae18001` vs `0x3ae17fff`), crossing midpoint `0x3ae18000` and storing
  `0x3ae2` while Krasis stores `0x3ae1`. Row `0` remains a BF16-store
  control. Diagnostic only: no Krasis conversion, SSD, conv, out-proj, router,
  correction-loader, production-path, protected-config, additional-forced-row,
  or INT4 change. Also corrected a stale GPU unit-test launch argument so it
  matches the restored raw `sigmoid_topk_kernel` ABI. A temporary diagnostic
  trace-slot edit initially disabled prefill PTX compile; it was corrected
  before the accepted rebuild/runtime and the failed log is preserved.

- Added a Nano BF16 HF gated-norm store-provenance gate after the `19:50`
  conversion/store boundary. Revalidated `19:50`, carried the latest Nemotron
  header (`4.4/4.3 tok/s`, long prefill min free `14070 MB`, decode min free
  `24626 MB`, heatmap `1646`, HCS `1230`) and Gemma baseline (`5619.6`
  prefill, `92.43` internal decode, `155.69` HTTP), and added HF/reference
  metadata only around `MambaRMSNormGated.forward ->
  mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn`. The new trace proves row
  `1`, dim `439` manual reconstruction was stale for the actual store:
  manual HF pre-store bits are `0x3ae17fff` below midpoint `0x3ae18000`, but
  the actual fused producer replay emits `0x3ae18001`, producing BF16
  candidate/store `0x3ae2`. Row `0` remains the clean control. Krasis remains
  `0x3ae17fff -> 0x3ae1` and stores the standard candidate, so no Krasis BF16
  conversion patch is justified. Diagnostic only: no additional forced rows,
  router variant, SSD/conv/out-proj/correction-loader patch, production-path
  change, protected config edit, or INT4 work.

- Added a Nano BF16 layer-0 row-1 BF16 conversion/store boundary gate after
  the `18:38` gated-RMSNorm producer result. Revalidated `18:38`, printed the
  latest Nemotron header (`4.4/4.3 tok/s`, long prefill post-alloc/min free
  `14070/14070 MB`, decode min free `24626 MB`, heatmap `1603`, HCS `1230`)
  plus the Gemma baseline (`5619.6` prefill, `92.43` internal decode,
  `155.69` HTTP). Existing traces lacked exact conversion/store bits, so
  added request-gated metadata only, rebuilt with `./dev build`, generated HF
  rows `0/1`, and reran the internal Krasis trace on the same 42-token input
  with only layer-1 rows `39`/`40` forced as reproduction setup. Fresh startup
  stayed normal: `4.4/4.3 tok/s`, long prefill min free `14070 MB`, decode min
  free `24626 MB`, heatmap `1646`, HCS `1230`. Row `0` is the clean control.
  Row `1`, dim `439` reports identical HF/Krasis pre-store FP32 bits
  `0x3ae17fff`, one ULP below midpoint `0x3ae18000`; standard BF16
  round-to-nearest-even candidate is `0x3ae1`. Krasis stores `0x3ae1` and
  matches the candidate; HF reports stored tensor bits `0x3ae2` while its
  reported candidate is `0x3ae1`. Diagnostic only: no additional forced rows,
  router variant, SSD/conv/out-proj/correction-loader patch, production-path
  change, protected config edit, or INT4 work.

- Added a Nano BF16 layer-0 row-1 gated RMSNorm producer gate after the
  `13:41` out-proj boundary. Revalidated `13:41`, printed the latest Nemotron
  header (`4.4/4.3 tok/s`, long prefill post-alloc/min free
  `14070/14016 MB`, decode min free `24626 MB`, heatmap `1665`, HCS `1230`,
  ready free `845 MB`) plus the Gemma baseline (`5619.6` prefill, `92.43`
  internal decode, `155.69` HTTP). Existing traces lacked dim `439`
  gated-RMSNorm detail, so added request-gated metadata only, rebuilt with
  `./dev build`, regenerated HF/Krasis row `0/1` evidence on the same
  42-token input, and kept only layer-1 rows `39`/`40` forced as reproduction
  setup. Fresh rebuilt startup stayed normal at `4.4/4.3 tok/s`, long prefill
  post-alloc/min free `14070/14070 MB`, decode min free `24626 MB`, heatmap
  `1603`, HCS `1230`. Row `0` remains a clean control. Row `1`, dim `439`
  matches through post-conv `x`, `D*x`, local scan, SSD pre-store/stored norm
  input, gate `z`, SiLU gate, gated product, RMS stats, norm weight, and FP32
  pre-store output. The first material split is final BF16 store into
  pre-out-proj: HF `0x3ae2` / `0.0017242431640625`, Krasis `0x3ae1` /
  `0.00171661376953125`, one BF16 step lower. Diagnostic only: no additional
  forced rows, router variants, SSD/conv/out-proj/correction-loader patch,
  production-path change, protected config edit, or INT4 work.

- Added a Nano BF16 layer-0 row-1 out-proj boundary gate after the `12:58`
  Mamba-internals result. Revalidated `12:58`, printed the latest Nemotron
  header (`4.4/4.3 tok/s`, long prefill min free `14070 MB`, decode min free
  `24626 MB`, heatmap `1675` carried / `1657` fresh, HCS `1230`) plus the
  Gemma baseline (`5619.6` prefill, `92.43` internal decode, `155.69` HTTP).
  Existing traces lacked full/expanded row `0/1` pre-out-proj and out-proj
  details, so added request-gated diagnostic metadata only, rebuilt with
  `./dev build`, regenerated HF/Krasis row `0/1` evidence on the same
  42-token input, and kept only layer-1 rows `39`/`40` forced as reproduction
  setup. Fresh rebuilt startup stayed normal at `4.4/4.3 tok/s`, long prefill
  post-alloc/min free `14070/14016 MB`, decode min free `24626 MB`, heatmap
  `1665`, HCS `1230`. Row `0` is a clean control. Row `1` first differs in
  full pre-out-proj input before out-proj GEMM at dim `439` (`HF 0x3ae2`,
  Krasis `0x3ae1`, delta `-7.62939453125e-06`). The first out-proj output
  diff is downstream at dim `91`, with matching weight row hash
  `0x09ddbaf6f4585f51` and no bias. Diagnostic only: no additional forced
  rows, router variants, SSD/conv/out-proj patch, production-path change,
  protected config edit, or INT4 work.

- Added a Nano BF16 layer-0 Mamba row-1 internals gate after the `12:15`
  layer-0 row-1 producer result. Revalidated `12:15`, printed the latest
  Nemotron header (`4.4/4.3 tok/s`, long prefill min free `14070 MB`, decode
  min free `24626 MB`, heatmap `1675`, HCS `1230`) plus the Gemma baseline
  (`5619.6` prefill, `92.43` internal decode, `155.69` HTTP). Existing traces
  lacked row `0/1` post-conv and scan accumulator coverage, so added
  diagnostic-only selected-row metadata to the archived HF harness and the
  `/v1/internal/reference_test` Krasis trace path, rebuilt with `./dev build`,
  generated HF rows `0/1`, and ran one Krasis row `0/1` metadata trace on the
  same 42-token input with only layer-1 rows `39` and `40` forced as
  reproduction setup. Fresh metadata-server startup stayed normal at
  `4.4/4.3 tok/s`, long prefill min free `14070 MB`, decode min free
  `24626 MB`, heatmap `1657`, HCS `1230`. Row `0` remains a clean control.
  Row `1` matches HF by full-row hash and selected BF16 dims through layer-0
  input, RMSNorm output, mixer input, in-proj, raw pre-conv `x/b/c/dt`, and
  post-conv `x/b/c`; selected BF16 scan/component terms also match. Tiny FP32
  decay/scale diagnostic deltas exist (`max_abs=2.384185791015625e-07`) but
  collapse before BF16 terms/pre-store. First material full-row split remains
  layer-0 Mamba out-proj/mixer branch output (`HF 0xd7b46918f987463a`,
  Krasis `0xad683fc6e71b3ead`) with selected output dims still matching
  `17/17`. Diagnostic only: no additional forced rows, router variant,
  production change, SSD/conv patch, protected config edit, or INT4 work.

- Added a Nano BF16 layer-0 row-1 producer gate after the `11:40`
  scan-history producer result. Revalidated `11:40`, printed the latest
  Nemotron header (`4.4/4.3 tok/s`, long prefill post-alloc/min free
  `14070/14026 MB`, decode min free `24626 MB`, heatmap `1686`, HCS `1230`)
  plus the current Gemma baseline (`5619.6` prefill, `92.43` internal decode,
  `155.69` HTTP). Existing traces lacked HF layer-0 row-index internals, so
  added archived-HF diagnostic-only `--diagnose-layer0-row-indices`, rebuilt
  with `./dev build`, generated HF rows `0/1`, then ran two internal Krasis
  row `0/1` traces on the same 42-token input with only layer-1 rows `39` and
  `40` forced as reproduction setup. Fresh startup stayed normal at
  `4.4/4.3 tok/s`, long prefill min free `14070 MB`, decode min free
  `24626 MB`, heatmap `1675`, HCS `1230`. Row `0` is a clean control through
  layer-0 output/handoff. Row `1` matches through layer-0 input/RMSNorm,
  mixer input, in-proj, and raw pre-conv `x/b/c/dt`; earliest available
  scan-internal split is Mamba `state_pre`, and the first exact BF16 full-row
  split is the layer-0 Mamba mixer/output branch (`HF 0xd7b46918f987463a`,
  Krasis `0xad683fc6e71b3ead`). Diagnostic only: no additional forced rows,
  router variant, production change, conv/SSD fix, protected config edit, or
  INT4 work.

- Added a Nano BF16 layer-2 scan-history producer gate after the `11:23`
  SSD/chunk-scan boundary. Reverified `11:23`, printed the latest Nemotron
  header (`4.4/4.3 tok/s`, long prefill post-alloc/min free `14070/14026 MB`,
  decode min free `24626 MB`, heatmap `1686`, HCS `1230`) plus the current
  Gemma baseline (`k4 restore clean speed`: `5619.6` prefill, `92.43`
  internal decode, `155.69` HTTP). Existing traces lacked rows `0/1`, so
  generated request-gated HF/Krasis rows `0/1/41` evidence and one upstream
  layer-1 row `1` evidence pass, keeping only layer-1 rows `39` and `40`
  forced as reproduction setup. Fresh Krasis startup stayed normal at
  `4.4/4.3 tok/s`, long prefill min free `14070 MB`, decode min free
  `24626 MB`, heatmap `1652`, HCS `1230`, request `2.111946s`. Row `0`
  matches HF through layer-2 input/RMSNorm/in-proj/raw preconv, row `41`
  remains the matched control, and row `1` is already split at layer-2 input
  (`HF 0x53c6b4a18c4a8409`, Krasis `0x207ea7dead2a8578`). Upstream trace
  shows the first proven split is layer-1 row `1` input/pre-norm
  (`HF 0x4d52b372606d71f6`, Krasis `0xbc069341c2692f4d`), so the next
  boundary is layer-0 to layer-1 handoff / layer-0 row-1 producer. Diagnostic
  only: no source edit, metadata edit, additional forced rows, router variant,
  production change, conv/SSD patch, protected config change, or INT4 work.

- Added a Nano BF16 layer-2 Mamba SSD/chunk-scan boundary gate after the
  `10:38` row39+row40 forced followup. Reverified the `10:38` artifacts and
  guardrails, then reduced existing HF and Krasis request-gated traces into a
  focused SSD/chunk-scan comparison without running a new server or editing
  source. The same 42-token forced-prefix2 decision still returns Krasis
  `1044` instead of HF `1321`, using only the existing layer-1 row `39` and
  row `40` forced-slot reproduction setup. Layer-2 row `41` remains clean
  through post-conv `x/b/c`; Krasis SSD GPU output matches its host recompute
  to one BF16 element (`max_abs=7.62939453125e-06`), state initialization is
  clean (`0.0` states / prior chunk state `0.0`), and selected `D*x` matches
  where available. The remaining split is caused by mismatched full-chunk
  scan-history inputs: selected `dA`/decay differs before local accumulation
  at token position `0`, and material selected `x`/raw-`dt`/`C@B` mismatches
  begin at chunk token position `1`. Diagnostic only: no source edit,
  metadata edit, additional forced rows, router variant, production router
  change, correction-loader change, conv patch, protected config change,
  server run, or INT4 work was made.

- Added a Nano BF16 row39+row40 forced causality followup gate after the
  `10:15` row-40 router causality run. Reverified the forced-slot metadata,
  relaunched Nano BF16 with `KRASIS_REFERENCE_MAMBA_TRACE_LAYERS=2 ./dev run
  tests/nemotron-nano-bf16-experts-a16.conf --test-endpoints` in tmux, and
  sent the same 42-token forced-prefix2 continuation input with only layer-1
  rows `39` and `40` forced through `/v1/internal/reference_test`. Fresh
  startup stayed normal at `4.4/4.3 tok/s`, long prefill min free
  `14026 MB`, decode min free `24626 MB`, heatmap `1686`, HCS `1230`. Forced
  row `40` now fixes the prior row-40 history/conv-window mismatch:
  layer-1 row-40 routed/shared/combined/handoff match HF, and layer-2 row-41
  pre-norm, RMSNorm, in-proj, raw pre-conv, and post-conv `x/b/c` match HF.
  The token still returns `1044` instead of HF `1321`; the first remaining
  split is layer-2 Mamba SSD/chunk-scan output, with only `3/14` available
  selected SSD dims matching. No additional rows were forced and no source
  edit, production router change, broad variant, correction-loader change,
  conv patch, protected config change, direct `cargo`, or INT4 work was made.

- Added a Nano BF16 row-40 router causality gate after the `10:01`
  router-selection audit. Reverified the `10:01` artifacts/guardrails and ran
  a fresh Nano BF16 `--test-endpoints` server through `./dev run` in tmux.
  Used the exact 42-token forced-prefix2 continuation input, preserving the
  layer-1 row `39` forced-slot reproduction setup
  `[7,18,24,39,87,116]`, then forced only layer-1 row `40` to HF final order
  `[39,43,102,111,116,114]` with raw-sigmoid weights paired by expert through
  `/v1/internal/reference_test`. Fresh startup stayed normal at
  `4.4/4.3 tok/s`, long prefill min free `14070 MB`, decode min free
  `24626 MB`, heatmap `1589`, HCS `1230`. The token did not flip:
  HF expects `1321`, while row-39-only and row-39+row-40-forced Krasis both
  return `1044`. Row-40 forcing narrowed the `1321 - 1044` log-prob gap from
  `-0.251919` to `-0.035735`, so it is contributory but insufficient. No next
  continuation window was run, and no source edit, production router change,
  broad variant, correction-loader change, conv patch, protected config
  change, or INT4 work was made.

- Added a Nano BF16 forced-prefix2 layer-1 row-40 router-selection audit after
  the `10:31` causal-history producer gate. Reverified `10:31` and printed the
  latest valid Nemotron header (`4.4/4.3 tok/s`, long prefill min free
  `14046 MB` fresh / `14070 MB` carry-forward, decode min free `24626 MB`,
  heatmap `1665`, HCS `1230`) plus the current Gemma benchmark row (`k4
  restore clean speed`: `5619.6` prefill, `92.43` internal decode,
  `155.69` HTTP). Used existing request-gated/offline artifacts only. The
  row-40 split is explained by selection scores: HF uses FP32
  `e_score_correction_bias` and selects `[39,43,102,111,116,114]`, while
  restored Krasis uses raw-sigmoid top-k `[39,102,116,0,87,43]`. FP32
  correction replaces raw-only experts `0`/`87` with corrected experts
  `111`/`114`; Nano group filtering is degenerate and no top-k boundary tie is
  present. Current BF16 correction expectation collapses relevant corrections
  to `56.75`, preserving raw order. Diagnostic only: no source edit, metadata
  edit, router variant, production router change, conv patch, protected config
  change, or INT4 work was made.

- Added a Nano BF16 forced-prefix2 layer-2 row-40 history-producer gate after
  the `09:42` downstream conv-window evidence. Reverified `09:42`, printed the
  latest valid Nemotron header (`4.4/4.3 tok/s`, long prefill min free
  `14070 MB`, decode min free `24626 MB`, heatmap `1650`, HCS `1230`) and the
  current Gemma benchmark row (`k4 restore clean speed`: `5619.6` prefill,
  `92.43` internal decode, `155.69` HTTP). Generated fresh archived-HF
  layer-1/layer-2 row-40 evidence, then ran traced
  `/v1/internal/reference_test` requests on the same 42-token forced-prefix2
  input using only the layer-1 row `39` forced slot-order reproduction control.
  Fresh startup stayed normal: `4.4/4.3 tok/s`, long prefill min free
  `14046 MB`, decode min free `24626 MB`, heatmap ranked `1665`, HCS
  soft-loaded `1230`. Layer-1 row `40` matches HF through pre-norm input,
  RMSNorm output, and MoE input, then splits at router top-k selection:
  HF `[39,43,102,111,116,114]`, Krasis raw `[39,102,116,0,87,43]`. First
  tensor split is routed pre-shared output (`HF 0x4e55806902de9b0d`, Krasis
  `0x3c9f9b0d0270f3dd`); shared output still matches. The mismatch then
  propagates through combined MoE output, layer-1 handoff, layer-2 pre-norm,
  RMSNorm, in-proj, and raw pre-conv `x/b/c/dt`. Diagnostic only: no source
  edit, metadata edit, router variant, production router change, conv patch,
  protected config change, or INT4 work was made.

- Added a Nano BF16 forced-prefix2 layer-2 row-41 downstream evidence gate
  after the `09:10` layer-1 continuation gate. Reverified the `09:10`
  artifacts and guardrails, generated fresh archived-HF layer-2 row-41
  evidence through the built `generate-reference` wrapper, then ran Nano BF16
  `/v1/internal/reference_test` requests on a fresh `--test-endpoints` server
  with only the layer-1 row `39` forced slot-order control needed to reproduce
  the prefix2-fixed state. A second request using the existing
  `KRASIS_REFERENCE_MAMBA_TRACE_LAYERS=2` diagnostic env exposed row-level
  Mamba raw/post-conv evidence. Fresh startup stayed normal: `4.4/4.3 tok/s`,
  long prefill min free `14070 MB`, decode min free `24626 MB`, heatmap ranked
  `1650`, HCS soft-loaded `1230`, request `2.074907s`. Row `41` still
  diverges at the token level (`HF 1321` / `" and"`, Krasis `1044` / `","`),
  but layer-2 pre-norm input, RMSNorm output, Mamba in-proj, and raw pre-conv
  `x/b/c/dt` match by row hash. The first concrete split is Mamba depthwise
  post-conv `x/b/c`; conv source positions/kinds, weights, and bias match, and
  all conv-window input mismatches come from causal-history row `40`. No
  production router change, router variant, conv patch, protected config
  change, or INT4 work was made.

- Added a Nano BF16 forced-prefix2 continuation gate after the row-39
  slot-order causality result. Built the exact prefix2-plus-two input where
  row-39 forcing has already produced `1321` and the next matched continuation
  token `1044`, generated fresh archived-HF row-41 layer-1 evidence through
  the built reference wrapper, then ran traced `/v1/internal/reference_test`
  requests using only the row-39 forced slot control needed to recreate the
  prefix2-fixed state. Fresh startup stayed normal: `4.4/4.3 tok/s`, long
  prefill post-alloc `14070 MB`, long prefill min free `14058 MB`, decode min
  free `24626 MB`, heatmap ranked `1657`, HCS soft-loaded `1230`. The first
  continuation divergence after the fixed prefix2 state is decision row `41`:
  HF selects `1321` / `" and"`, Krasis selects `1044` / `","`. Row `41`
  router expert set and raw-sigmoid weights match by expert, but slot order
  differs (`HF [7,18,24,39,87,116]`, Krasis raw
  `[18,39,87,24,7,116]`). Despite that, selected BF16 bits match HF through
  layer-1 input, RMSNorm/MoE input, routed branch, shared branch, combined MoE
  output, and layer-1 handoff (`17/17` selected dims at every stage; routed,
  shared, combined, and handoff hashes match). Diagnostic only: the
  continuation divergence is downstream of layer 1; no router variant,
  production router change, conv patch, protected config change, or INT4 work
  was made.

- Added a Nano BF16 slot-order causality gate for the row-39 prefix2
  corrected-router regression. Added request-gated
  `debug_router_forced_slot_orders` support on `/v1/internal/reference_test`
  only, leaving production raw-sigmoid routing unchanged/default. Built with
  `./dev build`, launched a fresh Nano BF16 `--test-endpoints` server, and
  forced layer 1 row `39` to HF slot order `[7,18,24,39,87,116]` with
  raw-sigmoid weights paired by expert. Fresh startup stayed normal:
  `4.4/4.3 tok/s`, long prefill min free `14070 MB`, decode min free
  `24626 MB`, heatmap ranked `1643`, HCS soft-loaded `1230`. The forced order
  fixes prefix2 versus the rejected corrected override (`1044` -> `1321`), but
  the six-token continuation from the exact prefix2 input is unchanged versus
  raw (`[1321,1044,1044,1044,1044,1044]`) and reconstructs max8 with first
  diff at index `4`. No production router change, broad variant, conv patch,
  protected config change, or INT4 work was accepted.

- Added a Nano BF16 prefix2 layer-1 corrected-router regression gate after the
  `07:31` layer-scope and `07:58` HF-router semantics gates. Identified the
  exact prefix2 decision row as row `39` from the saved `40`-token payload,
  generated fresh HF row-39 layer-1 router evidence through the built
  `generate-reference` wrapper, and sent raw plus layer-1
  `corrected_hf_unsorted` traced `/v1/internal/reference_test` requests on a
  fresh Nano BF16 `--test-endpoints` server. Fresh startup stayed normal:
  `4.4/4.3 tok/s`, long prefill min free `14070 MB`, decode min free
  `24626 MB`, heatmap ranked `1632`, HCS soft-loaded `1230`. Raw prefix2
  matches HF (`1321` / `" and"`); layer-1 corrected override still regresses
  prefix2 (`1044` / `","`). Row `39` proves the corrected override has the
  same expert set as HF but not the same slot order: HF
  `[7,18,24,39,87,116]`, Krasis override `[18,39,87,24,7,116]`. Raw-sigmoid
  values match by expert but are paired to slots in Krasis override order. No
  source edit, production router change, new variant, conv patch, INT4 work,
  direct `cargo`, or protected config change was made.

- Added a Nano BF16 diagnostic-only HF router semantics audit after the
  `07:31` router variant layer-scope gate. Reverified `13:47`, `14:17`,
  `06:38`, and `07:31` artifacts/guardrails, then generated an offline audit
  artifact set through `./dev python`; no source edit, model run, or server run
  was made. Latest valid speed header remains `4.5/4.3 tok/s`, long prefill
  min free `14070 MB`, decode min free `24626 MB`, heatmap ranked `1637`, HCS
  soft-loaded `1230`. The HF Nemotron router contract is now explicit:
  sigmoid logits, FP32 `e_score_correction_bias` for choice, top-2-per-group
  group scores, `topk_group` via `torch.topk(sorted=False)`, non-selected group
  mask to `0.0`, final `torch.topk(sorted=False)`, raw-sigmoid weight gather,
  normalization, and routed scale. Nano has `n_group=1`/`topk_group=1`, so
  group filtering is not causal for rows `40`/`42`; current
  `corrected_hf_unsorted` remains a Rust partition-order approximation, not an
  exact PyTorch unsorted top-k emulator. No variant accepted; production
  raw-sigmoid routing remains reverted/default. No conv patch, INT4 work,
  direct `cargo`, or protected config change.

- Added request-gated layer scoping to the Nano BF16 router variant diagnostic
  override on `/v1/internal/reference_test`. New debug-only
  `debug_router_variant_layers` accepts layer indices for corrected variant
  execution; omitted/empty scope preserves the existing all-layer debug
  override. Production raw-sigmoid routing and the normal CUDA
  `sigmoid_topk_kernel` path remain unchanged. Built with `./dev build`, then
  token-tested prefix2/max8 on a fresh `--test-endpoints` Nano BF16 server.
  Startup stayed normal (`4.5/4.3 tok/s`, long prefill min free `14070 MB`,
  decode min free `24626 MB`, heatmap ranked `1637`, HCS soft-loaded `1230`).
  No layer-scoped variant was accepted: layer-1-only corrected routing fails
  prefix2 and has max8 first diff `2`, layer-2-only is equivalent to raw
  baseline with first diff `6`, and all-layer corrected routing regresses max8
  first diff to `4`. No production router correction, conv patch, INT4 work,
  direct `cargo`, or protected config change was made.

- Added a request-gated Nano BF16 router variant execution override for
  `/v1/internal/reference_test` only. Production routing remains the restored
  raw-sigmoid top-k default; the override is disabled unless an internal
  reference-test request passes `debug_router_variant`. Built with
  `./dev build`, generated FP32 correction-vector payloads through
  `./dev python`, and token-tested restored raw baseline, HF-unsorted
  corrected order, sorted corrected order, and corrected-set/raw-slot-weight
  variants for prefix2 and max8. Fresh startup stayed normal (`4.4/4.3 tok/s`,
  long prefill post-alloc free `14070 MB`, long prefill min free `14036 MB`,
  decode min free `24626 MB`, heatmap ranked `1646`, HCS soft-loaded `1230`).
  No variant was accepted: raw baseline keeps max8 first diff at index `6`,
  HF-unsorted-corrected and sorted-corrected regress max8 first diff to `4`,
  and corrected-set/raw-slot-weights fails prefix2 and has max8 first diff
  `2`. The first server attempt without `--test-endpoints` returned 404s and
  was discarded as evidence. No production router correction, conv patch, INT4
  work, direct `cargo`, or protected config change was made.

- Added a Nano BF16 diagnostic-only router variant gate after the `14:17`
  slot-semantics pass. Reverified the `13:47` and `14:17` artifacts and
  guardrails, kept production router correction reverted, and generated a
  fresh offline artifact set through `./dev python`. Current speed header
  remains the `13:47` run: `4.4/4.4 tok/s`, long prefill min free
  `14070 MB`, decode min free `24626 MB`, heatmap ranked `1638`, HCS
  soft-loaded `1230`, request `1.22s`. The gate classified corrected expert
  set, HF `torch.topk(..., sorted=False)` order, sorted corrected order, and
  slot/weight pairing candidates. The only token-tested corrected variant
  remains the rejected FP32 sorted-corrected attempt, which regressed prefix2
  and max8 at generated index `2`; restored raw-sigmoid remains only the
  baseline with first max8 diff `6`. HF-unsorted-corrected and slot-pairing
  variants remain unexecuted row-only candidates because no narrow
  request-gated all-row router override exists. No source edit, production
  patch, conv patch, server run, INT4 work, direct `cargo`, or protected config
  change was made.

- Added a Nano BF16 diagnostic-only router slot-semantics gate for restored
  prefix-6 layer-1 rows `40`/`42` after the `13:47` producer run. Reverified
  artifacts/guardrails and kept production router correction reverted. Existing
  artifacts were sufficient, so no source edit or fresh server run was needed:
  the gate produced offline comparison artifacts from `13:47`, `09:17`, and
  `10:31` data through `./dev python`. Current speed header remained
  `4.4/4.4 tok/s`, long prefill min free `14070 MB`, decode min free
  `24626 MB`, heatmap ranked `1638`, HCS soft-loaded `1230`. HF actual route
  order uses corrected sets in `torch.topk(..., sorted=False)` order, while
  current Krasis uses raw-sigmoid order. BF16-loaded correction values collapse
  the relevant bias range to `56.75`, but FP32 correction was already
  token-rejected at `09:17`: row-40 routed/combined hashes matched, yet max8
  regressed at generated index `2`. Sorted-vs-unsorted order and slot/weight
  pairing remain unproven as direct causes. No router retry is accepted without
  prefix2/max8 token-level improvement; no production patch, conv patch, INT4
  work, direct `cargo`, or protected config change was made.

- Added a Nano BF16 restored-prefix-6 upstream layer-1 rows `40`/`42`
  producer evidence gate after the `13:17` layer-2 handoff run. Reverified
  artifacts/guardrails, kept router production correction reverted, and used
  existing selected-row diagnostics only: archived HF
  `./dev generate-reference --diagnose-layer1-row-indices 40,42` plus a fresh
  Krasis BF16 diagnostic server/request with
  `debug_prefill_device_trace_layer=1` and rows `[40,42]`. Startup stayed
  normal (`4.4/4.4 tok/s`, long prefill min free `14070 MB`, decode min free
  `24626 MB`, heatmap ranked `1638`, HCS soft-loaded `1230`, request
  `1.22s`). Rows `40` and `42` match HF through layer-1 pre-norm input,
  RMSNorm output, and MoE input. The first proven producer mismatch is
  layer-1 router top-k selection: row 40 HF `[39,43,102,111,116,114]` vs
  Krasis `[39,102,116,0,87,43]`; row 42 HF `[39,43,61,102,116,114]` vs
  Krasis `[39,102,116,87,43,0]`. First tensor split is routed pre-shared
  output; shared branch matches for both rows, then combined MoE output and
  layer-1/layer-2 handoff diverge downstream. No source edit, production
  patch, conv patch, router-correction retry, INT4 work, direct `cargo` probe,
  or protected config change was made.

- Added a Nano BF16 restored-prefix-6 layer-2 raw-history rows `40`/`42`
  evidence gate after the `12:29` layer-2 internals run. Reverified
  artifacts/guardrails, kept the router production correction reverted, and
  used existing selected-row diagnostics only: archived HF
  `./dev generate-reference --diagnose-layer2-row-indices 40,42` plus a fresh
  Krasis BF16 diagnostic server with `KRASIS_REFERENCE_MAMBA_TRACE_LAYERS=2`.
  Startup stayed normal (`4.4/4.3 tok/s`, long prefill min free `14070 MB`,
  decode min free `24626 MB`, heatmap ranked `1635`, HCS soft-loaded `1230`,
  request `2.23s`). Rows `40` and `42` both first diverge at layer-2
  pre-norm input / previous-layer handoff before RMSNorm, in-proj, or raw
  pre-conv `x/b/c/dt`: row 40 HF `0xd37e2e97026b920d` vs Krasis
  `0x3fb8f95fc94ffb2f`; row 42 HF `0x0266f6d429209f98` vs Krasis
  `0x49018f99d631ecc5`. Krasis layer-1 output-sum selected dims equal its
  layer-2 input selected dims (`17/17` for both rows). No source edit,
  production patch, conv patch, router-correction retry, INT4 work, direct
  `cargo` probe, or protected config change was made.

- Added a Nano BF16 restored-prefix-6 layer-2 internals evidence gate after
  the `11:03` baseline divergence and `10:31` router rollback. Router
  production correction remains reverted. Generated fresh HF layer-2 internals
  through the built archived reference command, then ran fresh Nano BF16
  diagnostic servers through `./dev run tests/nemotron-nano-bf16-experts-a16.conf
  --test-endpoints`, with the second run enabling existing request-gated
  layer-2 Mamba tracing. Startup stayed normal (`4.4/4.3 tok/s`, long prefill
  min free `14070 MB`, decode min free `24626 MB`, heatmap ranked `1660`, HCS
  soft-loaded `1230`, request `2.23s`). Prefix
  `[1044,1044,1321,1044,1321,1044]` still selects `1044` in Krasis while HF
  selects `1321`. Decision row `43` matches HF through layer-2 pre-norm input,
  RMSNorm/mixer input, Mamba in-proj, and raw pre-conv `x/b/c/dt`; first
  concrete split is layer-2 Mamba post-conv `x/b/c`. Conv source metadata,
  weights, and bias match, but selected conv-window inputs differ on causal
  history rows `40` and `42` (`71/136` window inputs match). No production
  patch, router correction retry, INT4 work, direct `cargo` probe, protected
  config edit, or broad architecture change was made.

- Added a Nano BF16 restored-baseline prefix-6 divergence evidence gate after
  the `10:31` router rollback. The production router correction remains
  reverted. Generated fresh HF evidence through `./dev generate-reference` and
  ran one fresh Nano BF16 diagnostic server through
  `./dev run tests/nemotron-nano-bf16-experts-a16.conf --test-endpoints`.
  Startup stayed normal (`4.4/4.3 tok/s`, long prefill min free `14070 MB`,
  decode min free `24626 MB`, heatmap ranked `1623`, HCS soft-loaded `1230`).
  Exact prefix `[1044,1044,1321,1044,1321,1044]` reproduces the remaining
  generated-index-6 divergence: HF selects `1321`, Krasis selects `1044`, and
  Krasis ranks `1321` third. Hidden summaries match through layer-1 output;
  the first aggregate split is HF layer-2 output vs Krasis layer-3 input.
  No production patch, router correction retry, INT4 work, direct `cargo`
  probe, protected config edit, or broad architecture change was made.

- Rejected and reverted the unaccepted Nano BF16 router score-correction
  production change after a focused regression gate. The `09:17` change fixed
  the row-40 routed/MoE hash, but it regressed max8 from the previous
  pre-router sequence `[1044,1044,1321,1044,1321,1044,1044,1321]` to
  `[1044,1044,1044,1044,1044,1044,1044,1321]`. A controlled revert restored
  the pre-router max8 sequence and restored the prefix `[1044,1044]` next
  token to `1321`; row-40 routing returned to raw-sigmoid top-k
  `[39,102,116,0,87,43]`, so the old row-40 routed mismatch is again the
  active boundary. The regression is caused by the production router/loader
  change as a whole, not accepted as an independent downstream bug.
  Correction-bias dtype alone is not the sole cause because both the BF16
  correction attempt and the FP32 correction attempt regressed at generated
  index `2`; sorted/unsorted top-k order remains a future diagnostic-only
  variant before any reattempt. Built and validated with `./dev build`,
  generated regression artifacts under
  `benchmarks/20260620_1031_nemotron_nano_bf16_router_regression_*`, and kept
  INT4 blocked. No protected config edit, broad architecture change, direct
  `cargo` probe, HCS/calibration/heatmap/safety change, fake reference, or
  fallback was made.

- Added diagnostic-only routed-path evidence for the Nano BF16 row-40 layer-1
  MoE mismatch. Reverified the `08:23` direct routed/shared branch artifacts,
  printed the accepted baseline header (`4.4/4.3 tok/s`, ready about `793s`,
  heatmap about `363s`), added request-gated Krasis selected-row router/routed
  path evidence, built with `./dev build`, and ran a fresh Nano BF16
  diagnostic server. Startup stayed normal (`4.4/4.3 tok/s`, long prefill min
  free `14058 MB`, decode calibration min free `24626 MB`, heatmap ranked
  `1667` experts, HCS soft-loaded `1230` experts), and the request completed
  in `1.220216s`. Row-40 router comparison proves the exact routed-path
  mismatch is top-k selection: HF actual top-k IDs are
  `[39,43,102,111,116,114]`, while Krasis selects `[39,102,116,0,87,43]`,
  exactly matching HF raw-sigmoid top-6 instead of HF correction-aware top-k
  with `e_score_correction_bias`. Routed BF16 output remains split
  (`0x4e55806902de9b0d` vs `0x3c9f9b0d0270f3dd`), shared output remains
  clean at `0x2248c576ca8e0845`, and combined MoE/mixer output remains
  downstream-split (`0x3195a7da319fc9ac` vs `0xb264b2fc925c2883`). No
  production patch, fake reference, direct `cargo` probe, INT4 work, HCS/
  routing/calibration/heatmap/safety/async change, or protected config edit
  was made.

- Added diagnostic-only direct branch evidence for the Nano BF16 row-40
  layer-1 MoE mismatch. Reverified the `08:42` row-40 layer-1 producer
  artifacts and accepted baseline header (`4.4/4.3 tok/s`, ready about
  `793s`, heatmap about `363s`), then added request-gated Krasis selected-row
  branch details for routed round-once pre-shared output, shared pre-add
  output, and combined MoE/mixer output. Built with `./dev build` and ran a
  fresh Nano BF16 diagnostic server. Startup stayed normal (`4.4/4.3 tok/s`,
  long prefill min free `14070 MB`, decode calibration min free `24626 MB`,
  heatmap ranked `1656` experts, HCS soft-loaded `1230` experts), and the
  request completed in `1.241s`. Direct row-40 comparison proves the mismatch
  is routed branch output before shared add: HF `0x4e55806902de9b0d` vs
  Krasis `0x3c9f9b0d0270f3dd`, while the shared branch matches exactly at
  `0x2248c576ca8e0845`; combined MoE/mixer output remains split
  (`0x3195a7da319fc9ac` vs `0xb264b2fc925c2883`). No production patch, fake
  reference, direct `cargo` probe, INT4 work, HCS/routing/calibration/heatmap/
  safety/async change, or protected config edit was made.

- Added diagnostic-only selected-row HF and Krasis evidence for the Nano BF16
  row-40 layer-1 producer boundary. Reverified the `08:08` handoff artifacts
  and accepted baseline header (`4.4/4.3 tok/s`, ready about `793s`, heatmap
  about `363s`), extended the approved HF `./dev generate-reference
  --diagnose-layer1-internals` path with `--diagnose-layer1-row-indices`,
  extended Krasis request-gated selected-row tracing, built with `./dev build`,
  generated the exact-prefix HF artifact, and ran a fresh Nano BF16 diagnostic
  server. Startup stayed normal (`4.4/4.3 tok/s`, long prefill min free
  `14070 MB`, decode calibration min free `24626 MB`, heatmap ranked `1643`
  experts, HCS soft-loaded `1230` experts), and the requests completed in
  `1.22s`/`1.92s`. Row `40` layer-1 pre-norm input, RMSNorm output, and MoE
  input match HF exactly, but the first available mismatch is layer-1
  MoE/mixer output: HF `0x3195a7da319fc9ac` vs Krasis handoff RHS proxy
  `0xb264b2fc925c2883`. Layer-1 output/layer-2 handoff remains split
  (`0xd37e2e97026b920d` vs `0x3fb8f95fc94ffb2f`). Direct Krasis row-40
  routed/shared branch BF16 hashes are still missing, so no production patch,
  fake reference, direct `cargo` probe, INT4 work, HCS/routing/calibration/
  heatmap/safety/async change, or protected config edit was made.

- Added diagnostic-only selected-row Krasis handoff evidence for the Nano BF16
  row-40 layer-1 to layer-2 boundary. Reverified the `21:33` row-40
  raw-history artifacts and accepted baseline header (`4.4/4.3 tok/s`, ready
  about `793s`, heatmap about `363s`), audited existing HF/Krasis evidence, and
  used the existing approved HF row-40 layer-2 input/norm details. Krasis lacked
  row-40 pre-norm handoff visibility, so request-gated trace support emitted
  `layer1_output_sum_selected_rows`, `layer2_rmsnorm_input_selected_rows`, and
  `layer2_rmsnorm_output_selected_rows`. A fresh Nano BF16 server through
  `./dev run tests/nemotron-nano-bf16-experts-a16.conf --test-endpoints`
  reached ready normally (`4.5/4.3 tok/s`, long prefill min free `14070 MB`,
  decode idle floor `628 MB`, HCS soft-loaded `1230` experts) and the row-40
  request completed in `2.039s`. The first mismatch is now proven at layer-2
  pre-norm input / layer-1 output handoff: HF `0xd37e2e97026b920d` versus
  Krasis `0x3fb8f95fc94ffb2f`, with `0/17` selected dims matching. Krasis is
  internally self-consistent (`layer1_output_sum == layer2_rmsnorm_input`), so
  layer-2 RMSNorm is downstream of an already-wrong handoff row. No production
  patch, fake reference, INT4 work, HCS/routing/calibration/heatmap/safety/
  async change, direct `cargo` probe, or protected config edit was made.

- Added diagnostic-only selected-row HF and Krasis evidence for the Nano BF16
  layer-2 Mamba row-40 raw-history boundary. Reverified the `20:42` conv
  window/state artifacts and accepted baseline header (`4.4/4.3 tok/s`, ready
  about `793s`, heatmap about `363s`), extended the approved HF
  `./dev generate-reference --diagnose-layer2-internals` path with
  `--diagnose-layer2-row-indices`, extended Krasis request-gated selected-layer
  Mamba tracing with `debug_prefill_device_trace_rows`, built via
  `./dev build`, generated the exact-prefix HF artifact, and ran a fresh Nano
  BF16 diagnostic server. Startup stayed normal with short/long decode
  `4.3/4.3 tok/s`, long prefill min free `14070 MB`, heatmap ranked `1643`
  experts, HCS soft-loaded `1230` experts, and request HTTP completed in
  `2.06s`. Row `40` already splits at layer-2 norm output / Mamba mixer input
  before in-proj and raw x/b/c/dt, so the raw split/layout hypothesis is
  disproven but no production-patch proof exists yet. No production patch,
  fake reference, INT4 work, HCS/routing/calibration/heatmap/safety/async
  change, or protected config edit was made.

- Added diagnostic-only HF and Krasis layer-2 Mamba2 conv window/state evidence
  for the Nano BF16 investigation. Reverified the `20:10` conv-contract
  artifacts and accepted baseline header (`4.4/4.3 tok/s`, ready about `793s`,
  heatmap about `363s`), extended the approved HF
  `./dev generate-reference --diagnose-layer2-internals` path plus Krasis
  request-gated selected-layer Mamba tracing, built with `./dev build`,
  generated the exact-prefix HF artifact, and ran a fresh BF16 diagnostic
  server. Startup stayed normal with short/long decode `4.4/4.4 tok/s`, long
  prefill min free `14070 MB`, heatmap ranked `1624` experts, HCS soft-loaded
  `1230` experts, and request HTTP `200` in `1.344731s`. The `18` common
  selected channels match on source positions/kinds, weights, bias, state-used
  flags, and Krasis manual formula; every window mismatch is at
  `kernel_index=2` / source row `40` in the pre-conv sequence history. This
  rules out the selected-row conv kernel formula/padding/bias/state path and
  moves the boundary to layer-2 raw x/b/c history at row `40`. No production
  patch, fake reference, INT4 work, HCS/routing/calibration/heatmap/safety/
  async change, or protected config edit was made.

- Added diagnostic-only Krasis layer-2 Mamba2 conv contract metadata for the
  Nano BF16 investigation. Reverified the `19:50` HF post-conv artifacts and
  accepted baseline header (`4.4/4.3 tok/s`, ready about `793s`, heatmap about
  `363s`), built with `./dev build`, launched a fresh normal BF16 diagnostic
  server through `./dev run ... --test-endpoints`, and compared the new Krasis
  request-gated trace against the `19:50` HF artifact. Startup stayed normal
  with short/long decode `4.4/4.3 tok/s`, long prefill min free `14070 MB`, and
  heatmap ranked `1628` experts. Krasis now exposes matching groups, padding,
  stride, dilation, kernel width, BF16-rounded weight/bias hashes, and raw
  pre-conv x/b/c hashes. Post-conv x/b/c still split, so the boundary is now
  layer-2 Mamba2 conv compute/padding-state after matching input, weight, bias,
  and layout. No production patch, fake reference, INT4 work,
  HCS/routing/calibration/heatmap/safety/async change, or protected config edit
  was made.

- Added diagnostic-only HF layer-2 Mamba2 post-conv evidence for the Nano BF16
  investigation. Reverified the `19:31` raw pre-conv artifacts/guardrails and
  accepted baseline header (`4.4/4.3 tok/s`, ready about `793s`, heatmap about
  `363s`). Extended `./dev generate-reference --diagnose-layer2-internals` to
  emit selected-layer conv metadata plus exact post-conv x/b/c BF16 row
  hashes/details, built with `./dev build`, generated the exact-prefix HF
  artifact through the built command only, and compared against the existing
  `19:01` Krasis request-gated trace. Raw pre-conv x/b/c/dt remains clean, but
  post-conv rows split: x HF `0xbd0dd9865ba962b7` vs Krasis
  `0x08d9da747b2bb981`, b HF `0x838b16d4d12fbfe4` vs Krasis
  `0x4ff888d4db721d05`, and c HF `0x3ec63fca949f77a9` vs Krasis
  `0x9ea5d4f23e94f1ad`. HF records depthwise conv groups/padding/stride/
  dilation and weight/bias hashes, while the existing Krasis trace lacks the
  corresponding explicit layout/hash fields, so no production patch, fake
  reference, INT4 work, HCS/routing/calibration/heatmap/safety/async change, or
  protected config edit was made.

- Added diagnostic-only HF layer-2 Mamba2 raw pre-conv split evidence for the
  Nano BF16 investigation. Reverified the `19:01` artifacts/guardrails and
  accepted baseline header (`4.4/4.3 tok/s`, ready about `793s`, heatmap about
  `363s`). Extended `./dev generate-reference --diagnose-layer2-internals` to
  emit selected-layer actual pre-conv Mamba2 `x/b/c/dt` rows with BF16 FNV
  hashes/details, built with `./dev build`, generated the exact-prefix HF
  artifact through the built command only, and compared against the existing
  `19:01` Krasis request-gated trace. Full-row BF16 hashes match for raw x
  (`0x23d4a705efa283f2`), raw b (`0x04010f7d7b2ea9c8`), raw c
  (`0xb92aa05c80c03354`), and raw dt (`0xaf157a8e8bf8445e`), so the layer-2
  in-proj split/raw pre-conv rows are clean. The next boundary remains the
  post-conv x/b/c split; no production patch, fake reference, INT4 work,
  HCS/routing/calibration/heatmap/safety/async change, or protected config edit
  was made.

- Added diagnostic-only HF layer-2 Mamba2 internals support for the Nano BF16
  investigation. Reverified the `18:42` artifacts/guardrails and printed the
  accepted baseline header (`4.4/4.3 tok/s`, ready about `793s`, heatmap about
  `363s`). Generated the exact-prefix HF artifact through the built
  `./dev generate-reference` command only, then ran a fresh normal Nano BF16
  server with existing request-gated Krasis selected-layer Mamba tracing
  (`KRASIS_REFERENCE_MAMBA_TRACE_LAYERS=2`). Startup remained normal:
  short/long decode `4.4/4.3 tok/s`, long prefill min free `14070 MB`, decode
  idle floor `628 MB`, and heatmap ranked `1640` experts. HF still selects
  `1321`; Krasis still selects `1044`. The first comparable Mamba result is:
  `raw_dt` matches by aggregate, then post-conv x/b/c split (HF x l2
  `17.2229900360` vs Krasis `16.8390668105`; B `9.1850337982` vs
  `9.2437427063`; C `9.0076084137` vs `8.9373777112`). Final mixer output
  still differs (`0.3381617963` vs `0.3431797191`). HF does not yet expose raw
  pre-conv x/b/c rows, so this is evidence only: no behavior patch, fake
  reference, INT4 work, HCS/routing/calibration/heatmap/safety/async change, or
  protected config edit was made.

- Added diagnostic-only HF layer-2 internals support to
  `./dev generate-reference` for the Nano BF16 investigation. Reverified the
  `18:34` layer-2 evidence blocker and guardrails, generalized the existing
  layer-1 HF internals hook to selected layer 2 without changing hook
  semantics, built with `./dev build`, then generated one exact-prefix archived
  HF artifact through the built command only. Layer-2 input and norm output
  match the existing post-fix Krasis full-prefill trace by full-row BF16 hash
  (`0x6d31f9291cd91ac0`, `0x44c780d740c0ddfe`). The first concrete split is
  layer-2 mixer output: HF `0x43d64de1c571fb83` / l2
  `0.33816179633140564`, Krasis `0xb43d429b1ea7d0d7` / l2
  `0.34317971913723944`. Layer 2 is a Mamba block and the generalized MoE hook
  does not expose Mamba2 substage internals, so no behavior patch, fake
  reference, INT4 work, HCS/routing/calibration/heatmap/safety/async change, or
  protected config edit was made.

- Ran the Nano BF16 HF-vs-Krasis layer-2 evidence audit gate for exact base
  input plus `[1044,1044,1321,1044]`. Reverified the `17:06` post-fix
  correctness artifacts and guardrails, then audited the approved archived-HF
  reference command surface. The built command exposes
  `--diagnose-hidden-summaries` plus layer-0/layer-1 internals only; there is
  no approved `--diagnose-layer2-*` internal or element-dim capture. The current
  boundary remains HF layer-2 output versus Krasis layer-3 input by aggregate
  only. No new HF layer-2 artifact, fake reference, behavior patch, INT4 work,
  HCS/routing/calibration/heatmap/safety/async change, or protected config edit
  was made.

- Ran the Nano BF16 post-fix correctness regression gate for the accepted
  `16:14` layer-1 routed-MoE round-once patch. Reverified artifacts and
  guardrails, then launched a fresh BF16 server with `--test-endpoints`.
  Startup stayed normal: short decode `4.4 tok/s`, long decode `4.3 tok/s`,
  long prefill min free `14016 MB`, decode calibration min free `24626 MB`,
  heatmap ranked `1644` experts, HCS soft-loaded `1230` experts, and
  request-time decode min free was `846 MB`. The restored-HF max8 oracle is
  `[1044,1044,1321,1044,1321,1044,1321,1044]`; prepatch Krasis first diverged
  at index `4`, while postpatch Krasis now produces
  `[1044,1044,1321,1044,1321,1044,1044,1321]` and first diverges at index `6`.
  Exact-prefix full-prefill for `[1044,1044,1321,1044]` still selects `1044`;
  full-prefill for the new matched prefix `[1044,1044,1321,1044,1321,1044]`
  also selects `1044` with `1321` rank 3. Layer-1 routed MoE remains fixed:
  postpatch round-once hash `0xd2ae787b4110daf7` matches HF and metadata
  confirms `routed_accum_rounded_before_shared=true`. Existing diagnostics now
  localize the next aggregate split to HF layer-2 output versus Krasis layer-3
  input. No behavior patch, fake reference, INT4 work, HCS/routing/
  calibration/heatmap/safety/async change, or protected config edit was made.

- Accepted the Nano BF16 efficient layer-1 routed-MoE round-once production
  fix. Reverified the `15:52` baseline artifacts/guardrails and printed the
  accepted timing header (`4.4/4.3 tok/s`, long prefill min free `14070 MB`,
  decode calibration min free `24626 MB`, ready `792.930s`) before changing
  code. Audited the rejected attempts against that baseline: the separate
  round pass had only an inconclusive heatmap stop, while the fused shared-add
  round flag remained rejected for stopping during long-prefill calibration.
  Added a BF16 sequential-prefill in-place round-once kernel before shared add
  and kept the change out of HCS, routing, calibration, heatmap, async streams,
  INT4, and protected configs. The candidate reached ready on baseline timing:
  hash-proof startup ready `792.258s`, heatmap `363.160s`, long calibration
  complete `425.517s`, long prefill min free `14058 MB`, decode calibration
  min free `24626 MB`. The approved HF routed full-row hash
  `0xd2ae787b4110daf7` matches the f32-accumulate/single-BF16-round hash;
  prepatch production `0x89e152dd83070888` and BF16-round-each-add
  `0x939e49099456cd1f` do not. Final exact-prefix distribution still selects
  `1044`, so BF16 correctness remains blocked for further localization and
  INT4 remains blocked.

- Ran the Nano BF16 baseline-startup verification gate from final
  diagnostic-only source before any new production work. Reverified the
  `14:48` artifacts/guardrails and printed the current accepted Nemotron
  speed header (`4.4 tok/s` short decode, `4.3 tok/s` long decode, long
  prefill min free `14070 MB`, decode calibration min free `24626 MB`, prior
  decode idle floor near `628 MB`). Launched one fresh normal server with
  `./dev run tests/nemotron-nano-bf16-experts-a16.conf`, no timing env, and
  no timing request flag. Baseline reached ready: expert load complete
  `96.518s`, short calibration complete `129.696s`, long calibration complete
  `426.266s`, heatmap start `426.619s`, heatmap ranked `1642` experts at
  `789.655s`, HCS loaded `791.688s`, ready `792.930s`. The heatmap phase was
  slow (`363.036s`) but completed, so the remaining diagnostic-only source is
  not the startup blocker. No production patch, fake reference, INT4 work,
  HCS/routing/calibration/heatmap/safety/async change, or protected config
  edit was made.

- Ran the Nano BF16 layer-1 routed-MoE round-once production-fix performance
  gate. Reverified the `13:47` full-row proof and guardrails, then audited
  the rejected production patch logs. The separate full accumulator round pass
  completed calibration normally (`4.5/4.3 tok/s`, long prefill min free
  `14070 MB`) but stalled at heatmap; the fused shared-add round flag stalled
  during long-prefill calibration before long decode/heatmap. A request-gated
  timing hook around the shared-add/round boundary was attempted and built,
  but repeated fresh normal BF16 startups with that hook completed VRAM
  calibration and then stalled at heatmap before request serving. The timing
  hook was rejected and removed; final source remains diagnostic-only. No
  production patch, fake reference, INT4 work, HCS/routing/calibration/
  heatmap/safety/async change, or protected config edit was made.

- Ran the Nano BF16 layer-1 routed-MoE accumulation full-row proof gate for
  exact base input plus `[1044,1044,1321,1044]`. Reverified the `13:36`
  artifacts/guardrails, protected config cleanliness, async shared-stream
  absence, and idle GPUs/processes. Added only request-gated diagnostics in
  full-prefill routed-MoE replay for the missing f32-accumulate then
  single-BF16-round full-row summary, rebuilt with `./dev build`, and used a
  fresh normal BF16 server. The approved HF routed full-row hash is
  `0xd2ae787b4110daf7`; Krasis diagnostic f32-round-once BF16 hash matches it
  exactly, while current production f32 routed input (`0x89e152dd83070888`)
  and BF16-round-each-add (`0x939e49099456cd1f`) do not. Two production patch
  attempts were rejected and reverted: a separate full accumulator round pass
  stalled at heatmap after normal calibration, and a fused shared-add round
  flag stalled during long-prefill calibration. Final source remains
  diagnostic-only; no accepted behavior patch, fake reference, INT4 work,
  HCS/routing/calibration/heatmap/safety/async change, or protected config edit
  was made.

- Ran the Nano BF16 HF-vs-Krasis layer-1 routed-MoE internals evidence gate
  for exact base input plus `[1044,1044,1321,1044]`. Reverified the `13:19`
  artifacts/guardrails, protected config cleanliness, async shared-stream
  absence, and idle GPUs/processes. Audited the approved
  `./dev generate-reference` surface and generated one extra archived-HF
  artifact using only the existing `--diagnose-layer1-element-dims` selector.
  The selected dims
  `0,1,2,3,336,672,1344,1599,1610,2016,2684,2685,2686,2687` match Krasis
  f32 routed accumulation rounded once for `14/14` dims, but only match the
  Krasis BF16-round-each-add diagnostic for `8/14`. The HF artifact still
  lacks per-expert contribution details, latent input, and routed per-expert
  output rows; the existing Krasis artifact lacks a full f32-rounded-once BF16
  row hash. No behavior patch, fake reference, INT4 work,
  HCS/routing/calibration/heatmap/safety/async change, or protected config edit
  was made.

- Ran the Nano BF16 HF-vs-Krasis full-prefill layer-1 output evidence gate for
  exact base input plus `[1044,1044,1321,1044]`. Reverified the `13:11`
  artifacts/guardrails, protected config cleanliness, async shared-stream
  absence, and idle GPUs/processes. Audited the approved
  `./dev generate-reference` path and generated one HF artifact with
  `--diagnose-layer1-internals`; compared it to the existing Krasis
  request-gated full-prefill trace through `./dev python`. Layer-1
  input/RMSNorm hashes match exactly (`0x15a2af5fde91bfd1` input,
  `0x412eabf39001c463` RMSNorm output). Route/top-k matches by expert set,
  score-rank order `[18,39,87,24,7,116]`, scatter order
  `[7,18,24,39,87,116]`, and per-expert weights within `1e-7`. First approved
  split is the routed MoE branch before shared add: HF bf16 l2
  `0.5961683392524719` versus Krasis f32 routed accumulator l2
  `0.5960776190029813`, with diagnostic row hash split
  `0xd2ae787b4110daf7` versus `0x939e49099456cd1f`. Shared branch still
  matches exactly, then branch/block output hashes differ. No behavior patch,
  fake reference, INT4 work, HCS/routing/calibration/heatmap/safety/async
  change, or protected config edit was made because full per-expert HF
  contribution rows/formula proof remain unavailable.

- Ran the Nano BF16 HF-vs-Krasis full-prefill hidden-state localization gate
  for exact base input plus prefix `[1044,1044,1321,1044]`. Reverified the
  `12:42` artifacts/guardrails, protected config cleanliness, async
  shared-stream absence, and idle GPUs/processes. Used only the existing
  approved HF single-token artifact and Krasis request-gated full-prefill
  response (input hash `0xd0b4dbab3506849d`), parsed through `./dev python`.
  HF had `53` hidden summaries and Krasis had `439` full-prefill stage
  snapshots. Aggregate comparison found embedding and layer-0 output match,
  then first visible divergence at HF `layer_1_output` versus Krasis layer-2
  input: HF l2 `1.092778205871582`, mean
  `-0.00007888532854849473`; Krasis l2 `1.0927264375435626`, mean
  `-0.00007982126304081508`; min/max still matched. No behavior patch, fake
  reference, new server run, INT4 work, HCS/routing/calibration/heatmap/
  safety/async change, or protected config edit was made. Full HF hidden rows
  remain unavailable, so the result is a layer aggregate boundary only.

- Ran the Nano BF16 full-prefill final-distribution gate for exact base input
  plus prefix `[1044,1044,1321,1044]`. Reverified the `12:18`
  artifacts/guardrails, protected config cleanliness, async shared-stream
  absence, and idle GPUs/processes. Generated a single-token restored-HF
  reference only through the approved built command
  `KRASIS_ALLOW_ARCHIVED_HF_REFERENCE=1 ./dev generate-reference
  nemotron-nano --profile greedy_chat_thinking_off --raw-input-json ...`
  with `--max-tokens 1`, `--diag-topk 12`,
  `--diagnose-pre-post-forward`, `--diagnose-hidden-summaries`, and
  `--prefill-only-first-token`. The HF artifact provides top-k/logprobs,
  selected raw logit, logits stats, and final hidden summary aggregates, but
  not the full raw logits vector or full hidden row values. With matching input
  hash `0xd0b4dbab3506849d`, HF selected token `1321` (`" and"`) with raw
  logit `11.25` and logprob `-1.7748818397521973`; Krasis fresh normal BF16
  full-prefill selected token `1044` (`","`) with raw logit
  `11.067996978759766` and logprob `-1.8136286747706034`. Token `1321` was
  Krasis rank 2 with raw logit `10.721786499023438` and logprob
  `-2.1598391545069315`; token `1044` was HF rank 2 with inferred raw logit
  `10.75` and logprob `-2.2748818397521973`. Final hidden aggregate differs
  too: HF l2 `136.15890502929688` / mean `0.027669787406921387` versus Krasis
  l2 `135.59655425792198` / mean `0.038715138321831113`. Startup stayed
  normal: short decode `4.4 tok/s`, long decode `4.3 tok/s`, long prefill min
  free `14070 MB`, heatmap ranked `1661` experts, HCS soft-loaded `1230/2944`
  experts / `23408.4 MB`, and decode idle floor `628 MB`. No behavior patch,
  layer localization, fake reference, INT4 work, HCS/routing/calibration/
  heatmap/safety/async change, or protected config edit was made.

- Ran the Nano BF16 full-prefill versus restored-HF oracle gate at appended
  prefix `[1044,1044,1321,1044]`. Reverified the `11:48`
  generated-index-4 artifacts/guardrails, protected config cleanliness, async
  shared-stream absence, and idle GPUs/processes. Audited the archived HF
  oracle before layer work: approved `./dev generate-reference` command,
  model `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, transformers `5.5.3`, torch
  `2.11.0+cu128`, tokenizers `0.22.2`, profile
  `greedy_chat_thinking_off`, `enable_thinking=false`,
  `add_generation_prompt=true`, `max_new_tokens=8`, `generate-use-cache off`,
  and base input hash `0xf62d9e4f5b39fdc7`. HF top-k is available at boundary
  position `41`: rank 1 token `1321` (`" and"`, logprob
  `-2.0274713039398193`, logit `11.0`) and rank 2 token `1044` (`","`,
  logprob `-2.1524713039398193`); full raw logits are not stored. Fresh
  normal Nano BF16 full-prefill selected `1044` for the same exact prefix,
  with Krasis rank 1 `1044` logprob `-1.813629` and rank 2 `1321` logprob
  `-2.159839`. Startup stayed normal: short decode `4.4 tok/s`, long decode
  `4.3 tok/s`, long prefill min free `14070 MB`, heatmap ranked `1645`
  experts, HCS soft-loaded `1230/2944` experts / `23408.4 MB`, and decode
  idle floor `628 MB`. No layer work, behavior patch, fake reference, INT4
  work, HCS/routing/calibration/heatmap/safety/async change, or protected
  config edit was made.

- Ran the Nano BF16 generated-index-4 correctness gate after the max8
  validation failure. Reverified the `11:24` artifacts/guardrails, protected
  config cleanliness, async shared-stream absence, and idle GPUs/processes.
  Launched a fresh normal Nano BF16 server on
  `tests/nemotron-nano-bf16-experts-a16.conf`; startup stayed normal with
  short decode `4.4 tok/s`, long decode `4.3 tok/s`, long prefill min free
  `14070 MB`, heatmap ranked `1648` experts, HCS soft-loaded `1230/2944`
  experts / `23408.4 MB`, and decode idle floor `628 MB`. Reproduced the
  restored-HF max_tokens=8 divergence at generated index `4`: HF expected
  `[1044,1044,1321,1044,1321,1044,1321,1044]`, while Krasis produced
  `[1044,1044,1321,1044,1044,1321,1044,1321]`. Exact matched-prefix
  full-prefill controls for suffix `[1044,1044,1321,1044]` also selected
  `1044`, so this is not a decode-only mismatch against Krasis full-prefill.
  Layer-0 selected Mamba2 in-proj rows match decode production, diagnostic
  forced batched GEMM, and full-prefill output bits/values; the first
  decode-vs-full-prefill split after in-proj is layer-0 Mamba2 conv output.
  No behavior patch, fake validate reference, HCS/routing/calibration/
  heatmap/safety/async/INT4 change, or protected config edit was made.

- Ran the Nano BF16 full correctness validation gate after the `07:24`
  layer-0 Mamba2 in-proj patch. Reverified the accepted `07:24`
  artifacts/guardrails, protected config cleanliness, async shared-stream
  absence, and idle GPUs/processes. Built the current source, launched a fresh
  normal Nano BF16 server on `tests/nemotron-nano-bf16-experts-a16.conf` with
  calibration/heatmap/HCS intact, then reran the existing restored-HF `04:23`
  corrected-38 `max_tokens=8` oracle. Startup was normal: short decode
  `4.5 tok/s`, long decode `4.3 tok/s`, long prefill min free `14070 MB`,
  heatmap ranked `1627` experts, HCS soft-loaded `1230/2944` experts /
  `23408.4 MB`, and decode idle floor was `628 MB`, close to the `600 MB`
  safety margin. The max8 check still diverges: HF expected
  `[1044,1044,1321,1044,1321,1044,1321,1044]`, while Krasis produced
  `[1044,1044,1321,1044,1044,1321,1044,1321]`, first divergence index `4`
  (`1321` expected, `1044` actual). Ran
  `KRASIS_ALLOW_ARCHIVED_HF_REFERENCE=1 ./dev validate
  tests/nemotron-nano-bf16-experts-a16.conf`; the validate tool built
  successfully but could not start the validation suite because no stored Nano
  reference directory exists under `krasis-internal/reference-outputs/output`
  or `tests/reference_outputs`. No source behavior, HCS policy, routing,
  calibration/heatmap/safety, async stream, INT4 path, or protected config
  change was made.

- Accepted the Nano BF16 layer-0 Mamba2 in-proj efficient compute-contract
  gate after the `22:51` diagnostic-only result. Reverified the `22:51`
  artifacts/guardrails, async shared-stream absence, protected config
  cleanliness, and idle GPUs/processes. Added request-gated variants for
  `batch_cols=1,2,4,8,16,32,33,34,35,36,37,38,39,40` with cuBLAS
  algorithm/layout/stride/bias/input-hash metadata. The smallest
  full-prefix-equivalent shape is `batch_cols=33`: `1` matched the original
  GEMV family, `2/4/8/16/32` matched the rejected intermediate family, and
  `33-40` matched the forced full-prefix/full-prefill family. Final rebuilt
  server evidence for the target row: production path
  `prefill_equivalent_batched_gemm_bf16_repeated_current_vector`,
  `production_batch_cols=33`, input hash `0x1e8798b99cbeefb4`, hash
  `3276c341f5204538`, and l2 `316.4128442253`, exactly matching forced
  full-prefix (`batch_cols=40`) and full-prefill. Final clean/traced tokens
  were stable at `[1044,1044,1321,1044]`, while full-prefill prefix
  `[1044,1044]` selected `[1044]`. Final calibration remained normal: short
  decode `4.4 tok/s`, long decode `4.3 tok/s`, so the previous full-prefix
  `0.3 tok/s` slowdown did not reproduce. The patch is BF16 Mamba2 in-proj
  only; no HCS policy, routing, calibration/heatmap/safety, async stream, INT4
  path, or protected `testconfigs` change was made.

- Added the Nano BF16 layer0-to-layer1 handoff gate after the `21:56`
  layer1-to-layer2 handoff result. Reverified the `21:56`
  artifacts/guardrails, async shared-stream absence, protected config
  cleanliness, and idle GPUs/processes. Used a fresh normal Nano BF16 server
  with no liveness timing, no graph override, no async stream patch, and
  calibration/heatmap/safety intact. Long calibration prefill min free was
  `14070 MB`; heatmap ranked `1651` experts; HCS soft-loaded `1230/2944`
  experts / `23408.4 MB`; startup-ready free VRAM was `845 MB`, above the
  `600 MB` margin. Stable wrong clean output reproduced as
  `[1044,1044,1321,1321]`; full-prefill prefix `[1044,1044]` selected
  `[1044]`. Request-gated traces showed layer 0 input/residual and RMSNorm
  are not the causal boundary: decode post-norm input hash
  `1e8798b99cbeefb4`, l2 `61.4528517918`, aligns with full-prefill l2
  `61.4528503418`. The first concrete producer split is layer-0 Mamba2
  in-proj: decode single-vector BF16 GEMV hash `d5880def84c0c839`, l2
  `316.4245787408`, while the forced batched-GEMM diagnostic hash
  `3276c341f5204538`, l2 `316.4128442253`, aligns with full-prefill batched
  in-proj l2 `316.4128417969`. Layer-0 hidden branch then differs
  (`3b61a3db8c81f6a9` / l2 `0.0706600154` vs full-prefill
  `0x94007e4eff8c3b3e` / l2 `0.0706756338`), and layer-1 materialized norm
  input differs (`9fb934c3ac20c8ff` vs `0x99e0fd392955b958`). No behavior
  patch was accepted. The next boundary is a BF16 decode layer-0 in-proj
  compute-contract decision, not layer0-to-layer1 handoff/copy. No HCS policy,
  routing, calibration/heatmap/safety, async stream, INT4 path, or protected
  `testconfigs` change was made.

- Added the Nano BF16 layer1-to-layer2 handoff gate after the `21:32`
  layer2-to-layer3 handoff result. Reverified the `21:32`
  artifacts/guardrails, async shared-stream absence, protected config
  cleanliness, and idle GPUs/processes. Used a fresh normal Nano BF16 server
  with no liveness timing, no graph override, no async stream patch, and
  calibration/heatmap/safety intact. Long calibration prefill min free was
  `14070 MB`; heatmap ranked `1634` experts; HCS soft-loaded `1230/2944`
  experts / `23408.4 MB`; startup-ready free VRAM was `845 MB`, above the
  `600 MB` margin. Stable wrong clean output reproduced as
  `[1044,1044,1321,1321]`; full-prefill prefix `[1044,1044]` selected
  `[1044]`. Request-gated traces showed the layer1-to-layer2 decode copy is
  internally consistent: layer-1 hidden output and layer-2 hidden operand both
  hash `3b5446e0e8beee93`, and layer-1 residual output and layer-2 residual
  operand both hash `9fb934c3ac20c8ff`. Full-prefill layer1-to-layer2 residual
  handoff self-matches at `0xfbdc431e030a1be8`. Because layer-1 output already
  differs, localization stopped inside layer 1: the first split is layer-1
  input residual before norm, decode hash `9fb934c3ac20c8ff`, l2
  `0.4489390263`, versus full-prefill hash `0x99e0fd392955b958`, l2
  `0.4489058256`. Layer-1 post-input-norm/router input also differs, while
  top-k IDs match (`18,39,87,24,7,116`). No behavior patch was accepted. The
  next boundary is layer0-to-layer1 handoff / layer0 output for decode step 1
  versus full-prefill prefix `[1044,1044]`. No HCS policy, routing,
  calibration/heatmap/safety, async stream, INT4 path, or protected
  `testconfigs` change was made.

- Added the Nano BF16 layer2-to-layer3 handoff gate after the `21:07`
  layer3-to-layer4 handoff result. Reverified the `21:07`
  artifacts/guardrails, async shared-stream absence, protected config
  cleanliness, idle GPUs/processes, and printed the existing Gemma record from
  records only (`5619.6` prefill, `92.43` internal decode, `155.69` HTTP).
  Used a fresh normal Nano BF16 server with no liveness timing, no graph
  override, no async stream patch, and calibration/heatmap/safety intact. Long
  calibration prefill min free was `14070 MB`; heatmap ranked `1654` experts;
  HCS soft-loaded `1230` experts / `23408.4 MB`; startup-ready free VRAM was
  `841 MB`, above the `600 MB` margin. Stable wrong clean output reproduced as
  `[1044,1044,1321,1321]`; full-prefill prefix `[1044,1044]` selected
  `[1044]`. Request-gated traces showed decode step 1 already enters layer 2
  with a different materialized norm input/prior layer output: decode hash
  `584d5e74a9b16b79`, l2 `1.0577923070`, versus full-prefill hash
  `0xfbdc431e030a1be8`, l2 `1.0595393181`. The layer2-to-layer3 decode copy
  itself is internally consistent: layer-2 hidden output and layer-3 hidden
  operand both hash `389666cb94c2eb4b`, and layer-2 residual output and
  layer-3 residual operand both hash `584d5e74a9b16b79`. Full-prefill
  layer2-to-layer3 residual handoff self-matches at `0xf85f706d28b23343`. No
  behavior patch was accepted. The next boundary is layer1-to-layer2 handoff /
  layer1 output for decode step 1 versus full-prefill prefix `[1044,1044]`. No
  HCS policy, routing, calibration/heatmap/safety, async stream, INT4 path, or
  protected `testconfigs` change was made.

- Added the Nano BF16 layer3-to-layer4 handoff gate after the `20:44`
  layer4-to-layer5 handoff result. Reverified the `20:44`
  artifacts/guardrails, async shared-stream absence, protected config
  cleanliness, and idle GPUs/processes. Used a fresh normal Nano BF16 server
  with no liveness timing, no graph override, no async stream patch, and
  calibration/heatmap/safety intact. Long calibration prefill min free was
  `14070 MB`; heatmap ranked `1601` experts; startup-ready free VRAM was
  `845 MB`, above the `600 MB` margin. Stable wrong clean output reproduced as
  `[1044,1044,1321,1321]`; full-prefill prefix `[1044,1044]` selected
  `[1044]`. Request-gated traces showed decode step 1 already enters layer 3
  with a different materialized norm input: decode hash `aba23db132521334`, l2
  `1.0717368059`, versus full-prefill hash `0xf85f706d28b23343`, l2
  `1.0729236603`. The layer3-to-layer4 decode copy itself is internally
  consistent: layer-3 hidden output and layer-4 hidden operand both hash
  `0f5bb3244d887d40`, and layer-3 residual output and layer-4 residual operand
  both hash `aba23db132521334`. Full-prefill layer3-to-layer4 also
  self-matches at `0x2b97d8a595162835`. No behavior patch was accepted. The
  next boundary is layer2-to-layer3 handoff / layer2 output for decode step 1
  versus full-prefill prefix `[1044,1044]`. No HCS policy, routing,
  calibration/heatmap/safety, async stream, INT4 path, or protected
  `testconfigs` change was made.

- Added the Nano BF16 layer4-to-layer5 handoff gate after the `20:18`
  layer5-to-layer6 handoff result. Reverified the `20:18`
  artifacts/guardrails, async shared-stream absence, protected config
  cleanliness, and idle GPUs/processes. Used a fresh normal Nano BF16 server
  with no liveness timing, no graph override, no async stream patch, and
  calibration/heatmap/safety intact. Long calibration prefill min free was
  `14070 MB`; HCS soft-loaded `1230` experts / `23408.4 MB`;
  startup-ready free VRAM was `845 MB`, above the `600 MB` margin. Stable wrong
  clean output reproduced as `[1044,1044,1321,1321]`; full-prefill prefix
  `[1044,1044]` selected `[1044]`. Request-gated traces showed layer 4 is
  already entered with a different materialized norm input: decode hash
  `3c9dd32196ea5038`, l2 `2.6921605096`, versus full-prefill hash
  `0x2b97d8a595162835`, l2 `2.6930806637`. The layer4-to-layer5 decode copy
  itself is internally consistent: layer-4 hidden output and layer-5 hidden
  operand both hash `9d2014208552edd4`, and layer-4 residual output and
  layer-5 residual operand both hash `3c9dd32196ea5038`. Full-prefill
  layer4-to-layer5 also self-matches at `0x0bdaeef7579a106a`. No behavior patch
  was accepted. The next boundary is layer3-to-layer4 handoff / layer3 output
  for decode step 1 versus full-prefill prefix `[1044,1044]`. No HCS policy,
  routing, calibration/heatmap/safety, async stream, INT4 path, or protected
  `testconfigs` change was made.

- Added the Nano BF16 layer5-to-layer6 handoff gate after the `19:53`
  layer-6 upstream-input result. Reverified the `19:53` artifacts/guardrails,
  async shared-stream absence, protected config cleanliness, and idle
  GPUs/processes. Used a fresh normal Nano BF16 server with no liveness timing,
  no graph override, no async stream patch, and calibration/heatmap/safety
  intact. Long calibration prefill min free was `14070 MB`; HCS soft-loaded
  `1230` experts / `23408.4 MB`; startup-ready free VRAM was `845 MB`, above
  the `600 MB` margin. Stable wrong clean output reproduced as
  `[1044,1044,1321,1321]`; full-prefill prefix `[1044,1044]` selected
  `[1044]`. Request-gated decode and full-prefill traces showed layer 5 is
  already wrong before the layer5-to-layer6 handoff: decode step 1 layer-5
  materialized norm input hash `1957cd1ab25375a0`, l2 `2.9557667507`, versus
  full-prefill hash `0x0bdaeef7579a106a`, l2 `2.9565575123`. The
  layer5-to-layer6 decode copy itself is internally consistent: layer-5 hidden
  output and layer-6 hidden operand both hash `9141ff448cdc19d0`, and layer-5
  residual output and layer-6 residual operand both hash `1957cd1ab25375a0`.
  No behavior patch was accepted. The next boundary is layer4-to-layer5
  handoff / layer4 output for decode step 1 versus full-prefill prefix
  `[1044,1044]`. No HCS policy, routing, calibration/heatmap/safety, async
  stream, INT4 path, or protected `testconfigs` change was made.

- Added the Nano BF16 layer-6 upstream input/expert gate after the `18:59`
  accumulation-dtype result. Reverified the `18:59` artifacts/guardrails,
  async shared-stream absence, protected config cleanliness, and idle
  GPUs/processes. Used a fresh normal Nano BF16 server with no liveness timing,
  no graph override, no async stream patch, and calibration/heatmap/safety
  intact. Startup-ready free VRAM was `845 MB`, above the `600 MB` margin.
  Stable wrong clean output reproduced as `[1044,1044,1321,1321]`; the
  full-prefill prefix `[1044,1044]` still selected `[1044]`. Request-gated
  layer-6 diagnostics showed decode step 1 already differs before layer-6
  router/expert execution: rounded norm input hash `a54da934a1bb4ca6`, l2
  `3.8844289680` versus full-prefill l2 `3.8806655407`, and RMSNorm/router/
  expert input hash `bf02bce3e3c50e9c`, l2 `15.1732784512` versus full-prefill
  l2 `15.1858062744`. Selected dimensions show BF16-value deltas in the
  layer5-to-layer6 hidden/residual operands and norm output. Per-expert
  W1/activation/W2 investigation was stopped per the gate condition. No
  behavior patch was accepted. No HCS policy, routing, calibration/heatmap/
  safety, async stream, INT4 path, or protected `testconfigs` change was made.

- Added the Nano BF16 layer-6 decode accumulation-dtype gate after the
  `18:14` full-prefill-order patch. Reverified the `18:14` artifacts/source
  guardrails, async shared-stream absence, protected config cleanliness, idle
  GPUs/processes, and printed the existing Gemma record only (`5619.6`
  prefill, `92.43` internal decode, `155.69` HTTP). Added request-gated
  diagnostics only for layer-6 BF16 MoE accumulation dtype: decode BF16
  round-each-add replay, decode F32 replay, shared-add/post-layer summaries,
  and request-layer-selectable full-prefill sequential-MoE scatter replay.
  Fresh normal Nano BF16 startup completed with calibration/heatmap/safety
  intact and startup-ready free VRAM `845 MB`, above the `600 MB` margin.
  Patched clean controls reproduced the stable wrong output
  `[1044,1044,1321,1321]`; full-prefill for prefix `[1044,1044]` still
  selected `[1044]`. Layer-6 decode step 1 uses the proven prefill order
  `3,21,71,81,84,95`, and its live raw routed accumulation exactly equals the
  BF16 round-each-add replay (`118b5ad4bed8bd10`). However all six layer-6
  per-expert output hashes already differ from full-prefill, and route weights
  differ slightly before the routed-scale boundary, so a BF16 accumulation
  dtype mismatch is not proven. No behavior patch was accepted. No HCS policy,
  routing, calibration/heatmap/safety, async stream, INT4 path, or protected
  `testconfigs` change was made.

- Proved the actual Nano BF16 full-prefill layer-6 MoE accumulation order for
  the `17:14` target split and applied the corresponding narrow BF16 decode
  order patch. Reverified the `17:14` artifacts/source guardrails, async
  shared-stream absence, protected config cleanliness, idle GPUs/processes, and
  printed the existing Gemma record only (`5619.6` prefill, `92.43` internal
  decode, `155.69` HTTP). Fresh normal Nano BF16 servers ran with no liveness
  timing, no graph override, no async stream patch, and calibration/heatmap/
  safety intact; startup-ready free VRAM was `845 MB`, above the `600 MB`
  margin. The prepatch sequence reproduced fresh/correct
  `[1044,1044,1321,1044]` then warm/wrong `[1044,1044,1321,1321]`. The
  request-gated full-prefill trace for prefix `[1044,1044]` proved exact layer-6
  scatter row order `3,21,71,81,84,95` (`row` order
  `8,41,124,160,169,186`) with actual prefill accumulation dtype `f32`;
  canonical top-k/warm order is `95,21,71,3,84,81`, and fresh source-mode order
  is `95,21,71,84,81,3`. BF16 decode now defers routed expert adds and applies
  them in prefill-stable `(expert_id, topk_pos)` order before the shared add,
  independent of cold/HCS source mode. Patched requests are stable but still do
  not pass the `max_tokens=4` oracle: both clean controls returned
  `[1044,1044,1321,1321]`. No HCS policy, routing, calibration/heatmap/safety,
  async stream, INT4 path, or protected `testconfigs` change was made.

- Added the Nano BF16 layer-6 MoE accumulation-order canonicalization gate
  after the `16:13` warm-resident reuse split. Reverified the `16:13`
  artifacts/source guardrails, async shared-stream absence, protected config
  cleanliness, idle GPUs/processes, then added request-gated diagnostics only:
  BF16 decode accumulation-order replay for layer-selected MoE and
  selected-layer full-prefill sequential-MoE trace labels via
  `KRASIS_REFERENCE_MOE_TRACE_LAYERS`. Fresh normal Nano BF16 servers ran with
  calibration/heatmap/safety intact, no liveness timing, no graph override, and
  no async stream patch; startup-ready free VRAM was `845 MB`, above the
  `600 MB` margin. The same-build traces reproduced fresh/correct
  `[1044,1044,1321,1044]` and warm/wrong `[1044,1044,1321,1321]`. At the
  target layer 6 decode step 1, route IDs/weights match
  (`95,21,71,3,84,81`), and all six per-expert output hashes match between
  fresh and warm, including expert `3` cold-DMA versus resident-HCS
  (`9811d21ea0f974fc`). The split is only accumulation order: fresh/correct
  live source order is `95,21,71,84,81,3`, while warm/wrong live order is
  `95,21,71,3,84,81`. Canonical top-k order is also
  `95,21,71,3,84,81`, so canonical top-k is the warm/wrong path, not an
  evidence-backed patch target. The aligned full-prefill layer-6 control for
  prefix `[1044,1044]` exposes the same route IDs and selected token `[1044]`,
  but does not prove canonical top-k order. No behavior patch was accepted. No
  HCS policy, routing, calibration/heatmap/safety, async stream, INT4 path, or
  protected `testconfigs` change was made.

- Added the Nano BF16 warm-resident reuse downstream split gate after the
  `15:10` HCS promotion/resident-cache equivalence result. Reverified the
  `15:10` artifacts/source guardrails, async shared-stream absence, protected
  config cleanliness, idle GPUs/processes, then added request-gated
  layer-selected MoE diagnostics so `debug_decode_hcs_equiv_layer` can expose
  per-layer BF16 route/source/output detail beyond the default early layers.
  Fresh normal servers reproduced the pattern with calibration/heatmap/safety
  intact: first clean request selected `[1044,1044,1321,1044]`, later clean
  request selected `[1044,1044,1321,1321]`, and startup-ready free VRAM stayed
  `845 MB`, above the `600 MB` margin. Layer 6 is the first promoted layer
  with a concrete fresh-vs-warm output split. At decode step 1, route IDs and
  weights match exactly (`95,21,71,3,84,81`), and shared expert output matches
  (`201b327d8c2b8181`), but source mode changes from HCS
  `95,21,71,84,81` plus cold-DMA expert `3` to all-HCS
  `95,21,71,3,84,81`. The BF16 routed accumulation differs before shared add
  (`b51b29e497dd516e` vs `45325c5241d7464b`) and remains different after
  shared add (`4f36c6befe3d092c` vs `790b6b4287adbfbe`). Layer 6 step 2
  matches, while layer 8 shows a downstream source-mode split for promoted
  expert `48`. No behavior patch was accepted because the canonical BF16
  decode accumulation order versus full-prefill/HF is not yet proven. No INT4
  work, routing-policy change, calibration/heatmap/safety change, async stream
  reapplication, or protected `testconfigs` edit was made.

- Added the Nano BF16 fresh-first HCS dynamic promotion/resident-cache
  equivalence gate after the `14:44` request-order boundary. Reverified the
  `14:44` artifacts/source guardrails, async shared-stream absence, protected
  config cleanliness, idle GPUs/processes, then added request-gated HCS-only
  diagnostics for dynamic promotion commit content and resident expert content
  hashes. Fresh normal rebuilt server reproduced the pattern: first clean
  no-trace `max_tokens=4` request selected `[1044,1044,1321,1044]`, while the
  later clean request selected `[1044,1044,1321,1321]`; server HCS counters
  moved from `request_promotions=7/276` to `0/276`. A second fresh normal
  diagnostic server captured the first-request promotion commits: seven
  promotions from `ungraphed_decode_cold_dma_pingpong`, promoted layer/expert
  pairs `(22,83)`, `(22,37)`, `(29,38)`, `(6,3)`, `(31,15)`, `(6,102)`,
  `(8,48)`, slots `967,968,992,975,972,998,976`. For every promoted expert,
  source-device and host-source hashes matched the resident slot for W13/W2
  packed regions. Warm-state layer-1 and layer-6 HCS resident-content traces
  also had zero mismatched regions, including promoted layer-6 experts `3` and
  `102`. No HCS promotion copy/resident-content corruption was proven, so no
  behavior patch was made. No INT4 work, routing-policy change,
  calibration/heatmap/safety change, async stream reapplication, or protected
  `testconfigs` edit was made.

- Added the Nano BF16 graph/no-trace versus traced-decode gate after the
  `14:10` in-proj compute-equivalence result. Reverified the `14:10`
  artifacts, async shared-stream absence, protected config cleanliness, idle
  GPUs/processes, then ran a fresh normal Nano BF16 server with no liveness
  timing, no graph override, and no async stream patch. Normal startup stayed
  intact: long calibration `39920` tokens, prefill min free `14070 MB`,
  heatmap ranked `1657` experts, HCS soft-loaded `1230/2944`, and
  startup-ready free VRAM was `845 MB`, above the `600 MB` margin. The fresh
  run did not reproduce the prior graph/no-trace framing: CUDA graph replay is
  ruled out because traced decode reports `ungraphed_steps=3`,
  `per_layer_steps=0`, and `per_layer_graphs_valid_end=false`. The first clean
  no-trace request selected the full-prefill/HF token
  `[1044,1044,1321,1044]`; later traced, HCS-only, decode-state-only, and
  clean no-trace requests selected `[1044,1044,1321,1321]`. HCS-only and
  decode-state-only controls prove early layer readback synchronization is not
  required for the wrong token. Server HCS lines show the first correct
  request performed `request_promotions=6/276`, while later wrong requests
  had `decode_request_promotions=0` with resident hash
  `5d3c22ec143196bd` at decode start/end. No behavior patch was made; no
  INT4 work, routing-policy change, calibration/heatmap/safety change, async
  stream reapplication, CUDA graph/no-graph change, or protected
  `testconfigs` edit was made.

- Added the Nano BF16 layer-0 Mamba2 in-proj compute-equivalence gate after
  the `12:42` norm/in-proj boundary. Reverified the `12:42` artifacts, async
  shared-stream absence, protected config cleanliness, idle GPUs/processes, and
  ran a fresh normal graph-enabled Nano BF16 server with no liveness timing, no
  graph override, and no async stream patch. Added request-gated diagnostics
  that compute the same layer-0 in-proj input/weight rows through the decode
  single-vector BF16 GEMV path and a diagnostic forced batched BF16 GEMM path
  with `position + 1` repeated columns. The forced batched output matches
  full-prefill selected-row/manual-dot details for prefix
  `[1044,1044,1321]`, while decode GEMV differs by BF16 rounding on selected
  rows (`forced` l2 `289.3143109676`, full-prefill l2 `289.3143005371`,
  decode GEMV l2 `289.2909784021`). This does not prove GEMV is the token
  mismatch root cause: the request-gated decode trace still used production
  GEMV and selected the expected token at generated index `3`
  (`[1044,1044,1321,1044]`), while three no-trace controls on the same fresh
  server selected `[1044,1044,1321,1321]`. No behavior patch was accepted; no
  INT4 work, routing-policy change, calibration/heatmap/safety change, async
  stream reapplication, or protected `testconfigs` edit was made.

- Added the Nano BF16 layer-0 decode input-norm/in-proj gate after the
  `10:39` route-weight fix. Reverified the `10:39` artifacts, async
  shared-stream absence, protected config cleanliness, idle GPUs/processes, and
  ran a fresh normal graph-enabled Nano BF16 server with no liveness timing and
  no async stream patch. Added request-gated diagnostics only around layer-0
  decode residual/RMSNorm/in-proj plus matching full-prefill selected in-proj
  projection details. The server completed normal calibration/heatmap/HCS with
  startup-ready free VRAM `845 MB`, above the `600 MB` margin. For prefix
  `[1044,1044,1321]`, residual/input and RMSNorm output match full-prefill by
  aggregate (`3.9e-08` and `-8.9e-07` l2 deltas), and the in-proj input hash
  matches the full-prefill selected projection input. The first numeric split
  is layer-0 Mamba2 in-proj output: decode single-vector BF16 GEMV and
  full-prefill batched BF16 GEMM produce different BF16-rounded values from the
  same input/weight rows. A TensorOp algorithm experiment did not change the
  mismatch and was reverted. No behavior patch was accepted, and no INT4 work,
  routing-policy change, calibration/heatmap/safety change, async-stream
  reapplication, or protected `testconfigs` edit was made.

- Added the Nano BF16 extended-generation revalidation gate after the `08:52`
  layer-1 BF16 gather/cold-staging fixes. Reverified the `08:52` artifacts,
  async shared-stream absence, idle GPUs/processes, and protected config
  cleanliness, then started a fresh normal graph-enabled Nano BF16 server with
  no liveness timing and no async stream patch. Against the existing restored
  HF no-cache `04:23` `max_tokens=8` oracle
  `[1044,1044,1321,1044,1321,1044,1321,1044]`, the first corrected-38
  no-trace run matched through index `4` and then diverged. Request-gated
  decode diagnostics proved the early BF16 decode route weights were wrong:
  decode used the correction-biased selection scores as routed weights, giving
  near-uniform normalized weights, while prefill/HF semantics use
  `e_score_correction_bias` for top-k selection only and use raw
  `sigmoid(logit + bias)` values for the routed weights before normalization.
  Patched that decode weight-source formula and removed the unproven
  per-expert-scale hypothesis/plumbing. The route-weight fix changed layer-1
  decode weights from near-uniform to spread values matching the expected
  semantics, but extended generation still does not fully pass: stable repeats
  now match HF through `[1044,1044,1321]` and diverge at generated index `3`
  (`1321` vs HF/full-prefill `1044`). A matching full-prefill control for the
  same prefix selects `[1044]`; request-gated comparison shows embedding
  matches exactly and the first compared split is layer-0 Mamba2 in-proj, so
  the next gate should instrument the layer-0 decode input-norm/in-proj
  boundary before any further patch. No INT4 work, calibration/heatmap/safety
  change, async-stream reapplication, or protected `testconfigs` edit was made.

- Fixed the Nano BF16 layer-1 sequential MoE/gather-scatter nondeterminism
  gate. Request-gated diagnostics on repeated identical prefill-only
  `prompt + 1044`, `max_tokens=1` requests first proved the layer-1
  router/top-k outputs were stable, but GPU atomic map construction produced
  nondeterministic within-expert gather row order. Added a BF16-only stable
  `moe_build_maps_stable_kernel`; non-BF16/INT4 map construction is unchanged.
  A follow-up batch showed maps and gathered rows were stable, but cold BF16
  experts still produced differing outputs from identical gathered inputs while
  cached experts were stable. The cold path was using synchronous pageable
  `cuMemcpyHtoD_v2`, which is not ordered by `copy_stream` waits before
  reusing the two staging buffers. Added a BF16-only host wait on the recorded
  main-stream compute event before reusing a cold staging buffer. Fresh patched
  server startup stayed healthy: long calibration `39920` tokens in `286.87s`
  (`139.2 tok/s`), heatmap ranked `1639` experts, HCS soft-loaded
  `1230/2944`, startup-ready free VRAM stayed above the `600 MB` safety
  margin, and no async shared-stream patch was reapplied. Four traced requests
  were bit-stable through map/gather/per-expert output/routed accumulation/
  shared add/final logits, and four no-trace controls all returned `[1044]`
  with identical top-1 logprob. No routing-policy, calibration/heatmap/safety,
  async-stream, protected `testconfigs`, or INT4 behavior change was made.

- Added the Nano BF16 layer-2 Mamba2 prefill SSM producer gate after the
  `07:15` cross-request/HCS localization. Reverified the `07:15` artifacts,
  async shared-stream absence, protected config cleanliness, idle
  GPUs/processes, and then ran a fresh normal graph-enabled Nano BF16 server
  with normal calibration, heatmap, safety, and HCS intact. Added
  request-gated layer-2 Mamba2 full hidden-input and full in-proj snapshots
  only. Repeated identical prefill-only `prompt + 1044`, `max_tokens=1`
  requests reproduced output variation on the fresh server (`[1044]`,
  `[1044]`, `[1321]`, `[1044]`). The layer-2 SSM output and terminal SSM
  state matched the per-request CPU recompute, so no layer-2 SSM scan,
  terminal-write, registered-state copy, buffer-init, layout, or stream bug was
  proven. The first visible same-request-shape split is upstream:
  `layer1_sequential_moe_gather_weight_map`, followed by layer-1 routed/shared
  accumulation and then differing `layer2_mamba2_input_hidden_full`. No
  behavior patch, INT4 work, routing-policy change, calibration/heatmap/safety
  change, async-stream reapplication, or protected `testconfigs` edit was made.
  The next target is layer-1 BF16 sequential MoE/gather-scatter
  nondeterminism.

- Added the Nano BF16 cross-request nondeterminism/HCS gate after the `06:24`
  multi-step state-carry gate. Reverified the `06:24` artifacts, async
  shared-stream absence, protected config cleanliness, idle GPUs/processes,
  and printed the existing Gemma speed record only (`5619.6` prefill,
  `92.43` internal decode, `155.69` HTTP; not rerun). Ran repeated identical
  `prompt + 1044`, `max_tokens=3` requests on one ready Nano BF16 server and
  again after a fresh server restart with normal calibration/heatmap/HCS
  intact. Identical requests still varied after restart, so this is not only
  persistent server-state drift from earlier traces. HCS resident hashes and
  promotions correlate imperfectly but are not sufficient as the root cause:
  same-hash traces can diverge and output changes can occur without request
  promotions. Request-gated layer-2 lifecycle diagnostics show layer-2 Mamba2
  state is zero at request start and after cleanup, so it is not simple stale
  decode-state leakage. The first same-token stage split is layer-2 Mamba2
  state/SSM produced during prefill: layer-2 conv state is stable in the final
  lifecycle traces, but layer-2 SSM state differs before decode begins. No
  behavior patch, INT4 work, routing-policy change, calibration/heatmap/safety
  change, async-stream reapplication, or protected `testconfigs` edit was made.

- Added the Nano BF16 multi-step decode-state gate for the post-`04:23`
  extended-generation issue. Request-gated decode state snapshots now compare
  step-0-after-state to step-1-before-state across registered Mamba2 conv/SSM
  buffers, GQA KV carry rows, and HCS/MoE resident-cache summary. On the
  current rebuilt server, the old `04:23` full-prefill control
  `prompt + 1044 + 1044 -> [1321]` did not reproduce: the exact old payload
  now returns `[1044]`, matching the minimal `prompt + 1044`, `max_tokens=2`
  second token `[1044]`. Across three traced `max_tokens=3` runs, all 58
  step-0-after versus step-1-before state fields matched by hash with zero
  mismatches, so the requested Mamba2/attention state carry boundary is not the
  current split. Repeated identical `max_tokens=3` requests remain
  non-deterministic (`[1044,1294,1321]`, `[1044,1044,1044]`,
  `[1044,1294,1044]`, plus no-trace variants `[1044,1294,1294]` and
  `[1321,1044,1044]`), while HCS resident hashes/promotions differ between
  requests. No behavior patch, INT4 work, routing-policy change,
  calibration/heatmap/safety change, async-stream reapplication, or protected
  `testconfigs` edit was made.

- Added the Nano BF16 extended-generation validation gate after the token-2
  scaling-order fix. A restored-HF no-cache corrected-38 oracle with
  `max_tokens=8` produced `[1044,1044,1321,1044,1321,1044,1321,1044]`
  (`",, and, and, and,"`). Krasis BF16 graph-enabled no-trace runs still
  diverged from token index `2` and showed near-tie/non-deterministic behavior
  around `1044`, `1321`, and `1294`; `KRASIS_NO_GRAPH=1` also diverged at
  token index `2` (`[1044,1044,1044,1044,1044,1044,1044,1321]`), ruling out
  graph replay as the sole root cause. Request-gated layer-3 diagnostics were
  added only for localization; they show BF16 MoE metadata/path is sane under
  trace, with routed scaling before shared add still intact. No behavior patch,
  routing-policy change, calibration/heatmap/safety change, INT4 work,
  async-stream reapplication, or protected `testconfigs/` edit was made.
  Note: the restored-HF oracle JSON has one diagnostic inconsistency at token
  index `2` where the generated token is `1321` but the captured top-k ranks
  `1044` first, so the generated sequence was used as the requested oracle
  while that diagnostic mismatch was recorded.

- Fixed the Nano BF16 token-2 downstream decode scaling-order divergence.
  Request-gated diagnostics showed layer-1 decode had the correct BF16
  nonuniform route and expert outputs, but combined the shared expert before
  applying `routed_scaling_factor`, effectively scaling `routed + shared`.
  Prefill/Python semantics scale only the routed branch before adding shared.
  Decode MoE paths now apply `routed_scaling_factor` before the shared expert
  add when a shared expert is present, and skip final whole-output scaling in
  that case. Patched Nano BF16 no-trace and decode-trace corrected-38 now match
  the restored HF full-generation oracle for both generated tokens:
  `[1044,1044]` (`",,"`). Startup reached readiness without liveness timing or
  async stream binding; long calibration was `39920` tokens in `286.02s`
  (`139.6 tok/s`), heatmap ranked `1625` experts, HCS soft-loaded
  `1230/2944` experts, and request decode min free was `846 MB` above the
  `600 MB` margin. No INT4 work, routing-policy change, calibration/heatmap/
  safety change, async-stream reapplication, or protected `testconfigs/` edit
  was made.

- Fixed the Nano BF16 layer-1 decode MoE metadata/uniform-route blocker.
  Decode now keeps BF16 routed/shared experts on the BF16 cuBLAS path instead
  of INT4 Marlin `relu2_w2_*` labels, and non-swiglu
  `e_score_correction_bias` is registered as `e_score_corr_ptr` instead of a
  pre-sigmoid `gate_bias_ptr`. Unproven BF16 pinned-readback and gate-stream
  sync hypotheses were reverted before the final run. Nano BF16 reached
  readiness with the async shared-stream binding still absent; after the
  correction-pointer fix, heatmap ranked `1162` experts and layer-1
  corrected-38 decode selected nonuniform experts `18,39,7,87,24,110` with
  BF16 `bf16_cublas_relu2_accum` metadata instead of uniform `0..5` INT4
  routing. Full generation still diverges at generated token index `1`: HF
  restored oracle `[1044,1044]` (`",,"`) versus Krasis `[1044,1321]`
  (`", and"`). No routing policy, calibration, heatmap, safety, async-stream,
  INT4-path, or protected `testconfigs/` change was made beyond preserving the
  BF16 metadata/dispatch fix and request-gated diagnostics.

- Fixed the Nano BF16 Mamba2 prefill-to-decode state handoff and localized
  the remaining first-decode-step divergence. Registered Mamba2 decode
  conv/SSM buffers are now passed into Rust prefill, zeroed once at fresh
  prefill start, used as recurrent Mamba2 state during prefill, and the
  terminal conv state is updated by a dedicated kernel. Nano BF16 reached
  readiness with the async shared-stream patch still absent: long calibration
  `39920` tokens in `286.09s` (`139.5 tok/s`), HCS soft-loaded `1230/2944`
  experts (`23408.4 MB`), and startup-ready free VRAM `845 MB` above the
  `600 MB` margin. Layer-0 registered Mamba2 state now becomes nonzero
  immediately after prefill and remains populated before decode
  (`conv_l2=422.8540`, `ssm_l2=9378.4328`). Corrected-38 still diverges at
  generated token index `1`: restored HF `[1044,1044]` (`",,"`) versus
  Krasis `[1044,91605]` (`", Margare"`). A `prompt + 1044` full-prefill
  diagnostic returns `[1044]`, matching HF's second token, while incremental
  decode returns `91605`; sampler and HCS cold-load are ruled out. Layer-0
  incremental decode matches full prefill by aggregate, and the first clear
  remaining bad path is layer-1 decode MoE/expert metadata selecting INT4
  `relu2_w2_*` kernels with uniform experts `0..5` in this BF16 correctness
  config. No INT4 work, async stream reapplication, routing, expert,
  calibration, heatmap, safety, or protected `testconfigs/` change was made
  beyond the Mamba2 handoff fix and request-gated diagnostics.

- Added a Nano BF16 full-generation correctness gate beyond first-token
  prefill. Reverified the `23:48` shared-expert fix state, the async
  shared-stream revert, and protected config cleanliness, then generated a
  restored-HF oracle for the same corrected-38 raw payload with `max_tokens=2`
  and `top_k=1`. Archived HF cached generation remains broken for this
  model, so the successful oracle used full generation with
  `--generate-use-cache off`, not prefill-only. HF produced `[1044,1044]`
  (`",,"`), while Krasis Nano BF16 no-trace and request-gated decode trace
  produced `[1044,91605]` (`", Margare"`). The split is the first decode
  step. Sampler was ruled out because each side selected its own rank-1
  token; HCS/cold-load effects were ruled out because the decode trace had
  `unique_cold_experts=0`, `cold_load_count=0`, no DMA, and no pending reload
  at decode start. Layer-0 Mamba2 lifecycle diagnostics show registered
  decode conv/SSM state stays at zero through prefill and before decode, then
  becomes nonzero only after decode (`conv_l2=238.6501`,
  `ssm_l2=1092.1982`). No INT4 work, async stream reapplication, routing,
  calibration, heatmap, safety, or protected `testconfigs/` change was made.

- Fixed the Nano BF16 layer-1 shared-expert correctness blocker after the
  post-revert startup/liveness gate. With liveness timing disabled, Nano BF16
  reached readiness and corrected-38 no-trace/device requests both selected
  first token `[1044]` (`,`), matching the restored HF prefill oracle. The
  failing shared path was a metadata/registration bug: direct BF16 shared
  weights were misclassified as INT8 Marlin shared descriptors because size
  inference ran before Nemotron `gated=false` semantics were known. Internal
  MoE registration now passes explicit loaded shared-expert bits from
  `UnifiedExpertWeights`; legacy public registration still uses size inference
  only when metadata is unavailable. Patched shared diagnostics show
  `config_shared_expert_bits=16`, BF16 W1/W2 present, Marlin W1/W2 absent,
  no shared gate, and shared W1 input/output, activation/W2 input, W2 output,
  and shared add input matching the restored HF `1217` oracle within
  `4.5e-7` l2-summary delta. Raw safetensor row hashes for layer-1 shared W1
  and W2 row samples match the Krasis trace. No async shared-stream patch was
  reapplied, no liveness timing was enabled, and calibration/heatmap/safety,
  routing, INT4, and protected `testconfigs/` were not changed.

- Added a Nano BF16 startup/liveness diagnostic after the corrective async
  shared-stream revert. Reverified the reverted `set_stream(shared_stream)`
  binding and liveness metadata were still absent, then added startup-gated
  prefill liveness markers around calibration chunks, layer boundaries,
  layer-1 BF16 sequential MoE, routed cold experts, scatter, shared add, and
  heatmap prefills. The instrumented Nano BF16 run reached
  `KRASIS SERVER READY` without reapplying the async stream patch and without
  bypassing calibration, heatmap, or safety. Long calibration completed:
  `39920` tokens in `284.07s` (`140.5 tok/s`) with post-calibration free VRAM
  `24626 MB`. Layer-1 long calibration completed with `128` cold BF16 experts,
  `239520` routed activations, `9732.949 ms` scatter sync, `9.431 ms` shared
  add sync, and `10006.075 ms` total layer time. Heatmap completed all six
  held-out prompts, ranked `138` experts, soft-loaded `1230/2944` experts
  (`23408.4 MB`), and startup-ready free VRAM was `845 MB`, above the
  `600 MB` safety margin. No corrected-38 or other correctness request was
  sent.

- Reverted the unproven Nano BF16 async shared-stream cuBLAS binding from
  `src/gpu_prefill.rs` after review. The direct BF16-not-Marlin routed
  dispatch and request-gated shared diagnostics remain in place, but the
  `set_stream(shared_stream)` call in the BF16 async shared helper and its
  liveness-only metadata were removed because no corrected-38 request or
  shared W1/W2/add comparison proved a correctness fix. The current blocker
  remains the previously observed post-revert Nano BF16 startup blocker:
  removing the binding blocked in long calibration. This corrective pass did
  not start another Nano run; validation was static only (`./dev build`,
  `bash -n dev`, `./dev python -m py_compile`, `git diff --check`, protected
  config status, and source-scope/process/GPU checks).

- Added a narrow Nemotron Nano BF16 layer-1 shared-expert gate after the
  `1044` layer-1 localization. Reverified the `1044` artifacts and source
  diff first, keeping the direct BF16-not-Marlin routed dispatch because the
  prior diagnostics prove routed expert rows are finite/non-zero, while
  marking the async shared-stream correctness patch unproven because it did
  not improve first token or the post-shared explosion. Added restored-HF
  diagnostics for shared W1 input/weight/output, activation, W2
  input/weight/output, add inputs, and branch metadata; added matching
  request-gated Krasis shared-stage diagnostics. HF restored oracle completed
  and reports no shared gate, `NemotronHMLP`, shared W2/output l2
  `4.217198`, and first token `[1044]` (`,`). Krasis did not reach readiness,
  so no corrected-38 shared W1/W2/add comparison was collected. Three startup
  attempts were preserved: with stream binding present, long calibration
  completed but heatmap did not finish; after reverting the binding, startup
  blocked in long calibration; after restoring the binding with metadata
  saying it is required for async BF16 liveness and not a proven correctness
  fix, startup again blocked in long calibration. No calibration, heatmap,
  safety, routing, fallback, INT4, or `testconfigs/` change was made.

- Added a Nemotron Nano BF16 layer-1 correctness gate after the `1007`
  post-SSD layer-0 fix. Reverified the patched artifacts/source state first:
  restored HF and Krasis match through layer-1 input, and corrected-38 still
  mismatches later (`HF [1044]` / `,`, Krasis `[1586,91605]` /
  `ip Margare`). Existing Gemma speeds were reported from records only:
  `5619.6` prefill, `92.43` internal decode, `155.69` HTTP. Added
  request-gated layer-1 BF16 diagnostics and, based on the layer-1 evidence,
  narrowly changed BF16 prefill MoE to avoid the fused Marlin path and use
  sequential BF16 cuBLAS, with an explicit unsupported error for latent BF16
  sequential mode. Also bound the BF16 async shared cuBLAS handle to the
  shared stream and added request-gated layer-1 shared dispatch/sync trace
  metadata. Reruns still mismatch first token. Layer-1 input and input norm
  match restored HF (`l2 1.014168` and `11.958818`), and router/top-k selects
  the same expert set and weights by expert id. The completed-run explosive
  boundary is after shared expert/add: routed accumulation before shared is
  finite (`l2 18.811044`), but total accumulation after shared reaches
  `l2 32182593.155` with values from `-1081344` to `1900544`, while restored
  HF layer-1 shared output is small (`l2 4.217198`). A sync shared diagnostic
  attempt completed long calibration but did not reach readiness because
  startup stayed at heatmap build; it was stopped with `./dev kill`. No INT4
  work, correction-bias routing reapplication, calibration/safety change,
  fallback, or `testconfigs/` edit was made.

- Fixed the next Nemotron Nano BF16 layer-0 post-SSD correctness split after
  the restored `dt_bias` oracle gate. Reverified the `0934` artifacts/source
  state, reused/requested layer-0 diagnostics for corrected-38, and proved
  `D*x`, SSD output, and gated norm already matched while Mamba2 `out_proj`
  split. The projection input and GEMM math matched; the source mismatch was
  that HF runtime uses `out_proj.weight / sqrt(num_hidden_layers)` when
  `rescale_prenorm_residual=true`, while Krasis loaded raw safetensor rows.
  Added config parsing for `rescale_prenorm_residual` and applied that
  model-derived scale to Mamba2 `out_proj.weight` during weight loading. After
  `./dev build` and Nano BF16 corrected-38 reruns, layer-0 `D*x`, SSD output,
  gated norm, out-proj, mixer output, final layer-0 output, and layer-1 input
  all match restored HF by l2 delta `0.0`; selected out-proj output bits and
  weight row hashes match for dims `0,1,2,3,1024,1328,2047,2686,2687`.
  First-token correctness still fails later: HF restored oracle `[1044]`
  (`,`), Krasis patched no-trace/device `[1586,91605]` (`ip Margare`). No
  INT4 work, routing, expert, calibration, safety, fallback, or
  `testconfigs/` change was made.

- Added a narrow Nemotron Nano BF16 `dt_bias` source gate after the `0900`
  layer-0 SSD diagnostic. Reverified the `0900` artifacts, then instrumented
  `tests/generate_reference.py` to dump HF
  `backbone.layers.0.mixer.dt_bias` immediately after `from_pretrained`,
  immediately before the hooked `_chunk_cumsum_fwd` call, and from the raw
  safetensor. The HF oracle was wrong: post-load `dt_bias` did not match the
  checkpoint (`l2 37.3715`, sha `8a31...` versus raw safetensor
  `l2 64.9839`, sha `e6e0...`), and the prior `0900` oracle had a third
  all-negative value (`l2 40.8732`). Fixed only the HF reference harness by
  restoring `23` Nemotron-H `mixer.dt_bias` tensors from safetensors before
  oracle execution; the pre-cumsum HF parameter and `_chunk_cumsum_fwd`
  argument now match raw safetensor exactly, while the reverified Krasis trace
  already matched raw safetensor aggregates. With the restored oracle, layer-0
  SSD output now matches by l2 (`3984.1997` both sides), proving the stale SSD
  split was an oracle artifact. Corrected-38 still fails later: HF restored
  oracle `[1044]` (`,`), Krasis no-trace/device `[1586,91605]`
  (`ip Margare`). No SSD, routing, expert, calibration, safety,
  production-runtime, fallback, or `testconfigs/` change was made.

- Added a diagnostic-only Nemotron Nano BF16 layer-0 Mamba2 SSD gate after
  the `0818` BF16 baseline. Reverified the Nano BF16 artifacts/source state,
  then ran a corrected-38 layer-0 device trace with existing request-gated SSD
  diagnostics. No INT4 work, routing/expert/calibration/safety behavior
  change, fallback, or `testconfigs/` edit was made. Classification: the
  first split is before SSD recurrence math, at the `dt_bias` vector supplied
  to the scan path. Raw `dt` still matches exactly (`l2 12.20384693145752` on
  both sides), but `dt_bias` differs before softplus/cumsum/scan (`HF l2
  40.8731536865`, Krasis/checkpoint l2 `64.9838638306`; head 0 HF `-2.3125`
  vs Krasis `-1.0`; head 31 HF `-2.453125` vs Krasis `15.75`). Downstream
  SSD output then diverges (`HF l2 948.22`, Krasis l2 `3984.20`). No SSD
  formula/kernel mismatch was proven, so no SSD patch was made. Next target is
  the Nano HF oracle/load contract for `layer0.mixer.dt_bias`, because Krasis
  is using checkpoint `backbone.layers.0.mixer.dt_bias` while the HF prefill
  oracle reports a different all-negative vector.

- Added a Nemotron Nano BF16 correctness baseline for the staged Nano-first
  plan. Created `tests/nemotron-nano-bf16-experts-a16.conf`, built through
  `./dev build`, launched one Nano BF16 `--test-endpoints` server in tmux, and
  ran corrected-38 no-trace/device-trace requests plus an HF prefill-logit
  oracle through `./dev generate-reference`. Nano BF16 now fits and reaches
  readiness: long calibration passed at `686.8 tok/s`, post-calibration free
  VRAM was `24582 MB`, HCS soft-loaded `1230/2944` experts (`23408.4 MB`),
  and startup-ready free VRAM was `916 MB` after measured pressure eviction.
  Correctness is not yet passing: Krasis returned first token `1586`
  (`ip`), while HF prefill logits select token `1044` (`,`). Existing traces
  localize the earliest visible split to layer-0 Mamba2 SSD output: input,
  RMSNorm, in-proj, raw x/B/C/dt, and causal-conv x/B/C match by stats, then
  `layer0_mamba2_ssd_out_last` differs (`HF l2 948.22`, Krasis l2
  `3984.20`). No fallback, production behavior patch, safety-margin
  reduction, calibration bypass, GPU-prefill disablement, or `testconfigs/`
  edit was made.

- Added a BF16 startup measured cold-reserve fix candidate after the `1636`
  formula diagnostic. Reverified the prior formula artifacts, then patched the
  prefill path to stop using the all-expert pre-calibration cold-staging floor
  as the prepare reserve, to report measured cold-staging capacity before an
  impossible allocation, and to retry calibration prefill with a chunk cap
  derived from actual routed cold-slot demand. Built through `./dev build` and
  ran both BF16 startup calibration configs in tmux with
  `KRASIS_VRAM_LEDGER=1`, `KRASIS_PREFILL_DEBUG=1`,
  `KRASIS_STARTUP_DIAG=1`, and `KRASIS_STARTUP_EXIT_AFTER_CALIBRATION=1`.
  The candidate replaced the old `22272 MB` prepare-floor failure with
  measured route-demand failures: routed-BF16 retried `500 -> 229 -> 128` but
  still needed `490` cold BF16 expert slots at the minimum `125`-token planned
  chunk versus `242` measured safe slots; all-BF16 retried
  `500 -> 192 -> 128` and still needed `490` slots versus `204` safe slots.
  Warmup succeeded because the repeated-token warmup used only `43` cold
  slots. Classification: this fixes the formula/measurement problem but does
  not make BF16 Nemotron ready; the next blocker is BF16 expert residency/HCS
  or another measured residency strategy before calibration. No safety-margin
  reduction, calibration skip, GPU-prefill disablement, fallback, routing
  reapplication, or `testconfigs/` edit was made.

- Added a diagnostic-only Nemotron Super BF16 prefill VRAM formula gate after
  the `1603` BF16 expert baseline. Reverified the `1603` artifacts/source
  state, then reran both BF16 expert startup attempts through `./dev` with
  `KRASIS_VRAM_LEDGER=1`, `KRASIS_PREFILL_DEBUG=1`, and
  `KRASIS_STARTUP_DIAG=1`. No production behavior patch, fallback, safety
  margin change, GPU-prefill disablement, calibration bypass, or
  `testconfigs/` edit was made. Classification: the `22272 MB` prefill floor
  is not a measured BF16 prefill transient; it is the pre-calibration
  post-scratch reserve computed as `21672 MB` max cold staging
  (`512` routed experts at `42.328 MiB` each) plus the `600 MB` safety
  margin. Actual minimum scratch allocation before the guard fires was about
  `944 MB` in both configs. The all-BF16 config failed with `9264 MB` free
  after minimum scratch; the routed-BF16 comparable config failed with
  `10864 MB`. Next work should measure active routed expert/cold-slot demand
  before applying the 512-expert BF16 reserve, without reducing the safety
  margin or disabling GPU prefill.

- Added a diagnostic/baseline Nemotron Super BF16 expert gate after the
  `1547` oracle/config contract correction. Reverified that
  `tests/nemotron-super-bf16kv-a16.conf` is intentionally cached Marlin INT4
  experts, then added tests-only BF16 expert configs and validated the direct
  BF16 expert path. The no-server validation preflight now classifies
  `CFG_GPU_EXPERT_BITS=16` plus BF16 attention/shared/dense/lm-head surfaces
  as BF16, and the server CLI accepts/logs GPU BF16 experts. A one-layer
  loader smoke passed (`512` experts, `5376 MB`, `7.679s`), and two full
  tmux server attempts loaded all `20480` routed BF16 experts plus shared
  layers. Both failed before readiness during measured VRAM calibration:
  all-BF16 surfaces had `9264 MB` free after minimum scratch versus
  `22272 MB` required, while the comparable routed-BF16 config had
  `10864 MB` free versus `22272 MB` required. Corrected-38/HF comparisons
  were skipped because no server reached readiness. No production routing
  patch, fallback, or `testconfigs/` change was made.

- Added a diagnostic-only Nemotron Super oracle/config contract check after
  the `1525` W1 weight/dequant source gate. Reverified the `1525` artifacts
  and reverted raw-sigmoid routing source first, then inspected
  `tests/nemotron-super-bf16kv-a16.conf`, config parsing, and expert cache
  load selection without starting a server. Classification:
  `nemotron-super-bf16kv-a16.conf` is intentionally BF16 attention/KV with
  cached Marlin INT4 routed experts, not true BF16 expert weights. The config
  header says “BF16 attention/KV, INT4 experts”, sets
  `CFG_GPU_EXPERT_BITS=4` and `CFG_CPU_EXPERT_BITS=4`, and maps to the
  existing cache
  `experts_marlin_int4_g128_calamax.bin`. True BF16 expert loading requires
  `gpu_expert_bits=16`, which is explicitly an unvalidated debug-only path.
  Result: HF BF16 bit equality remains useful before the expert-weight
  quantization boundary, but must not be treated as the post-expert acceptance
  oracle for this config. Future routed-expert comparisons need an INT4-aware
  target based on the same cached Marlin weights/scales, routing weights, and
  production accumulation/cast semantics. No production behavior patch,
  fallback, full server run, or `testconfigs/` change was made.

- Added a diagnostic-only layer-1 W1 weight/dequant source comparison after
  the `1441` W1/up-proj localization. Reverified the `1441` artifacts and the
  reverted raw-sigmoid routing source first, then compared original safetensors
  BF16 up-proj rows against the cached Marlin INT4/dequant rows for experts
  `236`, `216`, `44`, `250`, `382`, `28`, and `473` across dim `0` and
  sentinel rows. The offline inverse was corrected to match
  `src/weights/marlin.rs` exactly; a provisional parser result was discarded.
  Final evidence: cache layout matches the expected W1 shape, group size is
  `128`, scale handling is BF16 `amax/7`, zero handling is symmetric U4B8 with
  no zero tensor, and all 49 compared rows have `0` scale mismatches and `0` q
  mismatches versus expected symmetric INT4 amax quantization. Classification:
  the W1 split is expected INT4 quantization/dequantization error versus HF
  BF16 weights, not a cache packing, scale, zero, expert-ID, or layout bug in
  the checked rows. No production behavior patch, correction-bias routing
  reapplication, fallback, or `testconfigs/` change was made.

- Added a diagnostic-only layer-1 W1/up-proj localization pass after the
  `1420` routed expert math split. Reverified the `1420` artifacts and the
  reverted raw-sigmoid prefill routing source first, then added only
  request-gated selected-expert diagnostics for top-delta experts `236`,
  `216`, `44`, `250`, `382`, plus boundary experts `28` and `473`. Krasis now
  reports selected-expert W1/up-proj input hashes, Marlin INT4 layout/dequant
  candidates, scale metadata, production W1 bits, and dim-0 contribution
  details; HF reports corresponding BF16 W1/up-proj manual-dot details. An
  initially over-broad HF diagnostic was killed through `./dev kill`, narrowed
  to selected rows/dims, and rerun successfully. Classification: pre-MoE input
  and raw-route expert set match, transpose/layout is ruled out, and HF BF16
  manual dots match HF actual output. The split is from the quantized INT4
  Marlin W1/dequant weight path versus HF BF16 expert weights, with
  accumulation/order and output rounding downstream. No production behavior
  patch, correction-bias routing reapplication, fallback, or `testconfigs/`
  change was made.

- Added a diagnostic-only layer-1 routed expert math split after the `1343`
  MoE contribution diagnostic. Reverified the `1343` artifacts and reverted
  raw-sigmoid prefill routing source first, then mined existing Krasis device
  trace stages before adding only request-gated HF per-expert stage
  diagnostics. HF now reports raw-route W1/up-proj, activation, W2/down-proj,
  and W2 dim-0 contribution details for the selected raw-route experts; Krasis
  already had matching stage hashes from the `1343` device trace. No
  production route patch, fallback, full Krasis server rerun, or
  `testconfigs/` change was made. Classification: the active raw route expert
  set and weights are effectively the same, and Krasis routed-scatter
  recompute matches actual (`max_abs=5.59e-09`), so latent accumulation/order
  is not the producer. All 22 compared experts differ already at W1/up-proj
  output, with activation and W2 downstream. Latent-up projection is
  downstream because its input BF16 row already differs. The next target is
  the per-expert W1/up-proj / quantized expert matmul input path.

- Added a diagnostic-only layer-1 MoE contribution split after the rejected
  and reverted `1256` prefill `e_score_correction_bias` candidate. Reverified
  the `1256` artifacts and current reverted runtime first, then added
  request-gated contribution diagnostics only: Krasis now reports latent-MoE
  routed-scatter per-slot dim contributions with the correct latent scatter
  width, and HF reports corrected-route versus raw-route per-expert
  contributions for experts `28` and `473`. Exact corrected-38 no-trace and
  device-trace requests ran on one `--test-endpoints` server, then HF ran
  through `./dev generate-reference`. Classification: matching the selected
  expert set is necessary but not sufficient. HF raw-to-corrected routing moves
  routed hidden dim `0` only from `0x3c69` to `0x3c6c`, while Krasis is still
  much lower (`0x3c54` current raw route, `0x3c56` in the rejected corrected
  candidate). Shared output dim `0` matches (`0x3cae`), so the dominant
  producer is routed expert/latent-up math before shared add, not router
  selection alone. No production routing patch, fallback, or `testconfigs/`
  change was made.

- Rejected a narrow production patch candidate for Nemotron Super prefill
  `e_score_correction_bias` routing. Reverified the `1206` routing artifacts
  first, then implemented general prefill sigmoid top-k correction-bias
  selection plumbing: select on `sigmoid(logit) + bias` while preserving
  uncorrected sigmoid weights for normalization/scatter scale. Built through
  `./dev build`, ran exact corrected-38 no-trace/device requests on one
  `--test-endpoints` server, ran three warmed no-debug speed samples, then ran
  HF through `./dev generate-reference`. The candidate fixed the layer-1
  expert set for the boundary: expert `28` was excluded and expert `473` was
  selected, matching HF. It failed acceptance because layer-1 branch dim `0`
  did not improve (`0x3d12` HF vs `0x3d0c` Krasis) and layer-1 output dim `0`
  stayed `0x3c42` HF vs `0x3c2a` Krasis. The candidate was reverted and
  rebuilt back to the prior accepted runtime. No fallback or `testconfigs/`
  change was made.

- Added a diagnostic-only layer-1 router selection semantics split after the
  `1128` MoE producer gate. Reverified the top-k artifacts first, inspected
  HF's actual Nemotron gate path and Krasis prefill routing, then added
  request-gated diagnostics around boundary experts `28` and `473` plus
  neighboring scores/weights. Exact corrected-38 no-trace/device requests ran
  on one `--test-endpoints` server and HF ran through
  `./dev generate-reference`. Classification: HF computes FP32 router logits,
  applies sigmoid, selects on `sigmoid(logit) + e_score_correction_bias`, then
  gathers uncorrected sigmoid scores for normalization/scatter scale. Krasis
  prefill `sigmoid_topk_kernel` has no correction-bias ABI and observed
  `moe_e_score_corr_ptr=0x0`, so it selects raw sigmoid expert `28` and
  excludes HF-selected expert `473`. Grouping, score dtype, sigmoid,
  renormalization, selected order, and expert math are not the first producer.
  No behavior patch, fallback, or `testconfigs/` change was made.

- Added a diagnostic-only layer-1 MoE producer split after the post-`0939`
  handoff rebaseline. Reverified the `1012` artifacts first, then added
  request-gated HF selected-row diagnostics for layer-1 MoE router/top-k and
  branch stages; existing Krasis reference trace MoE stage coverage was
  sufficient. Exact corrected-38 no-trace/device requests ran on one
  `--test-endpoints` server and HF ran through `./dev generate-reference`.
  The layer-1 pre-MoE router-input BF16 row hash matches HF exactly
  (`0x6637fa373508e183`), and Krasis matches HF's global router-logit top-22
  rank order. The first producer is routing selection semantics: HF actual
  `get_topk_indices` selects expert `473` and excludes expert `28`, while
  Krasis selects global-router-logit expert `28` and excludes `473`.
  Expert-input dim `0` still matches (`0xbee1`), shared output dim `0`
  matches (`0x3cae`), but routed latent/output and final branch dim `0`
  differ downstream. No behavior patch, fallback, or `testconfigs/` change
  was made.

- Added a diagnostic-only post-`0939` Nemotron Super first-token rebaseline.
  Kept the accepted inline shared/local `__nv_logf(1.0f + __expf(x))` SSD
  softplus patch in place, reverified its `6733`/`4025`/`1328` artifacts, then
  added request-gated full-row layer-1 handoff BF16 bit diagnostics on both
  Krasis and HF. Exact corrected-38 no-trace/device requests ran on one
  `--test-endpoints` server and HF ran through `./dev generate-reference`
  with cache disabled. Layer-0 handoff remained exact; the next real split is
  layer-1 handoff dim `0`, where residual matches but the layer-1 branch/MoE
  output differs (`0x3d12` HF vs `0x3d0c` Krasis), producing rounded output
  `0x3c42` HF vs `0x3c2a` Krasis. Residual mismatch count was `0`; branch
  mismatch count was `4024`; output mismatch count was `3981`. No behavior
  patch, fallback, or `testconfigs/` change was made.

- Accepted the narrow Nemotron Super inline shared/local `__nv_logf(1.0f +
  __expf(x))` SSD softplus patch after adding gated heatmap substage timing.
  Added `KRASIS_HEATMAP_SUBSTAGE_TIMING=1` timing rows for per-prompt
  heatmap tokenization/prefill/decode, export/write/cleanup, ranking, and
  post-heatmap-to-ready elapsed time, then built through `./dev build` and ran
  one timed `--test-endpoints` server. Exact corrected-38 no-trace/device
  requests and HF through `./dev generate-reference` passed the narrow gate:
  token-23/state-79/dim-6733 scaled-CB now matches HF `0xb983`, dim `6733`
  SSD/pre-out-proj match HF (`0x3a4d`/`0x39ab`), downstream dim `4025`
  branch output matches HF `0x3a88`, and prior dim `1328` remains fixed
  (`0x3d70`/`0xba83`). Heatmap timing was bounded: prompt loop `109.891 s`,
  total heatmap `109.904 s`, post-heatmap-to-ready `2.002 s`. Warmed
  no-debug corrected-38 prefill samples were `1179.0`, `1183.0`, and
  `1167.0 ms`. No fallback or precompute/shared-buffer design was added, and
  `testconfigs/` stayed untouched.

- Added a diagnostic-only production-context heatmap timing reclassification
  for the reverted Nemotron Super fast shared-scale patch. Re-verified the
  `0913` SSD-loop probe, `0853` rejection/revert artifacts, and reverted
  source first, then parsed timestamped accepted/reverted and rejected
  candidate startup logs through `./dev python` without starting a full
  corrected-38 acceptance server, patching production, adding fallbacks, or
  touching `testconfigs/`. The exact heatmap startup path is `_build_heatmap`
  running six held-out prompts through `rust_prefill_tokens` and
  `gpu_generate_batch(max_tokens=256)` before HCS ranking. Existing logs do
  not show a candidate-specific `__nv_logf` explosion: accepted/reverted
  reference heatmaps were `183.396-183.513 s`, the `0853` inline candidate was
  `184.523 s`, and the `0918` shared-scale candidate was `183.914 s`. The
  prior `308 s` server-ready rejection was too coarse because normal heatmap
  decode dominates startup.

- Added a diagnostic-only SSD-loop-shaped CUDA microbenchmark for the rejected
  Nemotron Super fast shared-scale patch. Re-verified the `0853`
  rejection/revert artifacts first and kept production reverted, then ran a
  standalone probe through `./dev python` with `nvcc -O3 --use_fast_math`.
  The probe used an SSD local-scan-shaped loop (`L=38`, state `128`, `8192`
  lanes) with dt/C/B/x loaded from device arrays and compared the accepted
  split, shared/local `__nv_logf(1 + __expf)`, local-only `__nv_logf`, and
  bounded approximations across the 52 observed `dt+bias` values. Shared and
  local-only `__nv_logf` matched token-23/state-79/dim-6733 HF bits and the
  `-3.5078125` control; shared `__nv_logf` was only `1.011534x` accepted in
  the loop, with `23-26` registers, no spills, and identical occupancy. This
  does not explain the prior `308 s` heatmap regression, so no production
  patch was proposed.

- Rejected the narrow Nemotron Super fast shared-scale production patch from
  the `08:37` CUDA probe winner. The candidate patched only the inline SSD
  shared/local scale softplus path to `__nv_logf(1.0f + __expf(x))`, without
  reintroducing the rejected precompute/shared-buffer design, adding
  fallbacks, or touching `testconfigs/`. The candidate built through
  `./dev build`, but the single `--test-endpoints` server reached ready only
  after `308 s`, failing the startup/heatmap acceptance check before the
  corrected-38 request pair, HF oracle, or warmed no-debug speed samples were
  run. Source was reverted and rebuilt back to the last accepted runtime.

- Added a diagnostic-only standalone CUDA probe for the rejected Nemotron Super
  shared-scale `dt_out` path. Re-verified the reverted runtime and `0918`
  rejection artifacts first, then ran the probe through `./dev python`
  orchestration without starting a full Nemotron server, patching production,
  adding fallbacks, or touching `testconfigs/`. The probe covered 52 observed
  `dt+bias` values from `-5.666015625` to `1.125` under
  `nvcc -O3 --use_fast_math -arch=sm_80`. Only `__nv_logf(1 + __expf)` and
  `__nv_logf(1 + __nv_expf)` matched the token-23 dim `6733` HF `dt_out`
  bits (`0x3c44637a`) and scaled-CB BF16 side (`0xb983`) plus the
  `-3.5078125` control bits. The preferred bounded-cost candidate is
  `__nv_logf(1 + __expf)`: median `0.256704 ms` for 268,435,456 probe ops,
  `1.028725x` the accepted `log1pf(expf)` local-emission path.

- Rejected the BF16 chunk-cumsum `dt_out` production patch candidates after a
  trace-first implementation gate. The recurrence-only candidate aligned
  Krasis chunk-cumsum `dt_out` with HF actual Triton `log(1 + exp)` semantics,
  but token-23 dim `6733` scaled-CB still rounded to `0xb982` instead of HF
  `0xb983`, and downstream branch output dim `4025` stayed `0x3a8a` instead
  of HF `0x3a88`. Focused speed for that rejected candidate was exact
  no-trace prefill `2187.3 ms` and no-debug warmed prefill samples
  `2163.9`, `2131.7`, `2131.4 ms`. New data proved the accepted local
  emission scale was implicated, so a shared-`dt_out` scale candidate was
  built, but it reached ready only after more than `303 s`; heatmap completed
  very late and the server was killed before requests. No patch was accepted,
  no fallback was added, `testconfigs/` stayed untouched, and the source was
  rebuilt back to the last accepted runtime.

- Added a BF16-only trace-first HF Triton `_chunk_cumsum_fwd` `dt_out`
  semantics gate for Nemotron Super dim `6733`. Re-verified the 06:11
  artifacts first, then added request-gated HF diagnostics to split raw
  `dt+bias`, softplus input/output precision, actual Triton `dt_out`,
  dA/cumsum propagation, decay, scale, and BF16 scaled-CB cast for token
  `23`, state `79`, dim `6733` plus adjacent controls. No Krasis behavior
  patch, fallback, or `testconfigs/` edit was made. Exact corrected-38
  Krasis no-trace returned `11745,63467` (`Hereelmi`) and device trace
  returned `11745,13968` (`Here Code`); HF stayed `1975,29896`
  (` ```python`). Final classification: raw dt bits (`0x3f65`), dt bias
  (`-5.3125`), and FP32 dt+bias (`-4.41796875`) match, while HF actual
  Triton `dt_out` is `0.011986607685685158` / `0x3c44637a`, exactly matching
  the CUDA `log(1 + exp(x))` candidate. `log1p(exp)`,
  `torch.nn.functional.softplus`, and `torch.log1p(torch.exp(x))` are lower
  by `3.259629011154175e-08`. dA/cumsum and decay differences are downstream
  propagation from actual `dt_out`; HF rounds scaled C@B to `0xb983`, Krasis
  to `0xb982`.

- Added a BF16-only trace-first token-23 scale-producer provenance gate for
  Nemotron Super dim `6733`. Re-verified the 05:44 artifacts first, then
  added request-gated HF diagnostics for actual chunk-cumsum raw-dt layout,
  raw dt, bias, dt+bias, recomputed softplus, actual `dt_out`, dA/cumsum,
  decay, scale, and BF16 scaled-CB cast. No Krasis behavior patch or
  `testconfigs/` edit was made. Exact corrected-38 Krasis no-trace and
  device-trace requests both returned `11745,1051` (`Here3`); HF stayed
  `1975,29896` (` ```python`). An initial HF rerun exposed a trace-only
  raw-dt indexing bug; fixed it to chunk-major
  `batch,head,chunk,chunk_position` indexing and reran HF. Final
  classification: raw dt bits, dt bias, FP32 dt+bias, and recomputed
  `log1p(exp)` match, but HF actual Triton chunk-cumsum `dt_out` is
  `3.4458935260772705e-08` above Krasis local-emission dt. That dt-side
  value is needed to round scaled C@B to `0xb983`; Krasis rounds to
  `0xb982`. dA/decay differs downstream because it is accumulated from
  `dt_out`, not because it is the first producer.

- Added a BF16-only trace-first actual chunk-scan accumulator/pre-store
  provenance gate for Nemotron Super dim `6733`. Re-verified the 05:21
  artifacts first, then added request-gated HF diagnostics that split the
  actual Triton FP32 replay against Krasis local scan by token contribution,
  C@B, dt/decay/scale, `D*x`, accumulation, and final pre-store sum. No
  Krasis behavior patch or `testconfigs/` edit was made. Exact corrected-38
  Krasis no-trace and device-trace requests both returned `3149,10575`
  (`def dog`); HF stayed `1975,29896` (` ```python`). The first material
  local chunk-scan split for dim `6733` is token position `23`: HF actual
  scale rounds scaled C@B to `0xb983`, while Krasis recorded scale rounds it
  to `0xb982`, producing a term delta of `3.129243850708008e-07` and the
  downstream pre-store split `0x3a4c89f8` vs `0x3a4c74fa`. The selected C@B
  state dot is not the producer; paired with the same scale, HF actual, HF
  manual, and Krasis C@B variants choose the same BF16 side, and the top
  state contributor remains state `79`. `D*x`, prior state, x bits, final
  BF16 store/cast, and final accumulation order are downstream or matching.

- Added a BF16-only trace-first HF actual SSD store/cast boundary gate for
  Nemotron Super dim `6733`. Re-verified the 04:49 artifacts first, then
  added request-gated HF diagnostics that replay the actual
  `mamba_ssm.ops.triton.ssd_chunk_scan._chunk_scan_fwd_kernel` with the same
  forward inputs but an FP32 output buffer, leaving normal HF BF16 forward
  output unchanged. No Krasis behavior patch or `testconfigs/` edit was made.
  Exact corrected-38 Krasis no-trace and device-trace requests both returned
  `3149,10575` (`def dog`); HF stayed `1975,29896` (` ```python`). For dim
  `6733`, HF actual pre-store is `0.0007802541367709637` / `0x3a4c89f8`,
  which rounds to and stores `0x3a4d`. The HF manual reconstruction and
  Krasis pre-store both remain `0.0007799413288012147` / `0x3a4c74fa`,
  which rounds to `0x3a4c`; Krasis stores that candidate. Classification:
  the final HF BF16 store/cast is not wrong. The split is actual HF Triton
  chunk-scan accumulator/pre-store semantics not represented by the manual
  reconstruction; only dim `6733` crosses the BF16 boundary among selected
  dims.

- Added a BF16-only trace-first SSD output dim-`6733` pre-store/cast
  provenance gate for Nemotron Super after the 04:30 pre-out-proj provenance
  result. Existing Gemma speed record was carried forward for status only,
  not rerun: `5619.6` prefill, `92.43` internal decode, `155.69` HTTP.
  Added only minimal request-gated diagnostic exposure for exact SSD output
  pre-store/cast fields already present in the device trace: FP32
  `y_pre_store` bits, BF16 candidate bits, and stored-vs-candidate checks;
  added matching HF manual candidate fields. No behavior patch or
  `testconfigs/` edit was made. Exact corrected-38 Krasis no-trace returned
  `11745,1051` (`Here3`), selected device trace returned `11745,1657`
  (`Here``), and HF stayed `1975,29896` (` ```python`). Dim `6733` remains
  the only selected SSD stored-output mismatch, but exact pre-store FP32
  value/bits match HF and Krasis: `0.0007799413288012147`, `0x3a4c74fa`.
  The BF16 candidate from that value is `0x3a4c` on both sides; Krasis stores
  that candidate while HF actual stored SSD output is `0x3a4d`. `D*x`, prior
  state, local chunk-scan summary, B/C/X/DT hashes, x/dt bits, local-scan
  token inputs, BF16 `CB * scale` cast bits, gated norm, and pre-out-proj
  store/aliasing are not the producer. Do not patch from this gate; next
  target is HF actual SSD store/cast semantics for dim `6733`.

- Added a BF16-only trace-first pre-out-proj dim-`6733` provenance gate for
  Nemotron Super after the 04:09 layer-0 branch/output-proj result. Existing
  Gemma speed record was carried forward for status only, not rerun:
  `5619.6` prefill, `92.43` internal decode, `155.69` HTTP. Re-verified the
  04:09 artifacts first: output dim `4025` is driven by exactly one full-row
  pre-out-proj input mismatch, dim `6733`. Reused the existing request-gated
  selected-dim SSD/gated-norm/pre-out-proj trace path for dims
  `6731,6732,6733,6734,6735` plus output witness `4025`; no source
  instrumentation, behavior patch, or `testconfigs/` edit was needed. Exact
  corrected-38 Krasis no-trace and device-trace requests both returned
  `11745,2259` (`Here </`); HF stayed `1975,29896` (` ```python`). Dim
  `6733` first visibly differs at SSD output/gated-norm input: HF `0x3a4d`
  (`0.000782012939453125`) vs Krasis `0x3a4c`
  (`0.0007781982421875`). Gate bits, norm weight, gated-norm output store,
  pre-out-proj element aliasing, and adjacent controls are clean; next target
  is exact SSD output pre-store/cast provenance for dim `6733`.

- Added a BF16-only trace-first layer-0 branch/output-proj dim-`4025`
  provenance gate for Nemotron Super after the 03:32 post-dt-softplus
  rebaseline. Re-verified the 03:32 artifacts first, then reused the existing
  request-gated branch/out-proj path for dims `1328,4023,4024,4025,4026,4027`
  without changing behavior or touching `testconfigs/`. Valid corrected-38
  Krasis no-trace and device-trace requests ran on one `--test-endpoints`
  server after a recorded false start against an unrelated local port `8000`;
  both valid requests returned `3149,10575` (`def dog`). HF stayed
  `1975,29896` (` ```python`). For output dim `4025`, selected mixer input,
  SSD output, gate, selected pre-out-proj value, and out-proj weight row hash
  all match HF, while final branch output still differs HF `0x3a88`
  (`0.00103759765625`) vs Krasis `0x3a8a` (`0.0010528564453125`). Full
  pre-out-proj contribution analysis found exactly one producer input dim:
  `6733`, HF `0x39ab` (`0.0003261566162109375`) vs Krasis `0x39aa`
  (`0.000324249267578125`), contributing `3.795139491558075e-08` to the
  output-`4025` manual-dot delta. Next target is gated-norm/SSD provenance for
  pre-out-proj dim `6733`; scaled-`B` plus SSD emission remains provisional.

- Added a BF16-only post-dt-softplus corrected-38 rebaseline for Nemotron
  Super after the accepted layer-0 SSD dt-softplus patch. Re-verified the
  02:58 artifacts first: dim `1328` and the downstream output-dim flips at
  `3265`, `739`, `1067`, and `445` remain fixed. Ran the exact corrected-38
  no-trace/all-layer device pair, HF through `./dev generate-reference`, and
  focused selected-dim provenance for dim `4025` plus controls, all through
  built `./dev` commands and without touching `testconfigs/` or patching
  behavior. Layer-0 handoff aggregate parity is restored, but exact BF16
  bits expose the next remaining split at layer-0 Mamba2 branch output dim
  `4025`: HF branch `0x3a88` (`0.00103759765625`) vs Krasis `0x3a8a`
  (`0.0010528564453125`). The residual operand matches (`0xbb48`), while
  the layer-1 norm/router input becomes HF `0xbb04` vs Krasis `0xbb03` and
  stored RMSNorm output becomes HF `0xbcc0` vs Krasis `0xbcbe`. All selected
  controls match. The dt-softplus patch remains accepted; scaled-`B` plus SSD
  emission remains provisional; top-k, experts, logits, and decode step-1
  remain out of scope.

- Implemented and accepted the BF16-only layer-0 SSD dt-softplus production
  correctness patch for Nemotron Super. The production Mamba2 SSD local
  chunk-scan emission scale now uses the HF-matching precise
  `log1pf(expf(x))` softplus ordering at the BF16 `CB * scale` cast point,
  while leaving BF16 `dt+bias`, state/decay recurrence, HCS, tokenization,
  logits, layer 4, decode, fallbacks, and `testconfigs/` untouched. `./dev
  build`, the exact corrected-38 no-trace/device-trace pair on one idle
  `--test-endpoints` server, HF through `./dev generate-reference`, and a
  focused no-debug timing check passed. Dim `1328` SSD output now matches HF
  exactly (`0x3d70`, `0.05859375`), gated-norm input matches, and the
  pre-out-proj dim is the HF bit `0xba83`. The four downstream out-proj BF16
  flips at dims `3265`, `739`, `1067`, and `445` all disappeared, with all
  adjacent controls still matching. Focused timing showed no material prefill
  regression: exact same-payload prefill changed from `2172.4ms` to
  `2179.1ms` (`+0.31%`), and three no-debug warmed requests averaged
  `2139.3ms`. Broader first-token correctness remains open: Krasis still
  returns first token `11745` while HF returns `1975`.

- Added BF16-only trace-first layer-0 dt-softplus scale dim-1328 provenance
  for the exact corrected-38 Nemotron Super prompt. Existing Gemma speed
  record was carried forward for status only, not rerun: `5619.6` prefill,
  `92.43` internal decode, `155.69` HTTP. Added request-gated selected-dim
  dt/scale diagnostics for dim `1328` plus adjacent controls across raw dt,
  dt bias, FP32 and BF16 dt+bias, CUDA fast softplus, precise
  `log1p(expf())` softplus, BF16 dt+bias softplus candidate, scale
  multiplication, and BF16 `CB * scale` cast bits, plus matching HF
  diagnostics through `./dev generate-reference --diagnose-layer0-element-dims`;
  no production behavior patch was made. Prompt hash stayed
  `0xf62d9e4f5b39fdc7`; HF oracle stayed `1975,29896` (` ```python`);
  Krasis no-trace and selected device trace both returned `11745,7297`
  (`Here)`). Raw dt bits, dt bias values, and BF16 dt+bias bits match HF
  exactly. The current CUDA fast softplus/scale path produces `16` BF16
  `CB * scale` cast mismatches, first at token position `1`, while the precise
  `log1p(expf())` scale path has `0` BF16 cast mismatches; the BF16 dt+bias
  candidate is worse with `19` mismatches, so the rejected BF16 `dt+bias`
  hypothesis remains rejected. The combined scaled-`B` plus SSD emission
  changes remain provisional and not accepted.

- Added BF16-only trace-first layer-0 SSD local chunk-scan dim-1328
  provenance for the exact corrected-38 Nemotron Super prompt. Existing Gemma
  speed record was carried forward for status only, not rerun: `5619.6`
  prefill, `92.43` internal decode, `155.69` HTTP. Added request-gated
  selected-dim local-scan diagnostics for dim `1328` plus adjacent controls
  across x/dt bits, dt softplus, A/decay/cumsum, C@B, BF16
  `CB * decay * dt` cast points, per-token terms, and cumulative state, plus
  matching HF diagnostics through
  `./dev generate-reference --diagnose-layer0-element-dims`; no production
  behavior patch was made. Prompt hash stayed `0xf62d9e4f5b39fdc7`; HF oracle
  stayed `1975,29896` (` ```python`); Krasis no-trace returned
  `11745,63467` (`Hereelmi`) and selected device trace returned `11745,2259`
  (`Here </`). For dim `1328`, raw x/dt bits, A, C@B source data, and prior
  state are not the producer. Mixed candidates prove the remaining
  `1.746e-7` local-scan delta follows the dt-softplus scale used before the
  BF16 `CB * decay * dt` cast. The combined scaled-`B` plus SSD emission
  changes remain provisional and not accepted.

- Added BF16-only trace-first layer-0 SSD output/norm-input dim-1328
  provenance for the exact corrected-38 Nemotron Super prompt. Added
  request-gated selected-dim Mamba2 SSD-output producer diagnostics for dim
  `1328` plus adjacent controls, with matching HF
  `layer0_mamba2_ssd_output_element_details` through
  `./dev generate-reference --diagnose-layer0-element-dims`; no production
  behavior patch was made. Prompt hash stayed `0xf62d9e4f5b39fdc7`; HF oracle
  stayed `1975,29896` (` ```python`); Krasis no-trace returned
  `11745,1051` (`Here3`) and selected device trace returned `11745,1356`
  (`Hereem`). Only dim `1328` differs at SSD output: HF `0x3d70`
  (`0.05859375`) vs Krasis `0x3d71` (`0.058837890625`). SSD output storage
  aliases cleanly into gated norm for all selected dims, and C/B/X/DT source
  hashes plus `D*x` match HF; the remaining difference is a `1.746e-7` local
  chunk-scan contribution delta that crosses the BF16 rounding midpoint. Next
  target is SSD emission internals for the local chunk-scan contribution/state
  computation at dim `1328`.

- Added BF16-only trace-first layer-0 gated-norm/pre-out-proj dim-1328
  provenance for the exact corrected-38 Nemotron Super prompt. Added
  request-gated selected-dim Mamba2 gated-norm diagnostics for dim `1328` plus
  adjacent controls `1327`, `1329`, `1326`, and `1330`, and matching HF
  `layer0_mamba2_gated_norm_element_details` through
  `./dev generate-reference --diagnose-layer0-element-dims`. No production
  behavior patch was made. Prompt hash stayed `0xf62d9e4f5b39fdc7`; HF oracle
  stayed `1975,29896` (` ```python`); Krasis no-trace returned
  `11745,84855` (`Here courtesy`) and selected device trace returned
  `11745,63467` (`Hereelmi`). Dim `1328` first differs at gated-norm norm
  input / SSD output: HF `0x3d70` (`0.05859375`) vs Krasis `0x3d71`
  (`0.058837890625`). Gate bits, silu gate, norm weight, adjacent control
  stored bits, and final BF16 store semantics are not the producer. Next target
  is upstream of gated norm, at the layer-0 SSD output/norm-input value for
  dim `1328`.

- Added BF16-only trace-first layer-0 full-row pre-out-proj contribution
  provenance for the exact corrected-38 Nemotron Super prompt. Added a strict
  request-gated `debug_prefill_device_trace_full_pre_out_proj` device-trace
  flag, a bulk full-row pre-out-proj element trace kernel, and matching HF
  `layer0_mamba2_out_proj_contribution_rows` diagnostics behind
  `./dev generate-reference --diagnose-layer0-element-dims`; no production
  behavior patch was made. Prompt hash stayed `0xf62d9e4f5b39fdc7`; HF oracle
  stayed `1975,29896` (` ```python`); Krasis no-trace and full-row device
  trace both returned `11745,1051` (`Here3`). Exactly one pre-out-proj BF16
  input dim differs: dim `1328`, HF `0xba83` vs Krasis `0xba84`. That one-ULP
  input delta accounts for the manual-dot deltas on all selected output rows
  and flips the four drift outputs (`3265`, `739`, `1067`, `445`) while
  controls stay on the same BF16 output side. Next target is the layer-0
  gated-norm/pre-out-proj producer for dim `1328`.

- Added BF16-only trace-first layer-0 Mamba2 mixer branch output provenance
  localization for the exact corrected-38 Nemotron Super prompt. Added
  request-gated selected-dim Krasis diagnostics for mixer input,
  pre-out-proj branch internal output, out-proj output bits, and selected
  out-proj row hashes/manual dots, plus matching HF
  `./dev generate-reference --diagnose-layer0-element-dims` diagnostics; no
  production behavior patch was made. Existing Gemma speed record was carried
  forward for status only, not rerun: `5619.6` prefill, `92.43` internal
  decode, `155.69` HTTP. Prompt hash stayed `0xf62d9e4f5b39fdc7`; HF oracle
  stayed `1975,29896` (` ```python`); valid idle-server Krasis no-trace and
  selected device trace both returned `11745,1051` (`Here3`). All selected
  mixer input values, selected pre-out-proj values, and selected out-proj
  weight rows match HF. The four drifting branch-output dims split at final
  out-proj BF16 output; the full pre-out-proj projection-input row hash differs
  and tiny manual-dot deltas flip BF16 rounding on those output dims. The next
  upstream target is therefore unselected/full-row pre-out-proj or gated-norm
  contributors feeding the layer-0 out-proj dot.

- Added BF16-only trace-first layer-0 final handoff pair-sum provenance
  localization for the exact corrected-38 Nemotron Super prompt. Added
  selected-dim HF layer-0 residual/branch/output element details via
  `./dev generate-reference --diagnose-layer0-element-dims` and explicit
  Krasis materialization metadata for the existing request-gated layer-0
  output-sum trace; no production behavior patch was made. Prompt hash stayed
  `0xf62d9e4f5b39fdc7`; HF oracle stayed `1975,29896` (` ```python`);
  Krasis no-trace returned `11745,2259` (`Here </`) and selected device trace
  returned `11745,13968` (`Here Code`). For all selected dims, Krasis
  residual operands match HF; the four drifting dims differ only in the
  layer-0 hidden/branch operand bits, causing the rounded pair-sum/stored
  output differences. All controls match through residual, branch, rounded
  sum, and stored output. The layer-0 branch output is therefore the next
  upstream producer to localize.

- Added BF16-only trace-first layer-1 fused residual-add operand provenance
  localization for the exact corrected-38 Nemotron Super prompt. Added
  request-gated layer-0 output-sum source metadata and selected-dim element
  summaries plus layer-1 fused-add source metadata; no production behavior
  patch was made. Prompt hash stayed `0xf62d9e4f5b39fdc7`; HF oracle stayed
  `1975,29896` (` ```python`); final detail-fix Krasis no-trace and selected
  device trace both returned `11745,2259` (`Here </`). For all selected dims,
  layer-0 residual/hidden operands match the layer-1 fused-add
  residual/hidden operands bit-for-bit, the layer-0 rounded handoff sum
  matches the layer-1 rounded fused-add sum, and the stored norm input matches
  the rounded sum. The four drifting dims still differ from HF while all eight
  controls match. The four-element drift is therefore already present in the
  layer-0 final handoff pair-sum, not introduced by stale, double-added, or
  unrounded layer-1 fused-add operand composition.

- Added BF16-only trace-first layer-1 fused residual-add element provenance
  localization for the exact corrected-38 Nemotron Super prompt. Added
  request-gated selected-dimension fused-add RMSNorm input/output diagnostics
  plus `./dev generate-reference --diagnose-layer1-element-dims`; no
  production behavior patch was made. Prompt hash stayed
  `0xf62d9e4f5b39fdc7`; HF oracle stayed `1975,29896` (` ```python`);
  Krasis no-trace and selected device trace both returned `11745,2259`
  (`Here </`). For all selected dims, Krasis rounded residual-add sum matches
  the stored norm input. The four drifting dims (`3265`, `739`, `1067`,
  `445`) differ between HF layer-0 output/layer-1 norm input and Krasis
  rounded sum; all eight adjacent control dims match HF exactly. The producer
  is therefore Krasis fused residual-add input composition/rounding that builds
  the layer-1 norm input, upstream of RMSNorm scale/store/router.

- Added BF16-only trace-first layer-1 RMSNorm/router-input element drift
  localization for the exact corrected-38 Nemotron Super prompt. Added
  request-gated exact 4096-element fused-add RMSNorm input/output diagnostics;
  no production behavior patch was made. Prompt hash stayed
  `0xf62d9e4f5b39fdc7`; HF oracle stayed `1975,29896` (` ```python`);
  Krasis no-trace and device trace both returned `11745,2259` (`Here </`).
  Exact stored-output/router-input hashes still differ (HF
  `0x6637fa373508e183`, Krasis `0x29511d55b3faf412`), but the difference is
  only four BF16 dimensions. Those same four dimensions already differ in the
  rounded fused RMSNorm input; Krasis rounded residual-add bits match its norm
  input exactly, weights match exactly, HF stored candidate matches HF actual
  output, and rsqrt differs only `3.8147e-6`. The producer is therefore
  upstream of RMSNorm scale/cast/store, in the fused residual-add input
  elements.

- Added BF16-only trace-first layer-1 router-logit semantics localization for
  the exact corrected-38 Nemotron Super prompt. Added request-gated exact
  router-input/weight-row hash and manual-logit diagnostics; no production
  behavior patch was made. A trace-only FNV hash seed bug was found before
  classification, fixed, rebuilt, and rerun. Existing Gemma speed record was
  carried forward for status only, not rerun: `5619.6` prefill, `92.43`
  internal decode, `155.69` HTTP. Prompt hash stayed `0xf62d9e4f5b39fdc7`; HF
  oracle stayed `1975,29896` (` ```python`). Krasis no-trace returned
  `11745,63467` (`Hereelmi`) and device trace returned `11745,1051`
  (`Here3`), with first-token top-k/logprobs matching exactly and step-1 still
  out of scope. The layer-1 exact BF16 router-input row hash differs (HF
  `0x6637fa373508e183`, Krasis `0x29511d55b3faf412`), all 512 router weight
  row hashes match exactly, and manual sequential FP32 logits from the same
  input/weight slices diverge at the same scale as production logits
  (`0.001059%` vs `0.0009876%` relative l2). The raw router-logit split is
  therefore residual BF16 router-input element drift, not router weight
  layout/rows or GEMM accumulation/cast behavior.

- Added BF16-only trace-first layer-1 router-logit parity localization for the
  exact corrected-38 Nemotron Super prompt. Added request-gated diagnostics for
  router input, router weight aggregate, raw logits, sequential-FP32 logit
  candidate, sigmoid, top-k IDs/weights, and routed scaling; no production
  behavior patch was made. Prompt hash stayed `0xf62d9e4f5b39fdc7`; HF oracle
  stayed `1975,29896` (` ```python`); Krasis no-trace returned
  `11745,63467` (`Hereelmi`) and device trace returned `11745,2259`
  (`Here </`). Layer-1 RMSNorm/router-input parity still holds at
  `0.00000927%` relative l2 and router weight aggregate matches exactly. The
  first material post-RMSNorm split remains raw router logits: HF
  `47.89646530151367`, Krasis `47.89693832397461` (`0.0009876%` relative
  l2), before sigmoid, top-k, routed scaling, or expert execution. The combined
  scaled-`B` plus SSD emission changes remain provisional, and decode
  trace/no-trace step-1 divergence remains recorded out of scope.

- Added BF16-only trace-first layer-1 post-RMSNorm MoE localization for the
  exact corrected-38 Nemotron Super prompt after the accepted fused-add
  RMSNorm cast-semantics patch. Added request-gated Krasis layer-1 MoE device
  summaries and matching HF router-logit summaries under
  `./dev generate-reference --diagnose-layer1-internals`; no production
  behavior patch was made. Prompt hash stayed `0xf62d9e4f5b39fdc7`; HF oracle
  stayed `1975,29896`; Krasis no-trace returned `11745,1051` (`Here3`) and
  device trace returned `11745,2259` (`Here </`). Layer-1 RMSNorm parity still
  holds at `0.00000927%` relative l2. The first remaining post-RMSNorm split is
  router logits: HF `47.89646530151367`, Krasis `47.89693832397461`
  (`0.0009876%` relative l2), before top-k IDs/weights and expert execution.
  Dynamic HCS reported `copy_failures=0`; cleanup clear. The combined
  scaled-`B` plus SSD emission changes remain provisional, and decode
  trace/no-trace step-1 divergence remains recorded out of scope.

- Implemented the BF16-only fused-add RMSNorm cast-semantics fix for the exact
  corrected-38 Nemotron Super prompt. The production change is limited to
  `fused_add_rmsnorm_batched_kernel`: round/store the residual add to BF16,
  then compute mean-square/rsqrt from the rounded BF16 value while preserving
  the existing parallel reduction shape. Prompt hash stayed
  `0xf62d9e4f5b39fdc7`; HF oracle stayed `1975,29896`. Krasis remains
  first-token wrong (`11745`), but the layer-1 stored RMSNorm split moved from
  the prior `0.003357%` relative l2 to `0.00000927%` vs HF
  (`20.5669002532959` vs `20.566898345947266`). Warmed request-local timing
  after one warm request showed no hot-path regression: measured prefill
  `2108.6ms` / `2115.5ms` and `[PREFILL-TIMING]` norm `1.8ms` / `1.7ms`.
  Dynamic HCS reported `copy_failures=0`; decode low-water stayed `866 MB+`.
  Accepted for layer-1 fused-add RMSNorm parity only. The combined scaled-`B`
  plus SSD emission changes remain provisional, final logits remain wrong, and
  decode trace/no-trace step-1 divergence remains recorded and out of scope.

- Added BF16-only layer-1 RMSNorm reduction implementation planning for the
  exact corrected-38 Nemotron Super prompt, keeping the combined scaled-`B`
  plus SSD emission changes in place but provisional and not accepted. Added a
  diagnostic-only contiguous RMSNorm reduction candidate plus a selected-layer
  replay benchmark; no production RMSNorm behavior patch was made. Existing
  Gemma speed record was carried forward, not rerun: `5619.6` prefill,
  `92.43` internal decode, `155.69` HTTP. Prompt hash stayed
  `0xf62d9e4f5b39fdc7`; HF oracle artifact stayed `1975,29896`; Krasis
  no-trace returned `11745,95707` (`Here}^\`) and device trace returned
  `11745,1657` (`Here``), with identical first-token top-k/logprobs. Current
  fused-add production RMSNorm output still differs from HF at `0.003357%`
  relative l2, but a plain production RMSNorm replay from the already
  BF16-rounded residual reaches the HF-equivalent stored-output scale
  (`0.00000927%`). Planning conclusion: do not implement a naive serial
  reduction; the minimal GPU-safe target is fused-add RMSNorm cast semantics,
  where the residual add is rounded/stored to BF16 before mean-square/rsqrt is
  computed, preserving the one-kernel parallel reduction shape. Decode
  trace/no-trace step-1 divergence remains recorded and out of scope.

- Added BF16-only layer-1 RMSNorm semantic validation for the exact
  corrected-38 Nemotron Super prompt, keeping the combined scaled-`B` plus SSD
  emission changes in place but provisional and not accepted. Added
  request-gated Krasis sequential/index-order RMSNorm candidates and matching
  `./dev generate-reference --diagnose-layer1-internals` HF summaries; no
  production RMSNorm, HCS, tokenization, logits, layer-4, decode, fallback,
  HQQ, or optimization patch was made. The prompt hash stayed
  `0xf62d9e4f5b39fdc7`; HF stayed `1975,29896`; Krasis no-trace returned
  `11745,2259` (`Here </`) and device trace returned `11745,1356`
  (`Hereem`), with identical first-token top-k/logprobs. Production
  parallel-reduction RMSNorm still splits at BF16 stored output:
  HF `20.5669002532959` vs Krasis `20.56620979309082` (`0.003357%`), while
  the index-order FP32 mean-square candidate matches HF mean-square exactly
  and lands on the HF-equivalent BF16 stored output scale:
  HF sequential `20.566896438598633`, Krasis sequential
  `20.566898345947266` (`0.00000927%`). HF actual norm output, HF manual
  stored candidate, and HF sequential stored candidate share the same SHA.
  Classification: RMSNorm reduction/rsqrt semantics are the minimal target for
  a future implementation gate; a production patch was intentionally deferred
  to avoid a naive hot-path serial reduction.

- Added BF16-only layer-1 RMSNorm parity localization for the exact
  corrected-38 Nemotron Super prompt, keeping the combined scaled-`B` plus SSD
  emission changes in place but provisional and not accepted. Added
  request-gated, non-perturbing Krasis device summaries plus matching
  `./dev generate-reference --diagnose-layer1-internals` HF summaries for
  RMSNorm input, weight, mean-square accumulator, eps, rsqrt, pre-store
  output, and stored output; no production behavior, HCS, tokenization, logits,
  layer-4, decode, fallback, HQQ, or optimization patch was made. Prompt hash
  stayed `0xf62d9e4f5b39fdc7`; HF stayed `1975,29896`; Krasis no-trace
  returned `11745,7155` (`Here https`) and device trace returned
  `11745,2259` (`Here </`), with identical first-token top-k/logprobs. Layer-1
  norm input matches HF exactly by l2, eps matches exactly, and weight stats
  match. Mean-square differs only by `2.33e-10` absolute and rsqrt by
  `3.81e-6`; pre-store FP32 output remains within `0.000027821%` relative l2,
  but BF16 stored norm output is the material split: HF
  `20.5669002532959` vs Krasis `20.56620979309082` (`0.003357%`). HF actual
  norm output matches its stored-output candidate, so this is not a summary
  artifact. The split is a tiny FP32 accumulation/rsqrt/pre-store precision
  delta amplified at BF16 cast/store, not input, weight layout/indexing, or
  eps. Decode trace/no-trace step-1 divergence remains open and out of scope.

- Added BF16-only post-emission downstream prefill localization for the exact
  corrected-38 Nemotron Super prompt, with the combined scaled-`B` plus SSD
  emission changes kept in place but provisional. Added a minimal
  `./dev generate-reference --diagnose-layer1-internals` HF diagnostic and ran
  exact no-trace/device-trace Krasis requests plus the HF oracle; no production
  behavior, HCS, tokenization, logits, layer-4, decode, fallback, HQQ, or
  optimization patch was made. The prompt hash stayed
  `0xf62d9e4f5b39fdc7`; HF stayed `1975,29896`; Krasis selected layer-1 rerun
  returned no-trace `11745,84855` and device-trace `11745,1013` with identical
  first-token top-k/logprobs. Layer-0 output now hands off to layer 1 at HF
  summary precision (`l2=2.1844191551208496` on both paths). The first
  remaining prefill split is layer-1 norm output, before MoE/router/expert
  execution: HF `l2=20.5669002532959` vs Krasis `20.56620979309082`
  (`0.003357%` relative l2). Layer-1 output then differs by `0.99337%`.
  Decode trace/no-trace step-1 divergence remains open and deferred until
  prefill correctness is re-established.

- Implemented the BF16-only generic Mamba2 prefill SSD output/chunk-scan
  emission semantics for the exact corrected-38 Nemotron Super gate, while
  keeping the 14:50 scaled-`B` state patch in place but provisional. A first
  all-prefix emission implementation was rejected after startup long
  calibration hung on the 8247-token probe; the final production change is
  chunk-bounded and emits the HF-matched form
  `D*x + dot(BF16(CB * decay * dt), x)` with BF16 store semantics. No HCS,
  tokenization, logits, layer-4, decode, graph/decode policy, calibration,
  fallback, HQQ, optimization, or unrelated Mamba path was changed. The final
  build exited `0`; exact prompt hash stayed `0xf62d9e4f5b39fdc7`; HF stayed
  `1975,29896`; Krasis no-trace returned `11745,63467` (`Hereelmi`) and
  device trace returned `11745,1051` (`Here3`) with identical first-token
  top-k/logprobs. The patch fixes layer-0 SSD output parity: production SSD
  output now matches HF actual chunk scan at summary precision (`0.0%`
  relative l2), gated norm matches (`0.0%`), and out-proj/mixer are within
  `0.00000643%`. Final logits are still wrong because HF token `1975` remains
  Krasis rank 2, so the combined scaled-`B` plus emission change remains
  provisional pending the next downstream correctness boundary.

- Added BF16-only layer-0 Mamba2 SSD output/chunk-scan emission localization
  for the exact corrected-38 Nemotron Super prompt, keeping the scaled-`B`
  production patch in place as provisional and not accepted. Added only
  request-gated, non-perturbing Krasis device summaries plus matching
  `./dev generate-reference --diagnose-layer0-internals` HF summaries for
  chunk-scan emission candidates; no further production behavior, HCS,
  tokenization, logits, layer-4, fallback, HQQ, or optimization patch was
  made. After a trace-only index fix and rebuild, the valid rerun kept prompt
  hash `0xf62d9e4f5b39fdc7`; HF stayed `1975,29896`, while Krasis no-trace and
  device-trace both returned `11745,1051` (`Here3`). The state and mixer
  distances remained at the 14:50 provisional values (`0.007222%` state,
  `0.009573%` mixer), and production SSD output remained worsened at
  `0.131121%`. The HF-like Krasis candidate
  `chunk_scan_y_bf16_cbscale_store` matched HF actual chunk-scan output at
  summary precision (`0.0%` relative l2 delta). Therefore the remaining SSD
  output regression is emission semantics: production emits `D*x + C*state`,
  while HF emits `D*x + dot(BF16(CB * decay * dt), x)` and stores BF16. Next
  implementation gate should align SSD output/chunk-scan emission and then
  reevaluate the provisional scaled-`B` patch as a combined change.

- Implemented the BF16-only scaled-`B` cast semantics for generic Mamba2
  prefill SSD state accumulation: the production sequential state update now
  casts `B * dt` to BF16 before FP32 recurrence, and the selected-position
  trace/CPU diagnostic mirrors the same path. No HCS, tokenization, logits,
  layer-4, decode, graph, calibration, chunk-size, fallback, HQQ, or
  optimization change was made. Build exited `0`, HF oracle exited `0`, and
  the exact corrected-38 prompt hash stayed `0xf62d9e4f5b39fdc7`. Krasis
  changed from the 14:20 baseline `11745,2259` (`Here </`) to no-trace
  `3149,63467` (`defelmi`) and device-trace `3149,10575` (`def dog`), while
  HF stayed `1975,29896`. The patch moved SSM final-state distance toward HF
  (`0.017221%` to `0.007222%`) and moved gated norm/out-proj/mixer distance
  toward HF (`0.013762%` mixer delta to `0.009573%`), but SSD output distance
  worsened (`0.001551%` to `0.131121%`). First-token top-k was identical
  between Krasis no-trace and device-trace, but step-1 still differed, so this
  is a mixed partial implementation result rather than full correctness
  acceptance.

- Added BF16-only layer-0 Mamba2 SSD state-accumulation localization for the
  exact corrected-38 Nemotron Super prompt. Added request-gated
  non-perturbing Krasis device summaries and matching
  `./dev generate-reference --diagnose-layer0-internals` HF summaries for
  initial SSM state, selected-token decay, post-decay state, FP32 update,
  BF16-cast update candidates, post-state candidates, final contribution
  candidates, and chunk-formula candidates. No production behavior, HCS,
  tokenization, logits, layer-4, fallback, HQQ, or optimization patch was
  made. Build exited `0`, HF oracle exited `0`, prompt hash stayed
  `0xf62d9e4f5b39fdc7`, Krasis returned `11745,2259` (`Here </`), and HF
  returned `1975,29896`. Shared selected rows `0`, `1`, `36`, and `37` match
  at trace precision for decay/update/state candidates. Actual HF chunk-state
  output is `l2=5219.86328125`, while Krasis production sequential state is
  `5220.76220703125` (`0.017221%`). The manual FP32 chunk formula matches
  Krasis production exactly, not HF; the BF16 scaled-B chunk formula matches
  actual HF within `0.00000935%`. The SSM drift is therefore HF chunk-state
  dtype/cast semantics: scaled `B` is cast to BF16 before FP32 dot/state
  accumulation. Initial state, decay application, state layout/indexing,
  softplus/dt-bias, and chunked-vs-sequential algebra are ruled out.

- Added BF16-only layer-0 Mamba2 SSD/scan localization for the exact
  corrected-38 Nemotron Super prompt. Added request-gated non-perturbing
  Krasis device summaries and matching
  `./dev generate-reference --diagnose-layer0-internals` HF summaries for
  A/D, dA, cumsum/decay, D*x, B*dt*x update, SSM state, C/state terms, SSD
  output, gated norm, out-proj, and mixer output. No behavior, HCS,
  tokenization, logits, layer-4, fallback, HQQ, or optimization patch was
  made. Build passed (`duration_s=138`), HF oracle exited `0`, prompt hash
  stayed `0xf62d9e4f5b39fdc7`, Krasis returned `11745,1051` (`Here3`), and
  HF returned `1975,29896`. A/D, dA, decay, D*x, and B*dt*x update match at
  trace precision; the first material post-softplus split is SSD scan/state
  accumulation: SSM state after SSD `5219.86328125` HF vs `5220.76220703`
  Krasis (`0.017221%`), C/state terms `0.017431%`, and C/state contribution
  `0.150736%`. The next BF16 target is SSD scan/state-passing semantics, not
  `dt+bias`, softplus, A/D, decay, D*x, or B-update.

- Added BF16-only layer-0 Mamba2 forward precision-semantics validation for
  the exact corrected-38 Nemotron Super prompt. Extended the built
  `./dev generate-reference --diagnose-layer0-internals` path to capture the
  actual HF cumsum/softplus forward path and a BF16-rounded downstream
  candidate. No production behavior, HCS, tokenization, logits, layer-4,
  fallback, HQQ, or optimization patch was made. HF oracle exited `0`
  (`duration_s=296`) with prompt hash `0xf62d9e4f5b39fdc7` and output
  `1975,29896`. Actual HF forward uses BF16 raw `dt`/`dt_bias` inputs but
  performs the `dt+bias` add in FP32 (`l2=55.20817565917969`) before FP32
  softplus (`l2=27.332555770874023`). Current Krasis `1302` summaries match
  that actual HF dt path, while the BF16-rounded candidate is a diagnostic
  artifact and does not track HF downstream at the mixer/output boundary.
  Therefore the proposed fix target is not BF16-rounding `dt+bias`; the next
  BF16 target is downstream SSD/scan or later Mamba2 math.

- Added BF16-only layer-0 Mamba2 `dt_softplus` parity localization for the
  exact corrected-38 Nemotron Super prompt. Added request-gated
  non-perturbing Krasis summaries for raw `dt`, `dt_bias`, `dt+bias`, and the
  exact SSD-kernel softplus path, and extended
  `./dev generate-reference --diagnose-layer0-internals` with matching BF16
  and FP32 HF summaries. No behavior, HCS, tokenization, logits, layer-4,
  fallback, HQQ, or optimization patch was made. Build passed
  (`duration_s=137`), HF oracle exited `0` (`duration_s=294`), and prompt hash
  stayed `0xf62d9e4f5b39fdc7`. Raw `dt` and `dt_bias` match exactly. The
  first BF16-HF-vs-Krasis divergence is `dt+bias`: HF BF16 add
  `l2=55.2308311462402`, Krasis `l2=55.2081756591797` (`0.0410196%` relative).
  Krasis matches the HF FP32 add exactly and matches HF FP32 softplus within
  `1.907e-6` l2 (`0.00000698%` relative), so the producer is dtype/precision at
  the bias-add boundary rather than bias slicing, stale state, summary artifact,
  or softplus formula.

- Added BF16-only layer-0 Mamba2 mixer internal localization for the exact
  corrected-38 Nemotron Super prompt. Added request-gated non-perturbing
  device summaries for gate split, xBC conv input, and dt softplus, and
  extended `./dev generate-reference --diagnose-layer0-internals` with matching
  HF Mamba2 split/conv/SSD/gated-norm/out-proj summaries. No behavior, HCS,
  tokenization, logits, layer-4, fallback, HQQ, or optimization patch was made.
  Build passed (`duration_s=138`), HF oracle exited `0` (`duration_s=294`),
  and prompt hash stayed `0xf62d9e4f5b39fdc7`. HF/Krasis match through
  input/norm/in-proj/gate/xBC split/raw dt; raw/conv C only show micro
  summary-reduction deltas with identical min/max. First material Mamba2
  mismatch is `dt_softplus`: HF `l2=27.334184646606445`, mean
  `0.9187623858451843`; Krasis `l2=27.33255386352539`, mean
  `0.9188207983970642` (`0.005966%` relative l2 delta). SSD and later stages
  drift from there; device/HF summaries had no NaN/Inf and Dynamic HCS reported
  `copy_failures=0`.

- Added BF16-only layer-0 internal/output correctness localization for the
  exact corrected-38 Nemotron Super prompt. Added request-gated
  non-perturbing layer-0 internal prefill device summaries and built-command
  HF layer-0 internal summaries via
  `./dev generate-reference --diagnose-layer0-internals`. No behavior, HCS,
  tokenization, logits, or layer-4 patch was made. Build passed
  (`duration_s=134`), HF oracle exited `0` (`duration_s=322`), and prompt hash
  stayed `0xf62d9e4f5b39fdc7`. HF and Krasis match exactly through layer-0
  input/embedding and input norm: input/norm input
  `l2=1.0884668827056885`, norm output/mixer input
  `l2=34.60029602050781`, with identical mean/min/max. The first layer-0
  substage divergence is Mamba2 mixer output: HF
  `layer0_mixer_output l2=1.8554539680480957` versus Krasis
  `layer_mixer_out_last l2=1.8557093143463135` (`0.0137619%` relative l2
  delta). Final layer-0 output then differs by `0.0185983%`. Layer 0 has no
  layer-local MLP/router in the HF block (`block_type=mamba`). Device trace
  used post-prefill download only, had no NaN/Inf, and Dynamic HCS stayed
  clean with `copy_failures=0`.

- Added BF16-only oracle-vs-Krasis prefill/logit localization for the exact
  corrected-38 Nemotron Super prompt. Added request-gated non-perturbing
  all-layer prefill device summaries and built-command HF hidden-summary
  capture through `./dev generate-reference --diagnose-hidden-summaries`.
  No model behavior, layer-4, HCS, tokenization, or logit patch was made.
  Build passed (`duration_s=139`). The prompt hash stayed
  `0xf62d9e4f5b39fdc7`; current rebuilt no-trace and all-layer repeats were
  stable at `11745,1051` (`Here3`), while the HF BF16 oracle remains
  `1975,29896` (code fence plus `python`). Tokenization and embedding matched
  exactly: HF embedding and Krasis layer-0 post-attn residual both had
  `l2=1.0884668827056885` with identical mean/min/max. The first
  oracle-comparable hidden divergence is layer 0 output:
  HF `l2=2.1844191551208496` vs Krasis `l2=2.1848254203796387`
  (`0.0186%` relative delta). Final hidden scale is close
  (`151.6925` HF vs `152.1458` Krasis), but first-token logits diverge:
  HF top-1 `1975` at logprob `-0.41007333993911743`, Krasis top-1 `11745`
  at `-0.975254`, with token `1975` Krasis rank 2 at `-1.227741`.
  All-layer traces were stable across repeats, had no NaN/Inf, and Dynamic HCS
  stayed clean with `copy_failures=0`. This narrows the next BF16 target to
  layer-0 internal/output correctness rather than layer 4.

- Added exact corrected-38 Nemotron Super BF16 reference-oracle coverage
  through the built `./dev generate-reference` path. `./dev generate-reference`
  and `tests/generate_reference.py` now support `--raw-input-json` for exact
  raw `input_token_ids`, record the same FNV-1a token hash as the server, and
  store raw-input metadata in the reference contract. No layer-4 behavior
  patch was made. Because no Nemotron GGUF witness is available locally, the
  capture used the existing archived HF/Transformers path under explicit
  `KRASIS_ALLOW_ARCHIVED_HF_REFERENCE=1` as a forensic BF16 oracle for this
  exact prompt. The corrected payload has 38 tokens and hash
  `0xf62d9e4f5b39fdc7`. The oracle selects token `1975` (code fence) then
  `29896` (`python`), while current Krasis selects `11745,2259` (`Here </`).
  The divergence is at first-token/prefill logits: oracle step-0 top-1
  `1975` has logit `80.0` and logprob `-0.41007333993911743`, while Krasis
  step-0 top-1 is `11745` at logprob `-0.953198` and oracle token `1975` is
  only rank 2 at `-1.178693`. Oracle pre/post first-token diagnostics matched
  and had no NaN/Inf. This closes the exact-prompt reference coverage gap
  before any model behavior change.

- Added a BF16-only Nemotron Super non-perturbing layer-4 Mamba2 prefill
  trace. The new request-gated `debug_prefill_device_trace=true` path records
  CUDA-side hashes/summaries for layer-4 Mamba2 input, in-proj, raw and
  convolved x/B/C/dt, SSM state before/after zero, SSD output, gated norm,
  out-proj, and next-layer input, then downloads only after full prefill/LM
  head. Build passed (`duration_s=140`). Corrected 38-token no-trace repeats
  and device-trace repeats all used prompt hash `0xf62d9e4f5b39fdc7` and
  selected `11745,2259` (`Here </`), so this trace does not perturb execution
  back to the old selected-trace `Here3` path. All 21 device trace stages were
  hash-stable across three repeats, SSM state was zero after the zero stage,
  layer-4 SSD/out-proj/next-layer summaries were stable, no device stage had
  NaN/Inf, and `copy_failures=0`. No behavior patch was made; this rules out
  repeat nondeterminism and stale layer-4 SSM state in the unsynced path, while
  confirming the old host-synced selected trace was perturbing execution.

- Added a BF16-only Nemotron Super stable model/logit correctness diagnostic.
  No behavior patch was made. The exact `fibonacci` code-gen prompt is absent
  from the stored Nemotron BF16 reference output, so this pass records a
  reference coverage gap rather than inventing an oracle. Internal tracing
  found the first reproducible cross-run boundary at layer-4 Mamba2 prefill:
  the 10:10 lifecycle run selected `11745,1051` (`Here3`) with layer-4
  `mixer_out_last` hash `0x88f0fa0e649e3cd0`, while the 10:24 unselected
  debug-reference run selected `11745,2259` (`Here </`) with layer-4
  `mixer_out_last` hash `0x02659ffecab15fe2`; layers 0-3 matched bitwise.
  A selected layer-4 Mamba2 trace then selected `Here3` twice and matched the
  10:10 layer-4 hash, showing the existing selected trace is synchronization
  perturbing. The selected layer-4 substages were internally stable, SSM state
  was zeroed before SSD, GPU SSD matched host recompute within one BF16 ULP in
  one lane, no selected trace snapshot had NaN/Inf, and `copy_failures=0`.
  Next BF16 work needs non-perturbing/device-side trace or an exact witness
  prompt with available oracle data before patching layer-4 Mamba2.

- Fixed the BF16-only Nemotron Super startup heatmap/request lifecycle leak.
  `_build_heatmap()` now tears down collection-only HCS and calls
  `model.server_cleanup()` through the existing request cleanup path after
  internal heatmap prefill/decode, before server readiness. No Mamba-only zero
  hack, HCS policy change, prewarm, forced residency, fallback, HQQ, or
  optimization work was added. Build passed (`duration_s=7`). Replayed the
  corrected 38-token `max_tokens=2`, `top_k=1` payload with
  `debug_mamba2_state_lifecycle_trace=true`: all no-trace/debug-reference
  repeats used prompt hash `0xf62d9e4f5b39fdc7`, selected stable tokens
  `11745,1051` (`Here3`), kept prefill raw logit
  `72.41891479492188`, and kept step-1 logprob `-6.775564`. The first
  external request now starts with zero layer-0 Mamba2 registered decode state
  (`conv_state` hash `7645daf3d8142325`, `ssm_state` hash
  `f8e3e56ce9222325`), and cleanup leaves it zero. `copy_failures=0`; decode
  lows stayed `790 MB`; debug-reference prefill snapshots had no NaN/Inf, with
  only the expected suppressed-token `-inf`. This accepts the first-request
  lifecycle/determinism gate; full Nemotron BF16 text is still wrong.

- Added a BF16-only layer-0 Mamba2 first-request state lifecycle diagnostic for
  Nemotron Super. Added request-gated trace-only
  `debug_mamba2_state_lifecycle_trace=true`, which captures registered Mamba2
  decode buffer summaries at request start, around prefill, before decode,
  after decode, and after server cleanup. Request unset preserves normal
  behavior. Build passed (`duration_s=133`). Replayed corrected 38-token
  `max_tokens=2`, `top_k=1` requests with stable prompt hash
  `0xf62d9e4f5b39fdc7`. No-trace request 1 produced `11745,11745`
  (`HereHere`); no-trace request 2+ and all debug-reference repeats stabilized
  at `11745,7297` (`Here)`). Request 1 already had nonzero registered
  layer-0 Mamba2 decode state before its own prefill (`conv_state l2=241.85`,
  `ssm_state l2=6130.47`), and that state was unchanged through prefill and
  HCS reload. Request cleanup zeroed both buffers, and every later request
  started zero. Source inspection shows `_build_heatmap()` runs held-out
  prompts with `rust_prefill_tokens()` and `gpu_generate_batch()` without
  request cleanup before server readiness, while warmup/calibration do call
  cleanup. `copy_failures=0`, no lifecycle NaN/Inf, and decode lows stayed
  `790 MB+`. No behavior patch, blind state zero/preserve, HCS policy change,
  prewarm, residency forcing, fallback, HQQ, or optimization work was done.
  Next BF16 target is a generic Mamba2 state lifecycle fix for startup
  heatmap/request handoff.

- Added a BF16-only decode early-layer HCS residency math-boundary diagnostic
  for Nemotron Super. Added request-gated trace-only `debug_decode_early_trace`
  capture for layer-0 decode input/output, layer-1 router/expert input,
  per-MoE-layer resident/cold source summaries, and layer-0 Mamba2 decode
  substages. Request unset preserves normal behavior. Both builds passed
  (`duration_s=133` each). Replayed corrected 38-token
  `max_tokens=2`, `top_k=1` requests with stable prompt hash
  `0xf62d9e4f5b39fdc7`. No-trace request 1 produced `11745,13144`
  (`Here-of`) with decode-start resident hash `6cdc483e704e3db0`; no-trace
  request 2+ and all debug-reference repeats stabilized at `11745,7155`
  (`Here https`) with hash `687a81dbd206f25e`. The first visible math
  difference is before layer-1 HCS: layer-0 decode input and `in_proj` are
  identical, but request 1 enters layer-0 Mamba2 with nonzero recurrent state
  (`conv_state l2=233.06`, `ssm_state l2=6044.81`), while the stable cycle
  enters with zero conv/SSM state. Per-layer source summaries show
  `hcs_swap_count=0`, `cold_kernel_matches_expected_activation=true`, and
  `relu2` resident/cold kernels; `copy_failures=0`; no nonfinite early trace
  snapshots; decode lows stayed `772 MB+`. No behavior patch, HCS policy
  change, prewarm, residency forcing, fallback, HQQ, or optimization work was
  done. Next BF16 target is decode Mamba2 state reset/handoff for request 1,
  not resident-vs-cold expert equivalence.

- Added a BF16-only startup HCS warm-state/residency diagnostic for Nemotron
  Super. Added request-gated trace-only HCS transition capture through
  `debug_hcs_transition_trace=true`, exposing server-side resident hashes
  before prefill eviction, after eviction, after HCS reload, plus decode-start,
  decode-end, and per-promotion incoming/victim slot metadata. Request unset
  preserves normal behavior. Build passed (`duration_s=133`). The existing
  `0833` artifacts showed a unique first-request resident hash but were too
  coarse to identify the transition cause. Fresh corrected 38-token repeats
  showed startup begins from the heatmap/startup state (`5106` residents), the
  first prefill reload creates a unique decode-start set (`4002`,
  `58af61182dea4240`), request 1 dynamic promotions end at
  `56284ef8ee3213c4`, and request 2+ converge to a stable cycle:
  `f78210c63cde745e` before prefill, `a7b70c9cf53868b6` after reload and at
  decode start, then `f78210c63cde745e` after decode. The first decode-start
  set differs from the stable set by exactly `10` resident experts each way.
  Layer-1 route/MoE data follows the same pattern: request 1 differs
  (`d825afbcb7e274b7`, `l2=0.0769`), request 2+ and debug-reference repeats
  are identical (`3511f6bd9c96b81d`, `l2=0.00709`). No HCS policy patch,
  prewarming, residency forcing, Dynamic HCS disable, fallback, HQQ, or
  optimization work was done. `copy_failures=0`; decode lows stayed
  `804 MB+`. Next BF16 target is a generic initial-HCS convergence policy
  decision, not cold activation or pending-copy visibility.

- Fixed the BF16 decode cold expert activation path for Nemotron Super
  `relu2` layers generically. Sequential resident and cold DMA decode now route
  activation type `1` through a single-expert `relu2_w2_accum` helper that
  reuses the existing resident `relu2_w2_batched` CUDA math and applies the
  route weight with `weighted_add_bf16`; no model, layer, prompt, expert, or
  GPU hardcode, no fallback, no HCS residency forcing, and no graph/HCS/decode
  policy change. Build passed (`duration_s=133`). Replayed corrected
  38-token `max_tokens=2`, `top_k=1` no-trace and debug-reference requests on
  one server process. All 16 requests used prompt hash
  `0xf62d9e4f5b39fdc7`, stable prefill top-1/logit
  `11745`/`Here`/`72.19525146484375`, no new NaN/Inf beyond the expected
  suppressed-token `-inf`, and clean HCS/copy (`copy_failures=0`). Trace now
  reports resident batched `relu2_w2_batched_int4`, resident sequential
  `relu2_w2_accum_int4`, cold `relu2_w2_accum_int4`, and
  `cold_kernel_matches_expected_activation=true`. Full stability acceptance is
  still partial: no-trace request 1 selected step-1 token `4179`, then
  no-trace request 2 and all later debug/no-trace repeats stabilized at
  `63467`. Next BF16 target is the remaining startup HCS warm-state/residency
  determinism boundary.

- Added a BF16-only layer-1 decode HCS resident-vs-cold equivalence
  diagnostic for Nemotron Super. Added request-gated trace-only capture through
  `debug_decode_hcs_equiv_trace=true`, returning layer-1 routed
  experts/weights, resident-vs-cold source, resident pointers, cold DMA copy
  proof, kernel labels, output hashes, and final MoE output hashes; request
  unset preserves default behavior. Build passed (`duration_s=133`). Replayed
  the corrected 38-token `max_tokens=2`, `top_k=1` payload on one server
  process. All 16 requests used prompt hash `0xf62d9e4f5b39fdc7`, stable
  prefill token `11745`/`Here`, no nonfinite logits except the expected
  suppressed-token `-inf`, and clean HCS/copy (`copy_failures=0`). The first
  two no-trace requests still diverged at step 1 (`65981`, then `1034`) before
  no-trace request 3 and all debug-reference repeats stabilized at `1041`. The
  concrete boundary is layer-1 decode cold expert execution: activation type
  `1` expects `relu2`, resident HCS experts use `relu2_w2_batched_int4`, but
  cold DMA experts use `fused_silu_accum_int4` and report
  `cold_kernel_matches_expected_activation=false`. Copy completion proof was
  clean (`CUDA_SUCCESS/CUDA_SUCCESS`), so this is not a pending-copy visibility
  issue. No HCS/decode behavior patch, HQQ, optimization, fallback,
  graph/HCS policy, calibration, chunk-size, or VRAM-policy change was made.
  Next BF16 target is a generic cold-kernel activation-selection fix and rerun
  of the step-1 stability gate.

- Added a BF16-only decode step-1 HCS warm-state diagnostic for Nemotron
  Super. Added request-gated trace-only raw-reference capture through
  `debug_decode_state_trace=true`, returning decode-start HCS residency and
  cold-load files, dynamic HCS counters, and raw top logits from the last
  decode step; request unset preserves default behavior. Build passed
  (`duration_s=131`). Replayed the corrected 38-token `max_tokens=2`,
  `top_k=1` payload on one server process. All 16 requests used stable prompt
  hash `0xf62d9e4f5b39fdc7`, stable prefill top-1 `11745`/`Here`, and no
  nonfinite logits except the expected one suppressed-token `-inf`; HCS/copy
  stayed clean (`copy_failures=0`) and decode lows stayed `766 MB+`. Step-1
  logits differed before sampling and correlated with decode-start HCS
  residency: no-trace request 1 had resident hash `826d79a535260643`, `131`
  cold loads, and selected `13576`; request 2 had hash `2b1fb25e440c2ca1`,
  `114` cold loads, and selected `1034`; request 3 through all debug-reference
  repeats had hash `506940081b895e82`, `117` cold loads, and stable `1034`.
  Reload was synchronous, `reload_pending_at_decode_start=false`, Mamba2 decode
  used the ungraphed path, and the sampler followed top-1. No decode behavior
  patch, HQQ, optimization, fallback, graph/HCS policy, calibration,
  chunk-size, or VRAM-policy change was made. Next BF16 target is HCS
  resident/cold path equivalence or dynamic-HCS warm-state determinism at the
  layer-1 decode MoE boundary.

- Fixed BF16 prefill routed MoE scatter determinism for Nemotron Super by
  replacing FP32 `atomicAdd` accumulation with deterministic slot-order
  accumulation in the Marlin sidecar path, PTX weighted scatter, sequential
  scatter fallback, and legacy fused scatter test path. The fix is
  shape-derived from `M`, `topk`, `total_sorted`, and hidden width; it does not
  hardcode layer 1, Nemotron, model names, prompt lengths, expert counts, or
  GPU indices, and it adds no fallback. Build passed, including Marlin MoE
  sidecar rebuild (`duration_s=134.830`); a final review rebuild after the
  legacy scatter patch also passed (`duration_s=135`). Corrected 38-token
  `max_tokens=2`, `top_k=1` repeats on one server process had stable prompt
  hash `0xf62d9e4f5b39fdc7`, stable prefill top-1/logit (`11745`/`Here`, raw
  logit `73.37806701660156`), stable debug-reference layer-1
  `moe_accum_post_routed_scatter_full_chunk` row-summary digest
  `47a7c866e6237b55c7aa1b2a1a9289afec113d8c999555b603d0b64325725803`, no
  nonfinite trace snapshots, and clean HCS/copy (`copy_failures=0`). The gate
  is only partially closed: no-trace run 1 still diverged at decode step 1
  (`129742`), no-trace run 2 selected `1034`, and later no-trace plus all
  debug-reference repeats stabilized at `1041`. Next BF16 target is decode
  step-1 first-request/HCS warm-state determinism, not prefill scatter.

- Added a BF16-only layer-4 Mamba2 SSD determinism/state diagnostic for
  Nemotron Super. Added selected-layer trace-only SSD internals behind
  `KRASIS_REFERENCE_MAMBA_TRACE_LAYERS`: SSM state before zero, after zero,
  after SSD, GPU SSD last-row versus host recompute, and full-row raw/convolved
  x/B/C/dt hashes. Request/env unset preserves default behavior. Both builds
  passed (`duration_s=131` each). Repeated corrected 38-token requests used
  prompt hash `0xf62d9e4f5b39fdc7`, fresh prefill/state-reset proof stayed
  clean, and HCS/copy had `copy_failures=0`. The SSM state before zero was
  stable, after-zero was all zero with stable hash `0xf8e3e56ce9222325`, and
  GPU SSD output matched host recompute within BF16 rounding (`max_abs=0.000244`).
  Full-row hashes showed the layer-4 SSD variation comes from a variable
  earlier-token input, first visible at row 33 before conv. Whole-trace row
  hashes identified the first upstream nondeterministic producer as layer-1
  `moe_accum_post_routed_scatter_full_chunk`: pre-scatter accumulator, fused
  expert output, top-k weights, and top-k ids were stable, while post-scatter
  FP32 accumulator hashes varied. No behavior patch, HQQ, optimization,
  fallback, graph/HCS, calibration, chunk-size, decode, or VRAM-policy change
  was made. Next BF16 target is deterministic layer-1 MoE routed scatter
  accumulation, not layer-4 SSD.

- Added a BF16-only corrected 38-token determinism localization diagnostic for
  Nemotron Super. Repeated identical raw-reference requests used
  `max_tokens=2`, `top_k=1`, one focused server process, and corrected prompt
  hash `0xf62d9e4f5b39fdc7`. State-reset proof showed fresh prefill, LA state
  zeroing, cleanup-before-request, KV position `38`, chunk/scratch `128`, and
  clean HCS/copy (`copy_failures=0`). No-trace and debug-reference both
  diverged before sampling: step-0 selected `11745`/`Here`, but prefill logits
  varied and step-1 tokens changed across runs. Added trace-only selected
  Mamba substage support through `KRASIS_REFERENCE_MAMBA_TRACE_LAYERS`; env
  unset preserves default behavior. Layer-2 Mamba substages were stable, while
  layer-4 selected Mamba tracing identified the first BF16-visible last-token
  producer as `layer4_mamba2_ssd_out_last`. No behavior patch, HQQ,
  optimization, fallback, graph/HCS, calibration, chunk-size, decode, or
  VRAM-policy change was made.

- Added a BF16-only corrected 38-token decode divergence diagnostic for
  Nemotron Super. Existing endpoints lacked exact normal-chat generated token
  IDs, so extended the existing request-gated
  `debug_first_token_boundary=true` chat diagnostic to include completion token
  IDs and per-step decode top-k/logprobs; request unset preserves normal chat
  behavior. Build passed (`20260615_0517_nemotron_super_corrected38_decode_divergence_build.log`,
  `duration_s=131`). Two short deterministic runs each were sent through
  normal chat, no-trace raw reference, and debug-reference, all with the
  corrected 38-token `code_gen` prompt hash `0xf62d9e4f5b39fdc7`. Normal chat
  and no-trace reference selected `11745`/`Here` on step 0 but diverged at
  decode step 1; debug-reference was also unstable, with one run selecting
  `3149`/`def` at step 0 and another selecting `Here`. The first
  post-first-token boundary for the `Here` paths is decode step 1, with
  materially different top-1 distributions (`72348`, `1034`, `101380`, or
  `1064` depending on run/path). No behavior patch, HQQ, optimization,
  fallback, graph/HCS, calibration, chunk-size, decode, or VRAM-policy change
  was made. HCS/copy stayed clean (`copy_failures=0`); decode lows stayed
  above margin, but chat prefill again dipped below `600 MB` (`482 MB`,
  `448 MB`).

- Reran the Nemotron Super BF16 layer-level diagnostic with the corrected
  chat-derived 38-token `code_gen` payload, invalidating prior 37-token layer
  conclusions for the real chat prompt. No behavior patch was made. The
  no-selected-layer trace selected `11745`/`Here`, stayed finite end-to-end,
  and had finite non-uniform logits. The earliest material remaining finite
  scale boundary is still layer 32 MLP (`pre_mlp l2=42.75`, max `6.81` ->
  `post_mlp l2=50.05`, max `18.125`, residual estimate `l2=90.63`, max
  `23.25`). Reused selected-layer layer-32 tracing with the same 38-token
  payload; top-k was normalized (`sum=1.0`, effective scaled sum `5.0`), routed
  W2/scatter was finite, latent-up row was `l2=39.96`, shared row was
  `l2=25.20`, and the post-add hidden row was `l2=50.10`, max `18.25`. No
  concrete shape/layout/scaling bug was identified inside layer 32. HCS/copy
  health stayed clean (`copy_failures=0`), but prefill low-water again dipped
  below the `600 MB` safety margin (`450 MB` and `482 MB`), so the measured
  VRAM budget issue remains visible. No HQQ, optimization, fallback,
  graph/HCS, calibration, chunk-size, decode, or VRAM-policy change was made.

- Added a BF16-only first-token boundary diagnostic for Nemotron Super. The
  existing endpoints could not expose rendered prompt IDs and first-token
  logits cleanly, so added request-gated trace-only response fields:
  `debug_first_token_boundary=true` for chat and `debug_prompt_trace=true` for
  raw reference. Env/request unset preserves default behavior. Build passed
  (`20260615_0442_nemotron_super_first_token_boundary_trace_build.log`). The
  `code_gen` chat path renders `38` tokens with hash
  `0xf62d9e4f5b39fdc7` and selects `11745`/`Here`; the older raw reference
  payload had `37` tokens, hash `0x4cd1456156df3a70`, and is exactly the chat
  token list with the first token `1010` dropped. With the chat-derived
  38-token payload, no-trace raw reference selected `Here` twice and
  debug-reference selected `Here` twice; the previous `def` flip did not
  reproduce. Old 37-token no-trace/debug replays also selected `Here`. The
  first-token logits are not bit-stable between requests and the top-token
  gaps remain small, so the next BF16 target is repeated hidden/logit
  stability on identical 38-token inputs, not a model-math patch yet. No HQQ,
  optimization, fallback, graph/HCS, calibration, chunk-size, decode, or
  VRAM-policy change was made.

- Added a BF16-only post-topk-fix layer-32 trace diagnostic for Nemotron Super.
  The existing `20260614_2350` trace stayed finite from layer 27 onward and
  showed the first remaining coarse scale boundary at layer 32 MLP
  (`pre_mlp l2=42.01`, max `6.47` -> `post_mlp l2=48.14`, max `18.25`).
  Existing selected-layer tracing split the layer but lacked row-level shared
  output, so added a trace-only selected-layer snapshot for
  `layer32_moe_shared_output_bf16_row36_pre_latent_add`; env unset keeps
  default tracing unchanged. Build passed
  (`20260615_0002_nemotron_super_layer32_moe_trace_build.log`,
  `duration_s=130`). The rebuilt trace shows normalized top-k
  (`sum=1.0000001`, effective scaled sum `5.0000005`), routed W2/scatter at
  modest scale, latent-up row `l2=39.85`, shared row `l2=21.75`, and post-add
  hidden row `l2=48.28`, all finite. Normal chat and no-trace raw reference
  still select `11745`/`Here`, but debug-reference trace with selected-layer
  sync flips the first token to `3149`/`def`; those top two logits are only
  `0.075` apart. No behavior fix, HQQ, optimization, fallback, graph/HCS,
  calibration, chunk-size, decode, or VRAM-policy change was made.

- Fixed Nemotron Super BF16 prefill sigmoid top-k normalization for models with
  `norm_topk_prob=true`. The layer-26 latent-up investigation showed
  `fc2_latent_proj` itself had the expected safetensor shape/orientation
  (`[4096,1024]`, `C = A @ W^T`) and matched decode BF16 GEMV layout; the
  concrete bug was prefill routing scale. Before the fix, layer-26 row-36
  sigmoid top-k weights summed to `5.6663337` and scatter applied
  `routed_scaling_factor=5.0`, producing effective sum `28.3316689` where the
  reference expects `5.0`. Prefill now normalizes sigmoid top-k weights when
  `moe_norm_topk_prob=true`, before routed scatter scaling. Build passed
  (`20260614_2341_nemotron_super_prefill_sigmoid_topk_norm_fix_build.log`,
  `duration_s=135`); startup then exposed a loader allow-list miss for the new
  PTX symbol, fixed by the loader rebuild
  (`20260614_2350_nemotron_super_prefill_sigmoid_topk_norm_loader_fix_build.log`,
  `duration_s=131`). The rerun passed the narrow gate: row-36 top-k sum
  `0.99999994`, effective routed sum `4.9999995`, and layer-26 latent-up
  dropped from prior `l2=96.68/97.25` to `l2=23.82/24.99`. Output is still not
  correct: chat begins `Here...` and hits `length`, while raw reference selects
  token `11745`; all snapshots are finite. No HQQ, optimization, fallback,
  graph/HCS, calibration, chunk-size, decode, or VRAM-policy change was made.

- Added a BF16-only selected-layer MoE trace diagnostic and localized the next
  Nemotron Super finite-scale producer after the fixed layer-0 Mamba2 path.
  Existing `20260614_2253` trace was finite end-to-end and showed the first
  later coarse jump at layer 26 MLP. The new trace-only selector
  `KRASIS_REFERENCE_MOE_TRACE_LAYERS=26` reuses the reference MoE snapshots
  with a dynamic layer prefix and includes the final prompt row; env unset keeps
  existing default trace behavior. Build passed
  (`20260614_2303_nemotron_super_layer26_moe_trace_build.log`,
  `duration_s=130`). The exact `code_gen` chat prompt is still wrong
  (begins `,a...`, finish `length`, `38/512` tokens), and raw reference
  selected token `1044`, but every traced snapshot is finite. Layer-26 row-36
  split identifies the next concrete producer: routed W2 post-call `l2=24.41`,
  routed accumulator after scatter `l2=50.34`, shared output `l2=3.10`, then
  LatentMoE latent-up/`fc2_latent_proj` output `l2=101.61`, max `49.25`.
  Final hidden/logits are finite and not uniform. No behavior fix, HQQ,
  optimization, fallback, graph/HCS, calibration, or chunk-size change was
  made.

- Implemented the missing Nemotron Super BF16 prefill Mamba2 reference stages.
  The Rust prefill path now applies shape-derived causal conv+silu over xBC
  before SSD, shape-derived gated group RMSNorm before `out_proj`, and wires
  the registered `conv1d.bias` pointer into prefill layer weights when present.
  Build passed (`20260614_2232_nemotron_super_mamba2_conv_norm_build.log`,
  `duration_s=134`). Review fixed the gated-RMSNorm launch to use a
  power-of-two block for shape portability and rebuilt successfully
  (`20260614_2232_nemotron_super_mamba2_conv_norm_review_fix_build.log`,
  `duration_s=130`). Rerunning the exact `code_gen` diagnostic on the rebuilt
  binary moved chat output from `_API` to wrong text beginning `2023...`,
  finishing after `23` completion tokens, so full correctness is still broken.
  Layer-0 scales are now finite and small: raw `x/B/C/dt`
  `l2=109.03/57.39/50.08/29.98`, convolved `x/B/C`
  `l2=71.74/6.11/5.97`, SSD `l2=624.59`, gated norm `l2=1.21`, and
  `out_proj` `l2=1.86`, all with `nan_count=0`; raw reference selected token
  `1267` and still generated junk. VRAM budgeting still needs measured
  follow-up: the first focused chat prefill hit `386 MB` min free, and the
  rebuilt reference request produced a VRAM monitor low of `320 MB`, both below
  the configured `600 MB` safety margin. No HQQ, optimization, fallback,
  graph/HCS, calibration, or chunk-size change was made.

- Fixed the Nemotron Super BF16 prefill Mamba2 SSD `A_log` interpretation.
  The prefill SSD scan now uses reference-compatible `A = -exp(A_log)` instead
  of treating the checkpoint `A_log` tensor as already-linear `A`. Build passed
  (`20260614_2219_nemotron_super_mamba2_alog_fix_build.log`,
  `duration_s=134`). Rerunning the exact `code_gen` diagnostic with the
  existing layer-0 Mamba2 substage trace cleared the narrow acceptance gate:
  `layer0_mamba2_ssd_out_last` dropped from the previous `l2=1.57e38`, max
  `4.32e37`, min `-7.51e37` to finite normal-scale `l2=29884.87`, max `9600`,
  min `-9344`, `nan_count=0`; `out_proj` was also finite (`l2=34717.77`).
  The chat output is still wrong (`_API`) and the raw reference trace still
  generates junk, so full BF16 correctness is not fixed. The next target is the
  missing reference Mamba2 causal conv+silu over xBC and gated group RMSNorm
  stages before `out_proj`. No HQQ, optimization, fallback, graph/HCS,
  calibration, or chunk-sizing change was made.

- Added a BF16-only trace diagnostic for Nemotron Super layer-0 Mamba2 prefill
  and reran the exact failing `code_gen` prompt. Build passed
  (`20260614_2204_nemotron_super_mamba2_layer0_substage_build.log`,
  `duration_s=129`). The chat path again returned `后汉书`, and the raw
  reference trace again selected token `131071`. The new substage split shows
  projection and raw extraction are finite at normal scale (`in_proj l2=165.54`,
  raw `x/B/C/dt l2=109.03/57.39/50.08/29.98`), then the current Rust Mamba2
  `ssd_sequential` stage is the first catastrophic producer (`ssd_out
  l2=1.57e38`, max `4.32e37`, min `-7.51e37`) before `out_proj` reaches
  `l2=1.92e38`. Metadata also confirms the current Rust path omits the
  reference Mamba2 causal conv on xBC and gated group RMSNorm stages, and the
  next BF16 correctness gate should address Mamba2 prefill algorithm/parameter
  interpretation before any HQQ or optimization work.

- Localized the remaining Nemotron Super `code_gen` network collapse to the
  prefill layer-0 Mamba2 mixer path. A focused test-endpoint run reproduced the
  exact prompt from `tests/test_network.py`: `/v1/chat/completions` returned
  `后汉书`. The raw reference trace for the rendered prompt selected token
  `131071` (`后汉书`) during prefill, before decode. Tensors were finite, but
  layer-0 `mixer_out_last` jumped from normal input-norm scale (`l2=34.6`) to
  catastrophic BF16-finite values (`l2=1.92e38`, max `1.19e37`, min
  `-1.40e37`); final hidden before LM head then collapsed to all zeros, making
  LM-head logits uniform `0.0` except the suppressed token. No HQQ or
  optimization work was started.

- Fixed the Nemotron Super prefill LatentMoE routed expert path. The
  diagnostic gate showed routed prefill was feeding full hidden-width rows
  (`4096`) into routed experts whose Marlin cache was built for
  `moe_latent_size=1024`, corrupting layer-1 W1 output and producing all-NaN
  final logits with token `131071` (`后汉书`). Prefill now applies the
  model-provided `fc1_latent_proj` before routed expert compute, runs fused
  routed MoE at latent width, projects the routed accumulator back to hidden
  width with `fc2_latent_proj`, and then combines with the full-hidden shared
  expert output. Standard MoE layers keep the existing hidden-width path, and
  LatentMoE layers fail visibly if either latent projection is missing. Build
  passed (`20260614_2215_nemotron_super_prefill_latent_moe_build.log`,
  `duration_s=129`). The raw `[0]` diagnostic then returned finite logits:
  selected token `88810` (`Spielbericht`), prefill logits `nan_count=0`,
  `finite_count=131071`, final hidden `4096/4096` finite, and layer-1 routed
  W1/W2 stages finite at latent width. Full Nemotron network and
  witness-style correctness remain pending.

- Ran the full Nemotron Super normal-path test after the prefill LatentMoE
  fix. `./dev test tests/nemotron-super-bf16kv-a16.conf` completed the
  benchmark with `1585.8` prefill, `53.52` internal decode, `71.79` HTTP, HCS
  `5106/20480`, min free decode VRAM `916 MB`, and zero Dynamic HCS copy
  failures. Network correctness still failed `2/14`: many prompts still
  collapse to `后汉书`, while others produce repetitive junk tokens, so the
  normal chat/network path remains the active correctness blocker despite the
  finite raw `[0]` diagnostic. The run also recorded one prefill VRAM
  low-water warning at `594 MB`, `6 MB` below the configured `600 MB` safety
  margin; that budget miss needs measurement before any performance work.

- Advanced Nemotron Super bring-up past cache load, CUDA warmup, Mamba2
  startup, calibration, graph selection, and the first benchmark OOM. Added
  Mamba2 BF16 projection registration for Rust prefill, passed actual Mamba2
  `conv_dim` into Rust decode descriptors, derived Mamba2 prefill dimensions
  from the registered descriptor instead of hidden-size assumptions, and
  structurally disabled CUDA graph decode for models with Mamba2 layers so they
  use the existing non-graph path without requiring manual `KRASIS_NO_GRAPH=1`.
  Also mapped zero-head MoE-only GQA placeholders to prefill MoE-only/pass
  layers instead of launching FlashAttention on placeholder weight id 0.
  `./dev build` and the startup/calibration preflight passed after these fixes;
  with `KRASIS_NO_GRAPH` unset the calibration reported short decode min
  `15544 MB`, long decode min `15538 MB`, decode HCS budget `14906 MB`, and
  long prefill min `1704 MB`. These changes are scoped to Mamba2 or zero-head
  MoE-only placeholder cases and do not change non-Mamba2 graph capture.

- Fixed Nemotron Super multi-chunk prefill cold-staging OOM without hardcoded
  prompt, model, GPU, or VRAM constants. The diagnostic run showed HCS eviction
  correctly freed all soft experts before the 25K benchmark prompt, but the
  chunk planner used the measured post-scratch reserve for a split prompt; the
  first chunk succeeded and the second chunk had no HCS left before allocating
  `185` cold-staging slots. Prefill chunk sizing now raises the runtime reserve
  to the dimension-derived worst-case cold-staging requirement only when the
  initial scratch cap would split the prompt. `./dev build` passed with
  `20260614_1919_nemotron_super_multichunk_cold_reserve_build.log`, and the
  follow-up functional benchmark completed without OOM.

- Nemotron Super functional status after the cold-staging reserve fix:
  `./dev test tests/nemotron-super-bf16kv-a16.conf` completed the benchmark on
  one RTX 5090 with `1548.5` prefill, `51.63` internal decode, `105.69` HTTP,
  HCS `5336/20480`, min free VRAM `922 MB`, and zero Dynamic HCS copy failures.
  The network correctness suite is still failing: `2/13` passed and nearly
  every response was exactly `后汉书`, with the final streaming multi-turn case
  returning `HTTP 0`. Runtime is now stable enough to benchmark, but output
  correctness remains the active blocker; next work should gather first-token,
  logit, or witness-style data before changing model-path kernels.

- Started `nemotron-dev` bring-up for `NVIDIA-Nemotron-3-Super-120B-A12B-BF16`
  with a new test config `tests/nemotron-super-bf16kv-a16.conf` using BF16
  attention/KV for first load preflight and INT4 experts. The initial contract
  run was intentionally blocked before server/model load because the available
  Super reference artifact is archived HF/Transformers-backed, so it was not
  used as a correctness authority. The first runtime/cache preflight built the
  Super GPU INT4 Marlin expert cache (`experts_marlin_int4_g128_calamax.bin`,
  59.0 GB in 138s) and then failed to load it with `All Marlin INT4 cache
  attempts failed`.

- Fixed Nemotron-H LatentMoE routed expert Marlin cache sizing/loading. The
  cache writer used the actual routed expert tensor width
  (`moe_latent_size=1024`, `up_proj [2688,1024]`, `down_proj [1024,2688]`),
  while the cache expected-size and read path still assumed full hidden width
  (`4096`) for routed experts. Added `moe_latent_size` to the Rust weight
  `ModelConfig` and use it only for routed Marlin expert byte sizing and
  routed Marlin layer reads; shared experts remain full hidden width. Existing
  models without `moe_latent_size` keep full-hidden routed expert behavior.
  `./dev build` passed (`20260614_1800_nemotron_super_latent_marlin_fix_build.log`,
  `duration_s=129`), and rerunning
  `./dev run tests/nemotron-super-bf16kv-a16.conf --build-cache` loaded the
  existing 59 GB cache in 36s and reached `BUILD CACHE COMPLETE`. No fallback,
  BF16 expert path, cache deletion workaround, HCS change, calibration change,
  or chunk-sizing change was added.

- Verified the restored Gemma HQQ4 k4v4 clean baseline after rejecting the
  HD512 q2-BC32 K/V-alias prototype. The verification gate was recorded in
  `krasis-internal/DEBUGLOG.md` before launch. Ran
  `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` with only
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and
  `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` enabled, with attribution clocks and q2
  prototype envs explicitly unset. The run passed `14/14`,
  `ALL TESTS PASSED`, and produced `5619.6` prefill, `92.43` internal decode,
  and `155.69` HTTP, confirming restored speed close to the accepted
  `5594.9/92.42` k4 marker. HCS stayed `3840/3840`, min free decode VRAM was
  `11474 MB`, and all Dynamic HCS rows had `copy_failures=0`. The log had no
  prefill attribution or q2 candidate markers. Cleanup required `./dev kill`
  for a residual benchmark server/zombie child, after which no tmux/server
  process remained and GPUs were idle.

- Implemented and rejected the env-gated HD512 q2-BC32 K/V-alias prefill
  prototype behind `KRASIS_PREFILL_HD512_Q2_BC32_KV_ALIAS=1`. The gate was
  recorded in `krasis-internal/DEBUGLOG.md` before source edits. The default
  env-off q2-BC16 path and the existing capacity-blocked
  `KRASIS_PREFILL_HD512_Q2_BC32=1` path were kept separate. `./dev build`
  passed with `20260614_1339_gemma4_hqq4_hd512_q2_bc32_kv_alias_build.log`
  (`duration_s=133`). Required k4 attribution then passed correctness `14/14`
  with `5270.0` prefill, `92.42` internal decode, `160.15` HTTP, HCS
  `3840/3840`, min free decode VRAM `11474 MB`, and zero copy failures; the
  log explicitly selected `hd512_q2_mode=bc32_kv_alias`, `hd512_tile_cols=32`,
  and `hd512_q_heads_per_block=2`. Representative `11824`-token HD512 q2
  rows were `373.1-374.4 ms/layer` with K load `93.9-94.2`, softmax
  `58.7-58.9`, V load `93.9-94.2`, and PV `97.4-97.8`. The required clean
  k4 speed run passed correctness but failed the full gate: `4817.6` prefill,
  `92.38` internal decode, `160.30` HTTP versus accepted clean k4 baseline
  `5594.9/92.42/165.46`. Reverted the alias source changes only and rebuilt
  restored source successfully with
  `20260614_1357_restore_after_hd512_q2_bc32_kv_alias_reject_build.log`
  (`duration_s=132`). No k6v6 or QCN guard was run because clean k4 speed
  failed.

- Completed a design-only inspection after the q2-BC32 capacity block. Current
  clean markers are k4 `5594.9/92.42/165.46` and k6v6
  `5243.5/65.08/119.07`; the q2-BC32 prototype correctly failed visibly on
  this RTX 5090 because it requires `106496` bytes opt-in shared memory and the
  device reports `101376`. q2-BC24 was rejected as a safe narrow follow-up even
  though its derived current-layout shared memory is `88576` bytes, because the
  current q2 WMMA loops require a 16-column tile multiple and `BC=24` would
  require padded/remainder handling. Selected one future design-only follow-up,
  not implemented here: q2-BC32 with temporal K/V shared-memory aliasing under a
  new env gate such as `KRASIS_PREFILL_HD512_Q2_BC32_KV_ALIAS=1`. The derived
  shared-memory requirement is `73728` bytes, below the `101376` byte device
  limit, by loading K for QK and then reusing the same shared tile buffer for V
  before PV. No performance code was edited and no benchmark was run.

- Implemented the planned env-gated HD512 q2-BC32 prefill prototype behind
  `KRASIS_PREFILL_HD512_Q2_BC32=1`. The default env-off HD512 q2-BC16 path
  remains unchanged. The implementation adds q2-BC32 normal/timed CUDA entry
  points, Rust symbol registration, runtime shared-memory capacity checks, and
  visible failure instead of silent fallback when the env flag is set but the
  device cannot supply the derived q2-BC32 shared memory. `./dev build` passed
  with final-source log `20260614_1316_gemma4_hqq4_hd512_q2_bc32_build.log`
  (`duration_s=128`). The required first k4 attribution run was attempted with
  the accepted dual-norm + GPU-softcap baseline plus `KRASIS_PREFILL_TIMING=1`,
  `KRASIS_PREFILL_HD512_KERNEL_CLOCKS=1`, and
  `KRASIS_PREFILL_HD512_Q2_BC32=1`, but it stopped before benchmark/correctness
  because the selected RTX 5090 reports `101376` bytes opt-in shared memory and
  q2-BC32 requires `106496` bytes. This confirmed the intended visible capacity
  gate and no hidden fallback to q2-BC16. No clean speed, k6v6, or QCN guard
  was run because k4 attribution could not start on this hardware.

- Recorded a design-only HD512 q2 prefill redesign plan before any performance
  code edit. The gate was written to `krasis-internal/DEBUGLOG.md` first.
  Source/log inspection confirmed the active long-row path is the Gemma HD512
  q2 kernel with `BR=16`, `BC=16`, `q_heads_per_block=2`, and q2 K/V sharing.
  k4 HD512 kernel clocks showed capped long-row per-layer total
  `627.0-628.4 ms`, dominated by PV `246.9-247.3 ms`, K/V tile load
  `177.7-178.0 ms`, and softmax/rescale/probability `156.9-157.2 ms`; k6v6
  attribution showed the same HD512 `custom_tiled` area at
  `1285.4-1287.2 ms` over five calls on capped `10524` rows. Selected one
  concrete future prototype plan, not an implementation:
  `KRASIS_PREFILL_HD512_Q2_BC32=1`, an opt-in q2 retile that keeps `BR=16`,
  `q_heads_per_block=2`, online-softmax math, chunk sizing, calibration, HCS,
  decode graph behavior, and non-Gemma behavior unchanged, while widening only
  the K/V tile to `BC=32` when runtime opt-in shared memory supports the
  derived requirement. If the env is set and shared memory is insufficient, the
  future prototype must fail visibly rather than silently falling back to
  q2-BC16. No performance code was edited and no benchmark was run in this
  design pass.

- Ran a focused source-free Gemma4 HQQ4/k6v6 attribution pass before any HD512
  redesign work. The gate and full tmux command were recorded in
  `krasis-internal/DEBUGLOG.md` before launch. The run used
  `tests/gemma-4-4-hqq4-k6v6-a16.conf` through `./dev test --timing` with
  accepted gates `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and
  `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`, plus existing attribution clocks
  `KRASIS_PREFILL_TIMING=1`, final, MoE-route, route-prep, MoE-W2, W2-preload,
  and GQA clocks. The heavy per-block HD512 kernel clock and rejected candidate
  envs were explicitly unset. Full log
  `20260614_1255_gemma4_hqq4_k6v6_dual_norm_softcap_attr_timing.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5537.2` prefill, `64.04` internal decode,
  `116.30` HTTP, HCS `3840/3840`, min free decode VRAM `11748 MB`, zero cold
  DMA, and `copy_failures=0`. The split showed capped `10524`-token prefill
  dominated by HD512 `custom_tiled` launch `1285.4-1287.2 ms` over 5 calls,
  plus GQA projection `157.8-157.9 ms`, FA2 `112.3 ms`, O projection
  `82.8-83.1 ms`, and MoE `257.3-257.5 ms`; decode rows showed total
  `15.43/15.89/16.28 ms/tok`, MoE expert `3.34`, GQA path
  `3.07/3.34/3.87`, route/topk `2.86`, and final segment `1.05`. Decision:
  attribution-only. The data did not expose a distinct non-rejected
  `>0.2 ms/tok` target; no source changes were made and no performance
  candidate was started.

- Ran a clean timing-off Gemma4 HQQ4/k6v6 baseline check before any broad
  HD512 redesign planning. Current k4 clean marker was `5594.9` prefill,
  `92.42` internal decode, `165.46` HTTP. Inspected available Gemma HQQ k6v6
  configs and selected `tests/gemma-4-4-hqq4-k6v6-a16.conf` for comparability
  with the accepted HQQ4 k4 marker. Only
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and
  `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` were enabled; known attribution clocks
  and rejected candidate envs were explicitly unset, `--timing` was omitted,
  and the metric gate plus full tmux command were recorded in
  `krasis-internal/DEBUGLOG.md` before launch. Full log
  `20260614_1247_gemma4_hqq4_k6v6_dual_norm_softcap_clean_speed.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5243.5` prefill, `65.08` internal decode,
  `119.07` HTTP, HCS `3840/3840`, min free decode VRAM `11748 MB`, zero cold
  DMA, and `copy_failures=0`. Startup built and loaded the Gemma Marlin expert
  cache for k6v6 before benchmark timing. No source changes were made and no
  performance candidate was started.

- Ran a clean timing-off Gemma4 HQQ4/k4v4 accepted-baseline speed check with
  only `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and
  `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` enabled. Known attribution clocks and
  rejected candidate envs were explicitly unset, `--timing` was omitted, and
  the metric gate plus full tmux command were recorded in
  `krasis-internal/DEBUGLOG.md` before launch. Full log
  `20260614_1238_gemma4_hqq4_k4v4_dual_norm_softcap_clean_speed.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5594.9` prefill, `92.42` internal decode,
  `165.46` HTTP, HCS `3840/3840`, min free `11474 MB`, zero cold DMA, and
  `copy_failures=0`. This replaces the anomalous accepted experimental
  timing-off marker `4931.0/92.31`; startup rebuilt the Gemma Marlin expert
  cache, but the benchmark speed rows were timing-off and attribution-free. No
  source changes were made.

- Recorded a design-only inspection of the Gemma4 HQQ4/k4v4 HD512 q2 prefill
  fallback after kernel attribution showed PV accumulation `246.7-247.0 ms`,
  K/V tile load `177.7-177.9 ms`, and softmax/rescale/probability
  `156.8-157.1 ms` per capped `14780`-token HD512 layer. The design gate was
  recorded in `krasis-internal/DEBUGLOG.md` before inspection. Source review
  found the active q2 path uses `BC=16`, `q_heads_per_block=2`, and `70656`
  bytes of dynamic shared memory; existing single-head BC32/48/64 variants
  require `86016/120320/154624` bytes. No performance candidate was selected:
  forcing BC32 is mechanically narrow but not clear-upside because it gives up
  q2 K/V sharing, while the real PV/softmax improvement path requires a broader
  HD512 attention retile/redesign. No performance code, build, or benchmark was
  run for this design-only pass.

- Added timing-only Gemma4 HQQ4/k4v4 HD512 `custom_tiled` q2 kernel
  attribution under accepted decode baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`.
  The metric gate was recorded before source changes. The debug path is opt-in
  with `KRASIS_PREFILL_HD512_KERNEL_CLOCKS=1` and splits the HD512 q2 fallback
  into Q load, K/V tile load, QK score, softmax/rescale/probability write, PV
  accumulation, final write, and residual overhead without changing default
  prefill kernel selection, calibration, HCS, graph/decode behavior, chunk
  sizing, or non-Gemma behavior. Initial build
  `20260614_1219_gemma4_hqq4_k4v4_prefill_hd512_kernel_attr_build.log` failed
  with Rust `E0308` from passing an `Option<f64>` event time into the report
  helper; fixed build
  `20260614_1221_gemma4_hqq4_k4v4_prefill_hd512_kernel_attr_fix_build.log`
  passed with `duration_s=129`. Attribution run
  `20260614_1224_gemma4_hqq4_k4v4_prefill_hd512_kernel_attr_timing.log`
  passed `14/14`, `ALL TESTS PASSED`, with `3865.8` prefill, `92.17`
  internal decode, `157.09` HTTP, HCS `3840/3840`, min free `11474 MB`, zero
  cold DMA, and `copy_failures=0`. The prefill number is instrumentation-heavy
  and not a speed-regression marker. Representative `14780`-token q2 rows
  showed each HD512 layer at `627.0-627.9 ms`, split into Q load `1.2 ms`,
  K/V tile load `177.7-177.9 ms`, QK score `44.4-44.5 ms`,
  softmax/rescale/probability write `156.8-157.1 ms`, PV accumulation
  `246.7-247.0 ms`, final write `0.1 ms`, and residual `0.0 ms`. Decision:
  attribution only; no performance candidate was attempted because the split
  points at a larger HD512 attention redesign, not a distinct narrow safe
  implementation path.

- Ran source-free Gemma4 HQQ4/k4v4 prefill attribution under accepted decode
  baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`
  with `KRASIS_PREFILL_TIMING=1`. Command:
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf`.
  The run rebuilt and loaded the Gemma Marlin expert cache during startup, then
  passed `14/14`, `ALL TESTS PASSED`, with `5562.6` prefill, `92.53`
  internal decode, `160.33` HTTP, HCS `3840/3840`, min free `11474 MB`, zero
  cold DMA, and `copy_failures=0`. The low prior timing-off prefill marker
  `4931.0` was not reproduced. Benchmark prefill rows were `2860.1` at
  `1000` tokens, `5562.6` at `4999`, `4660.3` at `10000`, and
  `3796.9/3794.6/3767.9` on capped `14780/14780/14779` rows. Long-row
  attribution showed the known HD512 custom-tiled fallback launch at
  `2503.4-2505.0 ms` over 5 calls, while KV append was only `3.4-3.5 ms`
  (`~0.3 ms` wall-minus-kernel). Decision: attribution only; no new safe
  non-rejected prefill optimization candidate was selected.

- Added timing-only Gemma4 HQQ4/k4v4 MoE hidden write/post-FFN attribution
  under accepted experimental baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`.
  The new debug split widens the Gemma MoE-route clock layout to separate
  `post_ffn_norm2`, dense+MoE add, `post_ffn_norm`, residual add, optional
  layer scalar, and endpoint overhead. Initial build
  `20260614_1140_gemma4_hqq4_k4v4_moe_hidden_attr_build.log` passed, but the
  first timing run
  `20260614_1144_gemma4_hqq4_k4v4_moe_hidden_attr_timing.log` was stopped as
  invalid because the new sub-span accumulators were not reset per request,
  producing negative residuals. Fixed build
  `20260614_1150_gemma4_hqq4_k4v4_moe_hidden_attr_fix_build.log` passed with
  `duration_s=128`. Valid attribution
  `20260614_1154_gemma4_hqq4_k4v4_moe_hidden_attr_timing.log` passed `14/14`,
  `ALL TESTS PASSED`, with `5507.8` prefill, `79.59` internal decode,
  `175.39` HTTP, HCS `3840/3840`, min free `11468 MB`, zero cold DMA, and
  `copy_failures=0`. Representative rows showed hidden write/post-FFN
  `0.58-0.68 ms/tok`, `post_ffn_norm2` `0.21-0.23`, dense+MoE add
  `0.05-0.07`, `post_ffn_norm` `0.21-0.23`, residual add `0.05-0.07`,
  layer scalar `0.05-0.07`, endpoint `0.02-0.03`, and residual `~0.00`.
  Decision: attribution only. The RMSNorms are real `>0.2 ms/tok` pieces, but
  no distinct safe non-rejected implementation path was identified from this
  split.

- Added timing-only Gemma4 HQQ4/k4v4 MoE post-output attribution under
  accepted experimental baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`.
  The new debug split widens the Gemma MoE-route clock layout to separate
  shared gate, shared W13/reduce, shared W2, routed scaling, hidden
  write/post-FFN work, and endpoint overhead. Build
  `20260614_1125_gemma4_hqq4_k4v4_moe_post_attr_build.log` passed with
  `duration_s=127`. Focused attribution
  `20260614_1130_gemma4_hqq4_k4v4_moe_post_attr_timing.log` passed `14/14`,
  `ALL TESTS PASSED`, with `5293.8` prefill, `80.69` internal decode,
  `186.31` HTTP, HCS `3840/3840`, min free `11470 MB`, zero cold DMA, and
  `copy_failures=0`. Representative rows showed post-output `0.55-0.63
  ms/tok`, shared gate/W13/reduce/W2 all `0.00`, routed scale `0.02-0.03`,
  hidden write/post-FFN `0.49-0.54`, endpoint `0.04-0.06`, and routed
  weighted combine `0.06-0.08`. Decision: attribution only. Hidden
  write/post-FFN is the only newly isolated `>0.2 ms/tok` piece, but it still
  groups multiple kernels and needs a further split before there is a distinct
  safe implementation path.

- Added an attribution-only Gemma4 HQQ4/k4v4 focused MoE timing pass under
  accepted experimental baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`.
  No source change was made; the run used existing MoE-route and W2 preload
  clocks. Run `20260614_1116_gemma4_hqq4_k4v4_moe_focus_attr_timing.log`
  passed `14/14`, `ALL TESTS PASSED`, with `5005.2` prefill, `81.13`
  internal decode, `187.11` HTTP, HCS `3840/3840`, min free `11470 MB`, zero
  cold DMA, and `copy_failures=0`. Representative rows showed MoE expert
  `3.36-3.53 ms/tok`, W13 `1.08-1.12`, W13 reduce `0.05-0.07`,
  activation+W2 `1.66-1.69`, activation `1.19`, W2 compute/output `0.41`,
  load `0.31-0.35`, SiLU/multiply `0.62-0.66`, shared store `0.21-0.23`,
  weighted accumulation `0.06-0.08`, graph/launch `0.04-0.06`, and
  post-output `0.50-0.57`. Decision: attribution only. W13 direct,
  activation precompute/reuse, paired-output W2, and FP32 shared activation
  are already rejected; weighted accumulation/launch are too small; post-output
  needs a separate split before it is a safe candidate.

- Added an attribution-only Gemma4 HQQ4/k4v4 route/final focused timing pass
  under accepted experimental baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`.
  No source change was made; the run used existing final, MoE-route, and
  route-prep clocks to inspect route-prep dense/overhead and final LM-head
  BF16 cuBLAS after moving off non-active GQA. Run
  `20260614_1108_gemma4_hqq4_k4v4_route_final_attr_timing.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5618.9` prefill, `83.63` internal
  decode, `192.76` HTTP, HCS `3840/3840`, min free `11466 MB`, zero cold DMA,
  and `copy_failures=0`. Representative rows showed route-prep dense pre
  `0.25-0.26`, gate `0.34-0.35`, up `0.34-0.35`, activation `0.06-0.07`,
  down `0.37-0.38`, dense post norms `0.42-0.43`, router norm `0.02-0.03`,
  remaining overhead `0.22-0.25`, final LM-head BF16 cuBLAS `0.92`, D2H
  logits `0.13`, graph residual `0.12`, and final segment/sync `1.05 ms/tok`.
  Decision: attribution only. Route-prep does not expose a clean non-rejected
  `>0.2 ms/tok` candidate, and final LM-head optimization remains blocked by
  the previously rejected Marlin INT8 path.

- Added timing-only focused attribution for Gemma4 HQQ4/k4v4 non-active GQA
  under accepted experimental baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`.
  The new debug split expands existing non-active GQA markers into projection,
  Q/K/V norm, RoPE, KV-cache write, attention/reduce, BF16/gated conversion,
  O-input prep, O projection, endpoint/gap, and the existing HD256
  score/weight-V/final-output clocks. Build
  `20260614_1052_gemma4_hqq4_k4v4_gqa_other_detail_attr_build.log` passed
  with `duration_s=128`. Focused attribution
  `20260614_1055_gemma4_hqq4_k4v4_gqa_other_detail_attr_timing.log` passed
  `14/14`, `ALL TESTS PASSED`, with `4950.5` prefill, `77.07` internal
  decode, `156.54` HTTP, HCS `3840/3840`, min free `11462 MB`, zero cold
  DMA, and `copy_failures=0`. Representative 49/99/249/511 rows showed
  projection `0.92/0.95/0.95/0.92 ms/tok`, Q/K/V norm
  `0.10/0.12/0.12/0.10`, RoPE `0.04/0.06/0.05/0.04`, KV write
  `0.11/0.13/0.12/0.11`, attention `0.68/0.85/1.30/1.88`, O projection
  `0.44/0.46/0.45/0.44`, and HD256 weight/V `0.42/0.56/0.89/1.33`.
  Decision: attribution only. Newly split norm/RoPE/KV pieces are below the
  `>0.2 ms/tok` useful-upside threshold; larger projection/attention areas map
  to already rejected or not-yet-safe GQA work, so no optimization candidate
  was attempted.

- Rejected the decode-only Gemma4 HQQ4/k4v4 route-prep fused pre-norm copy
  candidate. Scope used accepted experimental baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`
  and targeted the remaining non-rejected route-prep dense pre-norm bucket
  from fresh post-softcap attribution
  `20260614_1018_gemma4_hqq4_k4v4_post_softcap_fresh_attr_timing.log`
  (`0.23-0.26 ms/tok`, fresh summary `5597.4` prefill, `77.58` internal
  decode, `153.06` HTTP, HCS `3840/3840`, min free `11464 MB`, zero cold DMA,
  `copy_failures=0`). Candidate env
  `KRASIS_DECODE_ROUTE_PREP_FUSED_PRE_NORM=1` replaced only the Gemma graph
  route-prep residual copy plus pre-FFN RMSNorm pair with existing
  `fused_add_rmsnorm(..., first_layer=1)`, gated by both accepted env flags,
  Gemma k4v4 graph scope, nonzero `pre_ffn_norm_ptr`, and runtime
  hidden-size/shared-memory capacity. Build
  `20260614_1037_gemma4_hqq4_k4v4_route_fused_pre_norm_build.log` passed with
  `duration_s=129`. Timing gate
  `20260614_1040_gemma4_hqq4_k4v4_route_fused_pre_norm_timing.log` passed
  correctness (`14/14`, `ALL TESTS PASSED`) with `5611.7` prefill, `77.80`
  internal decode, `157.72` HTTP, HCS `3840/3840`, min free `11462 MB`, and
  `copy_failures=0`, but failed the performance gate: dense pre-norm only
  moved to mostly `0.23 ms/tok` with occasional `0.21-0.22`, route
  prep/overhead stayed mostly `2.60 ms/tok` and regressed on some rows, and
  the `77.58 -> 77.80 tok/s` timing delta was too small/noisy to accept.
  Reverted only this candidate; restore build
  `20260614_1048_restore_after_route_fused_pre_norm_reject_build.log` passed
  with `duration_s=127`, and candidate env/source symbols are absent. No
  timing-off Gemma speed or QCN guard was run because the timing gate failed.

- Rejected the decode-only Gemma4 HQQ4/k4v4 final LM-head Marlin INT8
  candidate. Scope used accepted experimental baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`
  and first ran fresh post-softcap attribution
  `20260614_1018_gemma4_hqq4_k4v4_post_softcap_fresh_attr_timing.log`, which
  passed `14/14` with `5597.4` prefill, `77.58` internal decode, `153.06`
  HTTP, HCS `3840/3840`, min free `11464 MB`, zero cold DMA, and
  `copy_failures=0`. The remaining final LM-head bucket was about
  `0.93 ms/tok`, D2H logits were `0.15-0.16`, host softcap was `0.00`, and
  weighted expert accumulation remained below threshold. The candidate was
  env-gated by `KRASIS_DECODE_FINAL_LM_HEAD_MARLIN_INT8=1`, Gemma-final only,
  and dimension-gated from the source INT8 LM-head tuple plus Marlin
  compatibility checks (`rows % 64 == 0`, `cols % 16 == 0`,
  `cols % group_size == 0`). Build
  `20260614_1029_gemma4_hqq4_k4v4_final_lm_head_marlin_int8_build.log`
  passed with `duration_s=7`, but timing gate
  `20260614_1030_gemma4_hqq4_k4v4_final_lm_head_marlin_int8_timing.log`
  failed before benchmark completion: LM head regressed to `9.40-9.52
  ms/tok`, then graph replay failed with `CUDA_ERROR_ILLEGAL_ADDRESS` at final
  sync and the boundary sync reported a fatal CUDA context error. HCS was
  still clean before failure (`3840/3840`, representative `240/0`,
  `copy_failures=0`). Reverted only this candidate; restore build
  `20260614_1033_restore_after_final_lm_head_marlin_int8_reject_build.log`
  passed with `duration_s=8`, and candidate env/source symbols are absent. No
  timing-off Gemma speed or QCN guard was run because the timing gate failed.

- Accepted an env-gated Gemma4 HQQ4/k4v4 GPU final-logit softcap path. A
  focused final/sync attribution pass was run first under accepted experimental
  baseline `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1`. Initial attribution build
  `20260614_0948_gemma4_hqq4_k4v4_final_attr_build.log` failed with
  `error[E0425]` for an out-of-scope `avg_sync_final` report variable; build2
  `20260614_0952_gemma4_hqq4_k4v4_final_attr_build2.log` passed with
  `duration_s=127`. Focused attribution
  `20260614_0955_gemma4_hqq4_k4v4_final_attr_timing.log` passed correctness
  (`14/14`, `ALL TESTS PASSED`) with `5626.1` prefill, `58.90` internal
  decode, `119.52` HTTP, HCS `3840/3840`, min free `11464 MB`, zero cold DMA,
  and `copy_failures=0`; it split final graph work into final RMSNorm
  `~0.01 ms/tok`, LM head `~0.92`, graph residual `~0.12`, D2H logits
  `0.12-0.13`, and host final-logit softcap typically `3.86-4.32 ms/tok`
  with one `4.84` row. Added `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`, gated to
  Gemma graph final segments with finite positive `final_logit_softcap` and
  runtime `vocab_size`, applying the same `tanh(logit / softcap) * softcap`
  formula to `d_logits` before D2H and skipping CPU softcap only when the GPU
  path captured. Candidate build
  `20260614_0952_gemma4_hqq4_k4v4_final_softcap_gpu_build.log` passed with
  `duration_s=136`. Timing gate
  `20260614_0958_gemma4_hqq4_k4v4_final_softcap_gpu_timing.log` passed
  `14/14`, moved host softcap to `0.00 ms/tok`, and improved timing-enabled
  internal decode from `58.90` to `77.84 tok/s` with HCS still clean. Timing-
  off Gemma run `20260614_1004_gemma4_hqq4_k4v4_final_softcap_gpu_speed.log`
  passed `14/14` with `4931.0` prefill, `92.31` internal decode, `160.16`
  HTTP, HCS `3840/3840`, min free `11474 MB`, and `copy_failures=0`; prefill
  was lower in that run despite no prefill-path changes and is recorded as a
  follow-up data point. QCN guard
  `20260614_1011_qcn_speed_test_after_final_softcap_gpu.log` passed via
  `./dev speed-test` with `6685.1` prefill, `90.07` internal decode,
  `198.82` HTTP, HCS `15957/24576`, min free `896 MB`, and `copy_failures=0`.

- Rejected the decode-only Gemma4 HQQ4/k4v4 MoE W2 FP32 shared-activation
  candidate. Scope stayed on the accepted experimental baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and targeted the fresh measured MoE
  activation/W2 bucket: fused activation+W2 `1.67-1.69 ms/tok`, with preload
  detail load `~0.34`, SiLU/multiply `~0.64`, and shared store
  `~0.21 ms/tok`; weighted expert accumulation stayed below threshold at
  `0.06-0.08 ms/tok`. The candidate was env-gated by
  `KRASIS_DECODE_MOE_W2_F32_ACT=1`, kept one output tile per block, did not
  retry paired-output W2, and runtime-gated shared-memory use from
  `topk/intermediate/expert_hs` dimensions and device capacity. Initial build
  `20260614_0909_gemma4_hqq4_k4v4_moe_w2_f32_act_build.log` passed with
  `duration_s=137`, but the first timing attempt failed before serving because
  the new PTX symbols were not added to `KERNEL_NAMES`. After fixing that
  integration issue, build2
  `20260614_0917_gemma4_hqq4_k4v4_moe_w2_f32_act_build2.log` passed with
  `duration_s=129`. Valid timing gate
  `20260614_0920_gemma4_hqq4_k4v4_moe_w2_f32_act_timing2.log` passed
  correctness (`14/14`, `ALL TESTS PASSED`) but failed the full metric gate:
  `4683.2` prefill, `58.41` internal decode, and `99.72` HTTP versus fresh
  broad-clock baseline `4731.8/59.29/119.22`. HCS stayed clean at `3840/3840`,
  min free `11464 MB`, representative `240/0`, zero cold DMA, and
  `copy_failures=0`. The target W2 bucket improved locally
  (`1.54/1.54/1.54/1.53 ms/tok`, internal `1.47-1.48`), but total internal
  decode regressed, so the candidate was reverted. Restore build
  `20260614_0927_restore_after_moe_w2_f32_act_reject_build.log` passed with
  `duration_s=135`; no timing-off Gemma speed or QCN guard was run.

- Rejected the decode-only Gemma4 HQQ4/k4v4 GQA fused-QKV split-elision
  candidate. First ran a fresh broad timing attribution with accepted
  experimental baseline `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1`:
  `20260614_0842_gemma4_hqq4_k4v4_dual_norm_fresh_attr_timing.log` passed
  correctness (`14/14`, `ALL TESTS PASSED`) with `4731.8` prefill, `59.29`
  internal decode, `119.22` HTTP, HCS `3840/3840`, min free `11464 MB`, zero
  cold DMA, and `copy_failures=0`. Weighted expert accumulation remained only
  `0.06-0.08 ms/tok`, so no weighted-add candidate was attempted under the
  requested `>0.2 ms/tok` upside rule. Excluding paired W2, W13 direct, dense
  gate+up fusion, dense-down custom GEMV, and rejected GQA tile-cap,
  score-cache, and HD256-specialization variants, the chosen candidate targeted
  non-active GQA projection by env-gating fused-QKV K/V split-copy elision under
  `KRASIS_DECODE_GQA_FUSED_QKV_SPLIT_ELIDE=1`. Candidate build
  `20260614_0852_gemma4_hqq4_k4v4_gqa_fused_qkv_elide_build.log` passed with
  `duration_s=127`. Timing gate
  `20260614_0855_gemma4_hqq4_k4v4_gqa_fused_qkv_elide_timing.log` passed
  correctness (`14/14`, `ALL TESTS PASSED`) but failed the metric gate:
  `5605.5` prefill, `58.56` internal decode, and `110.82` HTTP versus fresh
  broad-clock baseline `4731.8/59.29/119.22`. HCS stayed clean at `3840/3840`,
  min free `11464 MB`, representative `240/0`, zero cold DMA, and
  `copy_failures=0`. The target non-active GQA projection did not improve
  (`0.95/0.95/0.95/0.93 ms/tok` versus baseline
  `0.95/0.93/0.95/0.95`), non-active endpoint regressed
  (`2.44/2.59/3.04/3.56` versus `2.44/2.53/2.96/3.36`), and total internal
  decode regressed. Reverted only this candidate; restore build
  `20260614_0902_restore_after_gqa_fused_qkv_elide_reject_build.log` passed
  with `duration_s=127`, and the candidate env/symbols are absent. No
  timing-off Gemma speed or QCN guard was run.

- Rejected the decode-only Gemma4 HQQ4/k4v4 MoE W13 direct-BF16 candidate.
  Scope used the accepted experimental baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and targeted the measured W13 bucket
  in Gemma graph MoE. The candidate was env-gated by
  `KRASIS_DECODE_MOE_W13_DIRECT=1`, runtime-gated to Gemma4 k4v4 graph MoE
  with INT4 gated experts and `w13_ksplits_batched == 1`, and attempted to
  write W13 output directly as BF16 instead of launching the one-split
  `reduce_ksplits_bf16_batched` pass. It did not change calibration, HCS
  policy, graph capture structure, prefill, final segment behavior, or
  non-Gemma behavior, and avoided paired-W2, dense gate+up fusion, dense-down
  custom GEMV, GQA tile-cap, score-cache, and HD256 specialization. Candidate
  build `20260614_0827_gemma4_hqq4_k4v4_moe_w13_direct_build.log` passed with
  `duration_s=136`. Timing gate
  `20260614_0830_gemma4_hqq4_k4v4_moe_w13_direct_timing.log` used
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_MOE_W13_DIRECT=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed correctness (`14/14`, `ALL TESTS PASSED`) but failed the metric
  gate: `5626.1` prefill, `62.16` internal decode, and `112.01` HTTP versus
  accepted dual-norm timing baseline `5628.2/64.75/112.01`. HCS stayed clean
  at `3840/3840`, min free `11466 MB`, representative `240/0`, zero cold DMA
  and `copy_failures=0`. W13+reduce stayed essentially unchanged at about
  `1.15-1.18 ms/tok`, and total internal decode regressed, so the candidate
  was reverted. Restore build
  `20260614_0836_restore_after_moe_w13_direct_reject_build.log` passed with
  `duration_s=136`; no timing-off Gemma speed or QCN guard was run.

- Rejected the decode-only Gemma4 HQQ4/k4v4 route-prep dense-down custom BF16
  GEMV candidate. The candidate was env-gated by
  `KRASIS_DECODE_ROUTE_PREP_DENSE_DOWN_CUSTOM=1` on top of the accepted
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` baseline and was runtime-gated to the
  Gemma graph route-prep dense-down BF16 weight dimensions. It avoided paired
  W2, dense gate+up fusion, GQA tile-cap, score-cache, and HD256 specialization
  candidates, and did not change calibration, HCS policy, graph capture,
  prefill, final segment behavior, or non-Gemma behavior. Candidate build
  `20260614_0806_gemma4_hqq4_k4v4_route_dense_down_custom_build.log` passed
  with `duration_s=137`. Timing gate
  `20260614_0812_gemma4_hqq4_k4v4_route_dense_down_custom_timing.log` used
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_ROUTE_PREP_DENSE_DOWN_CUSTOM=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed correctness (`14/14`, `ALL TESTS PASSED`) but failed the full
  metric gate: `5622.1` prefill, `62.12` internal decode, and `112.43` HTTP
  versus accepted dual-norm timing baseline `5628.2/64.75/112.01`. HCS stayed
  clean at `3840/3840`, min free `11466 MB`, representative `240/0`, zero cold
  DMA and `copy_failures=0`. Dense-down improved from about
  `0.36-0.37 ms/tok` to `0.29-0.31 ms/tok`, but route prep did not improve
  enough and total internal decode regressed, so the candidate was reverted.
  Restore build `20260614_0830_restore_after_dense_down_custom_reject_build.log`
  passed with `duration_s=136`; no timing-off Gemma speed or QCN guard was run.

- Rejected the decode-only Gemma4 HQQ4/k4v4 HD256 GQA specialized
  single-kernel candidate. The candidate was env-gated by
  `KRASIS_DECODE_GQA_HD256_SPECIALIZED=1` on top of the accepted
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` baseline, and was runtime-gated to
  Gemma graph route GQA with k4v4, `head_dim == 256`, sliding/gated GQA, and
  active shared-memory capacity. It avoided the rejected tile-cap and
  score-cache GQA variants and did not change calibration, HCS policy, graph
  capture, prefill, final segment behavior, or non-Gemma paths. Candidate
  build `20260614_0748_gemma4_hqq4_k4v4_hd256_spec_build.log` passed with
  `duration_s=138`. Timing gate
  `20260614_0752_gemma4_hqq4_k4v4_hd256_spec_timing.log` used
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_GQA_HD256_SPECIALIZED=1 KRASIS_DECODE_GQA_HD256_ATTN_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed correctness (`14/14`, `ALL TESTS PASSED`) but failed performance:
  `5616.7` prefill, `61.87` internal decode, and `110.76` HTTP versus the
  accepted dual-norm timing baseline `5628.2/64.75/112.01`. HCS stayed clean
  at `3840/3840`, min free `11464 MB`, representative `240/0`, zero cold DMA
  and `copy_failures=0`. The measured GQA rows showed no consistent
  improvement (`3.22/3.33/3.97 ms/tok` versus baseline `2.97/3.35/3.89`), and
  total decode regressed. Reverted only this candidate; restore build
  `20260614_0807_restore_after_hd256_spec_reject_build.log` passed with
  `duration_s=136`. No timing-off speed or QCN guard was run.

- Added an env-gated decode-only Gemma4 HQQ4/k4v4 route-prep dual residual
  RMSNorm path under `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1`. Inspection showed
  two adjacent residual RMSNorm passes in the Gemma route-prep graph path:
  `pre_ffn_norm2` writes the BF16 scratch path and router-input RMSNorm+scale
  writes `d_hidden`, both reading the same `d_residual` with the same RMS
  denominator. The candidate fuses only those two norms and avoids the
  previously failed dense gate+up fusion path and all W2 paired-tile work. It
  does not change calibration, HCS policy, graph capture, prefill, final
  segment behavior, or QCN/non-Gemma paths. Build
  `20260614_0720_gemma4_hqq4_k4v4_route_dual_norm_build.log` passed with
  `duration_s=135`. Timing gate
  `20260614_0724_gemma4_hqq4_k4v4_route_dual_norm_timing.log` used
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed `14/14`, `ALL TESTS PASSED`, with `5628.2` prefill, `64.75`
  internal decode, `112.01` HTTP, HCS `3840/3840`, min free `11466 MB`, graph
  HCS `240/0`, and zero cold DMA/DMA calls or copy failures. The target
  duplicated residual-norm bucket improved from baseline dense-post+router
  `0.65/0.65/0.65/0.60 ms/tok` to `0.46/0.43/0.46/0.46`; route prep was
  `2.63/2.41/2.63/2.63` versus baseline `2.82/2.82/2.82/2.53`. Timing-off
  Gemma validation
  `20260614_0737_gemma4_hqq4_k4v4_route_dual_norm_speed.log` passed `14/14`
  with `5626.1` prefill, `66.96` internal decode, `120.00` round trip, HCS
  `3840/3840`, min free `11474 MB`, and `copy_failures=0`. QCN guard
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 ./dev speed-test` completed in
  `20260614_0733_qcn_speed_test_after_route_dual_norm.log` with `6452.1`
  prefill, `87.61` internal decode, `149.48` round trip, HCS `15957/24576`,
  min free `928 MB`, and `copy_failures=0`, confirming the Gemma-only gate did
  not route QCN through the new path.

- Rejected the decode-only Gemma4 HQQ4/k4v4 INT4 fused-W2 paired-output-tile
  candidate. Candidate build
  `20260614_0654_gemma4_hqq4_k4v4_w2_pair2_build.log` passed with
  `duration_s=137`. Timing gate
  `20260614_0700_gemma4_hqq4_k4v4_w2_pair2_timing.log` used
  `KRASIS_DECODE_MOE_W2_PAIRED_TILES=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and failed correctness: `13/14`, `multi_turn_t5_all` missing `Biscuit`,
  `1 TEST(S) FAILED`. Benchmark summary before failure was `5454.8 tok/s`
  prefill, `59.12 tok/s` internal decode, and `97.46 tok/s` round trip. HCS
  stayed clean with `3840/3840`, representative `240/0`, zero cold DMA/DMA
  calls, and `copy_failures=0`. The paired path was active
  (`paired_tiles:on`, `physical_tiles/seg:88.0`, `blocks/seg:704.0`) but
  regressed target timing: fused activation+W2 became
  `2.01/2.01/2.01/1.99 ms/tok` versus accepted attribution baseline
  `1.69/1.69/1.69/1.67`, internal W2 total became about `1.94-1.95` versus
  `1.63`, and SiLU/multiply rose from `0.65` to about `1.18`. Reverted only
  this candidate; restore build
  `20260614_0710_restore_after_w2_pair2_reject_build.log` passed with
  `duration_s=135`, and no paired-tile symbols or env gate remain. No
  timing-off `./dev speed-test` or QCN guard was run.

- Added diagnostic-only Gemma4 HQQ4/k4v4 fused W2 activation/preload
  sub-attribution under `KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1`, layered on the
  existing `KRASIS_DECODE_MOE_W2_CLOCKS=1` and MoE route graph timing gate.
  Two initial timing attempts were discarded: `20260613_231855...timing.log`
  used per-iteration clocks that perturbed activation wall time, and
  `20260613_232606...timing2.log` exposed a four-slot/eight-slot clock-base
  stride bug. After fixing the stride, build
  `20260613_233023_gemma4_hqq4_k4v4_moe_w2_preload_attr_build3.log` passed
  with `duration_s=127`, and valid timing run
  `20260613_233253_gemma4_hqq4_k4v4_moe_w2_preload_attr_timing3.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5602.3 tok/s` prefill, `59.89 tok/s`
  internal decode, `108.08 tok/s` HTTP, HCS `3840/3840`, min free
  `11470 MB`, graph HCS `240/0`, and zero cold DMA/DMA calls or copy failures.
  Representative 49/99/249/511 rows showed fused activation+W2
  `1.69/1.69/1.69/1.67 ms/tok`, activation `1.20` ms/tok, W2 prep `0.02`,
  W2 GEMV/output `0.41`, and activation detail load `0.33`, SiLU/multiply
  `0.65`, shared store `0.21`, sync `0.04`, residual `0.00`, repeated block
  work about `53.9 ms/tok`, and `1408.0` blocks/segment. Exactly one next
  candidate is selected: a decode-only Gemma INT4 fused-W2 paired-output-tile
  block variant for graph MoE, avoiding the rejected global activation
  precompute/reuse path.

- Rejected the decode-only Gemma4 HQQ4/k4v4 INT4 expert activation
  precompute/reuse candidate. Inspection confirmed the premise from the actual
  `fused_silu_w2_batched` kernel: launch shape is
  `(ceil(expert_hs / 16), 1, topk)`, and every `(output tile, expert)` block
  recomputes `SiLU(gate) * up` for the full `moe_intermediate_size` into
  block-local shared memory. Candidate build
  `20260613_225526_gemma4_hqq4_k4v4_act_precompute_build.log` passed with
  `duration_s=137`. Timing gate
  `20260613_225804_gemma4_hqq4_k4v4_act_precompute_timing.log` used
  `KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed `14/14`, `ALL TESTS PASSED`, with `5165.6 tok/s` prefill,
  `64.24 tok/s` internal decode, `109.83 tok/s` HTTP, HCS `3840/3840`, min
  free `11470 MB`, graph HCS `240/0`, and zero cold DMA/DMA calls or copy
  failures. The target bucket did not improve: activation dropped to
  `0.01 ms/tok`, but W2 prep rose to `1.07-1.08 ms/tok`, leaving
  activation+W2 at `1.48/1.51/1.49/1.50 ms/tok` versus the accepted marker
  `1.48/1.48/1.48/1.46`. Reverted only this candidate; restore build
  `20260613_230519_restore_after_act_precompute_reject_build.log` passed with
  `duration_s=135`, and no activation-precompute symbols remain. No timing-off
  Gemma speed or QCN guard was run.

- Added diagnostic-only Gemma4 HQQ4/k4v4 MoE activation/W2 attribution for the
  next proven decode bucket after rejecting dense gate+up. The instrumentation
  is gated by `KRASIS_DECODE_MOE_W2_CLOCKS=1`, graph timing, k4v4 captured
  graphs containing `Gemma4MoE`, target `GRAPH_SEG_ROUTE_GQA`, and the runtime
  gated INT4 expert path; normal decode stays on `fused_silu_w2_batched`.
  Build attempt `20260613_223530_gemma4_hqq4_k4v4_moe_w2_attr_build.log`
  failed with Rust `E0277` because the diagnostic timed kernel exceeded
  cudarc's tuple launch arity; the launch was corrected to use the raw
  pointer-vector form. Build
  `20260613_223732_gemma4_hqq4_k4v4_moe_w2_attr_build2.log` passed with
  `duration_s=128`. Timing run
  `20260613_224001_gemma4_hqq4_k4v4_moe_w2_attr_timing.log` used
  `KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed `14/14`, `ALL TESTS PASSED`, with `4915.2 tok/s` prefill,
  `62.16 tok/s` internal decode, `110.99 tok/s` HTTP, HCS `3840/3840`, min
  free `11470 MB`, graph HCS `240/0`, zero cold DMA/DMA calls, and dynamic-HCS
  `promotions=0 evictions=0 copy_failures=0`. Representative 49/99/249/511
  rows showed activation `1.05/1.06/1.06/1.05 ms/tok`, W2 prep
  `0.02/0.02/0.02/0.02`, W2 GEMV/output `0.34/0.34/0.34/0.34`, weighted
  accumulation `0.08/0.08/0.08/0.06`, and graph/launch overhead
  `0.06/0.06/0.06/0.05`. Exactly one next candidate is selected:
  decode-only Gemma INT4 expert activation precompute/reuse for graph MoE.

- Rejected the decode-only Gemma4 HQQ4/k4v4 route-prep dense gate+up
  dual-projection candidate. Inspection confirmed the current path uses BF16
  dense MLP tensors registered with `dtype=0`, two separate cuBLAS BF16 GEMM
  launches sharing the same `d_hidden` input, and adjacent `[gate | up]` output
  slices in `d_expert_gate_up`. Candidate build attempt
  `20260613_221418_gemma4_hqq4_k4v4_dense_gate_up_build.log` failed with Rust
  `E0133`; the wrapper was fixed and build
  `20260613_221555_gemma4_hqq4_k4v4_dense_gate_up_build2.log` passed with
  `duration_s=127`. Timing gate
  `20260613_221837_gemma4_hqq4_k4v4_dense_gate_up_timing.log` used
  `KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  but failed correctness during warmup graph replay with
  `sync routing[1]: CUDA_ERROR_ILLEGAL_ADDRESS`, followed by fatal
  decode/prefill boundary sync. Partial diagnostics showed HCS `3840/3840`,
  `240/0` hit/miss, and zero cold DMA/DMA calls, but no full benchmark summary
  was valid. Reverted only this candidate; restore build
  `20260613_222310_restore_after_dense_gate_up_reject_build.log` passed with
  `duration_s=127`, and no grouped dense gate+up symbols remain. No timing-off
  Gemma speed or QCN guard was run.

- Added diagnostic-only Gemma4 HQQ4/k4v4 route-prep sub-split attribution for
  the largest unresolved non-GQA decode bucket. The instrumentation is gated by
  `KRASIS_DECODE_ROUTE_PREP_CLOCKS=1`, graph timing, k4v4 captured graphs with
  `Gemma4MoE`, and target `GRAPH_SEG_ROUTE_GQA` layers; it also forces the
  parent MoE/route clocks so the denominator is recorded in the same run. Build
  `20260613_215750_gemma4_hqq4_k4v4_route_prep_attr_build.log` passed in
  `127s`. Timing run
  `20260613_220014_gemma4_hqq4_k4v4_route_prep_attr_timing.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5611.2 tok/s` prefill, `63.87 tok/s`
  internal decode, `109.14 tok/s` HTTP, HCS `3840/3840`, min free
  `11464 MB`, graph HCS `240/0`, zero cold DMA/DMA calls, and dynamic-HCS
  `promotions=0 evictions=0 copy_failures=0`. Representative 49/99/249/511
  rows showed route prep/overhead `2.82/2.82/2.82/2.53 ms/tok`, pre-GQA norm
  `0.26/0.26/0.26/0.23`, post-GQA norm/add `0.26/0.26/0.26/0.24`, dense
  pre-norm `0.26/0.26/0.26/0.23`, dense gate `0.35/0.35/0.35/0.33`, dense up
  `0.35/0.35/0.35/0.33`, dense activation `0.06/0.06/0.06/0.05`, dense down
  `0.38/0.38/0.37/0.36`, dense post norms `0.42/0.42/0.42/0.39`, router-input
  norm `0.23/0.23/0.23/0.21`, and remaining debug/graph overhead
  `0.25/0.25/0.25/0.17`. Exactly one next candidate is selected:
  decode-only Gemma dense gate+up dual-projection candidate for the route-prep
  graph path.

- Added diagnostic-only Gemma4 HQQ4/k4v4 MoE and route attribution for the
  next large decode buckets after rejecting HD256 GQA micro-optimizations. The
  instrumentation is timing/debug gated behind
  `KRASIS_DECODE_MOE_ROUTE_CLOCKS=1` and limited to k4v4
  `GRAPH_SEG_ROUTE_GQA` graph segments with `Gemma4MoE` layers. Build
  `20260613_214141_gemma4_hqq4_k4v4_moe_route_attr_build.log` passed in
  `125s`. Timing run
  `20260613_214409_gemma4_hqq4_k4v4_moe_route_attr_timing.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5598.9 tok/s` prefill,
  `62.57 tok/s` internal decode, `114.27 tok/s` HTTP, HCS `3840/3840`, min
  free `11470 MB`, graph HCS `240/0`, zero cold DMA/DMA calls, and dynamic-HCS
  `promotions=0 evictions=0 copy_failures=0`. Representative 49/99/249/511
  rows showed MoE expert `3.49/3.49/3.49/3.41 ms/tok`, GQA path
  `3.09/3.35/3.94/4.63`, route/top-k `3.09/3.09/3.09/2.96`, MoE W13
  `1.11/1.11/1.11/1.09`, activation+W2 `1.66/1.66/1.66/1.65`, route logits
  `0.15/0.15/0.15/0.14`, top-k `0.31/0.31/0.31/0.30`, scale/classify
  `0.13/0.13/0.13/0.12`, and route prep/overhead
  `2.49/2.49/2.49/2.39`. Exactly one next candidate is selected:
  decode-only Gemma route-prep sub-split attribution before optimization.

- Rejected the decode-only Gemma4 HQQ4/k4v4 sliding HD256 shared-score/cache
  candidate. The candidate was gated to CUDA graph GQA route decode only:
  `GRAPH_SEG_ROUTE_GQA`, `layer_idx == range_end`, k4v4, sliding
  `head_dim == 256`, gated GQA, and shared-score cache memory derived from
  `graph.kv_cache_len_for_layer(layer_idx)` plus active GPU shared-memory
  capacity. Candidate build
  `20260613_211248_gemma4_hqq4_k4v4_hd256_score_cache_build.log` passed in
  `135s`. Initial timing run
  `20260613_211550_gemma4_hqq4_k4v4_hd256_score_cache_timing.log` passed
  `14/14`; gate wiring was corrected so the candidate kernel opt-in updated
  the runtime capacity used by the gate. Corrected timing gate
  `20260613_212310_gemma4_hqq4_k4v4_hd256_score_cache_timing2.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5617.2 tok/s` prefill, `63.46 tok/s`
  internal decode, `114.75 tok/s` HTTP, HCS `3840/3840`, min free
  `11472 MB`, graph HCS `240/0`, zero cold DMA/DMA calls, and dynamic-HCS
  `promotions=0 evictions=0 copy_failures=0`. Representative 49/99/249/511
  rows showed sliding HD256 attention graph `0.70/0.87/1.28/1.89 ms/tok`,
  internal `0.64/0.81/1.24/1.84`, score `0.18/0.21/0.32/0.48`, weight+V
  `0.42/0.57/0.88/1.33`, and final `0.04/0.04/0.04/0.04`. Because
  weight+V did not improve versus the accepted marker, no timing-off Gemma
  speed or QCN guard was run. Reverted only the score-cache candidate and
  restore build `20260613_212911_restore_after_hd256_score_cache_reject_build.log`
  passed in `135s`; no score-cache symbols remain.

- Added diagnostic-only Gemma4 HQQ4/k4v4 sliding HD256 single-kernel attention
  attribution for the non-active attention/reduce bucket. The instrumentation
  is timing/debug gated behind `KRASIS_DECODE_GQA_HD256_ATTN_CLOCKS=1`, run
  with `KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1` and
  `KRASIS_DECODE_GQA_OTHER_CLOCKS=1`, and changes no prefill, HCS policy,
  final segment behavior, QCN/non-HD512 speed path, or output math. Corrected
  build `20260613_205442_gemma4_hqq4_k4v4_hd256_attn_attr_build5.log` passed
  in `126s`. Corrected timing run
  `20260613_205709_gemma4_hqq4_k4v4_hd256_attn_attr_timing4.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5605.5 tok/s` prefill, `63.80 tok/s`
  internal decode, `115.13 tok/s` HTTP, HCS `3840/3840`, min free
  `11472 MB`, clean graph HCS (`240/0` hit/miss, zero cold DMA/DMA calls), and
  dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. Representative
  49/99/249/511 rows showed sliding HD256 attention graph
  `0.70/0.87/1.28/1.89 ms/tok`, internal kernel
  `0.64/0.81/1.24/1.84`, graph overhead `0.06/0.06/0.04/0.05`, score/max
  `0.18/0.21/0.32/0.48`, weight+V accumulation `0.42/0.57/0.88/1.33`, and
  final reduce/output `0.04/0.04/0.04/0.04`. Runtime tiles averaged
  `1.00/1.00/1.22/1.63` against allocated graph max `59.00`. The actual
  accepted hot path is `gqa_attention_k4v4_single_g`; the rejected tile-cap
  branch targeted the live non-active HD256 route layers but replaced an
  already single-kernel path with tiled graph work. Exactly one next candidate
  is selected: decode-only HD256 k4v4 single-kernel shared-score/cache variant
  for sliding GQA, gated by runtime/config max attention length and active-GPU
  shared-memory capacity.

- Rejected the decode-only Gemma4 HQQ4/k4v4 sliding HD256 graph attention
  tile-cap candidate. The candidate was tightly gated to k4v4 graph decode,
  route-layer GQA, sliding `head_dim == 256`, allocated tiled buffers, and a
  per-layer `tile_cap > 1` derived from
  `ceil(kv_cache_len_for_layer(layer_idx) / graph.gqa_tile_size)` rather than
  a model hardcode. Build
  `20260613_201050_gemma4_hqq4_k4v4_hd256_tilecap_build.log` passed in
  `125s`. Timing gate
  `20260613_201352_gemma4_hqq4_k4v4_hd256_tilecap_timing.log` used
  `KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 KRASIS_DECODE_GQA_OTHER_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed `14/14`, `ALL TESTS PASSED`, with `5616.5 tok/s` prefill,
  `63.97 tok/s` internal decode, `114.51 tok/s` HTTP, HCS `3840/3840`, min
  free `11472 MB`, clean graph HCS (`240/0` hit/miss, zero cold DMA/DMA
  calls), and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. The
  target non-active attention/reduce bucket across representative 49/99/249/511
  rows was `0.76/0.96/1.46/2.16 ms/tok`, versus prior marker
  `0.75/0.95/1.47/2.16`, so it did not improve. Per gate, no timing-off
  Gemma speed or QCN guard was run. The candidate was reverted and restore
  build `20260613_201820_restore_after_hd256_tilecap_reject_build.log` passed
  in `124s`; no tile-cap candidate branch remains.

- Added diagnostic-only Gemma4 HQQ4/k4v4 non-active GQA geometry attribution
  for the remaining `other_mixed_gqa` bucket. The new report is timing/debug
  gated behind `KRASIS_DECODE_GQA_OTHER_CLOCKS=1`, run with
  `KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1`, and allocation is limited to k4v4
  graph sessions with tiled GQA buffers and at least one HD512 GQA layer. It
  changes no prefill, HCS policy, final segment behavior, QCN/non-HD512 path,
  output math, or speed-path behavior. Build
  `20260613_195304_gemma4_hqq4_k4v4_other_gqa_attr_build.log` passed in
  `127s`. Timing run
  `20260613_195613_gemma4_hqq4_k4v4_other_gqa_attr_timing.log` used
  `KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 KRASIS_DECODE_GQA_OTHER_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed `14/14`, `ALL TESTS PASSED`, with `5608.8 tok/s` prefill,
  `63.48 tok/s` internal decode, `114.54 tok/s` HTTP, HCS `3840/3840`, min
  free `11472 MB`, clean graph HCS (`240/0` hit/miss, zero cold DMA/DMA
  calls), and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`.
  Representative 49/99/249/511 rows showed mixed GQA total
  `3.23/3.48/4.10/4.80 ms/tok`, active HD512 span `0.70/0.74/0.85/0.89`,
  other mixed GQA `2.53/2.73/3.25/3.92`, and active endpoint/name coverage
  `100.0%`. The non-active split was projection `0.95/0.95/0.95/0.94`,
  norm/RoPE/KV `0.26/0.26/0.26/0.25`, attention/reduce
  `0.75/0.95/1.47/2.16`, BF16 `0.05/0.05/0.05/0.05`, O-input prep
  `0.02/0.02/0.02/0.02`, O projection `0.45/0.45/0.45/0.45`, endpoint
  `2.49/2.69/3.21/3.88`, and `24` non-active segments. The measured geometry
  is Gemma's sliding GQA path (`head_dim=256`, `num_attention_heads=16`,
  `num_key_value_heads=8`, `sliding_window=1024`, gated k4v4). Exactly one
  next optimization candidate is selected: decode-only sliding HD256 k4v4
  graph attention tile-cap specialization using per-layer max attention length
  to reduce inactive tile blocks in the row-growing attention/reduce bucket.

- Added diagnostic-only Gemma4 HQQ4/k4v4 mixed-segment coverage attribution for
  the graph GQA route path. The new report is timing/debug gated behind
  `KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1`, reuses existing mixed/path graph
  clock markers, and changes no prefill, HCS policy, final segment behavior,
  QCN/non-HD512 path, or output math. Build
  `20260613_180708_gemma4_hqq4_k4v4_coverage_attr_build.log` passed in
  `123s`. Timing run
  `20260613_180941_gemma4_hqq4_k4v4_coverage_attr_timing.log` used
  `KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed `14/14`, `ALL TESTS PASSED`, with `5602.5 tok/s` prefill,
  `64.32 tok/s` internal decode, `114.32 tok/s` HTTP, HCS `3840/3840`, min
  free `11474 MB`, clean graph HCS (`240/0` hit/miss, zero cold DMA/DMA
  calls), and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`.
  Representative 49/99/249/511 coverage rows showed mixed GQA total
  `3.08/3.24/3.96/4.41 ms/tok`, active HD512 mixed span
  `0.70/0.72/0.86/0.95`, other mixed GQA `2.38/2.52/3.11/3.46`, active
  endpoint `0.69/0.71/0.85/0.94`, active named submarkers
  `0.69/0.71/0.85/0.94`, active internal gap `0.00/0.00/0.00/0.00`, count
  coverage `17.2%`, active time coverage `22.8/22.3/21.6/21.5%`, and named
  endpoint coverage `100.0%`. The earlier path residual was a
  coverage/denominator mismatch: active HD512 markers fully cover the active
  path, but that path is only about one fifth of mixed GQA time. Exactly one
  next candidate is selected: decode-only non-active GQA layer geometry
  attribution before any optimization.

- Added diagnostic-only Gemma4 HQQ4/k4v4 boundary/stream residual attribution
  for the HD512 graph GQA path. The new report is timing/debug gated behind
  `KRASIS_DECODE_GQA_BOUNDARY_CLOCKS=1`, reuses existing graph clock markers,
  and changes no prefill, HCS policy, final segment behavior, QCN/non-HD512
  path, or output math. Build
  `20260613_175047_gemma4_hqq4_k4v4_boundary_attr_build.log` passed in
  `126s`. Timing run
  `20260613_175323_gemma4_hqq4_k4v4_boundary_attr_timing.log` used
  `KRASIS_DECODE_GQA_SPLIT_CLOCKS=1 KRASIS_DECODE_GQA_BOUNDARY_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed `14/14`, `ALL TESTS PASSED`, with `5566.7 tok/s` prefill,
  `64.09 tok/s` internal decode, `115.54 tok/s` HTTP, HCS `3840/3840`, min
  free `11472 MB`, clean graph HCS (`240/0` hit/miss, zero cold DMA/DMA
  calls), and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`.
  Boundary split across representative 49/99/249/511 rows showed GQA-entry
  gap `0.00/0.00/0.00/0.00 ms/tok`, post-O-proj exit gap
  `0.00/0.00/0.00/0.00`, and HQQ/graph stream debt
  `0.00/-0.00/-0.00/-0.00`. The active HD512 marked path was only
  `0.70/0.75/0.82/0.87`, while the mixed GQA-path total was
  `3.09/3.34/3.85/4.60`, proving the earlier path residual is a
  mixed-segment coverage/reporting artifact rather than hidden post-attention
  work. Exactly one next candidate is selected: decode-only coverage
  attribution for the mixed `GRAPH_SEG_ROUTE_GQA` label before any new
  optimization.

- Added diagnostic-only Gemma4 HQQ4/k4v4 post-attention residual attribution
  for the HD512 graph GQA path. The instrumentation remains timing/debug gated
  behind `KRASIS_DECODE_GQA_PATH_CLOCKS=1` and changes no prefill, HCS policy,
  final segment behavior, QCN/non-HD512 path, or output math. First build
  `20260613_172655_gemma4_hqq4_k4v4_post_attn_attr_build.log` passed in
  `124s`, but the first timing run
  `20260613_172936_gemma4_hqq4_k4v4_post_attn_attr_timing.log` did not emit
  the new path split because the added timing gate was too narrow; it still
  passed `14/14` and was cleaned up with `./dev kill` after the server stayed
  alive. After restoring the prior working path-clock gate shape, build
  `20260613_173518_gemma4_hqq4_k4v4_post_attn_attr_build2.log` passed in
  `124s`, and timing run
  `20260613_173800_gemma4_hqq4_k4v4_post_attn_attr_timing2.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5622.8 tok/s` prefill, `65.96 tok/s`
  decode, `115.79 tok/s` HTTP, HCS `3840/3840`, min free `11472 MB`, clean
  graph HCS (`240/0` hit/miss, zero cold DMA/DMA calls), and dynamic-HCS
  `promotions=0 evictions=0 copy_failures=0`. Post-attention split across
  representative 49/99/249/511 rows: `apply_gated_attn_bf16`
  `0.01/0.01/0.01/0.01 ms/tok`, O-proj input prep
  `0.00/0.00/0.00/0.00`, O projection `0.17/0.17/0.17/0.17`, marker residual
  `2.31/2.61/3.06/3.78`, and final graph sync debt
  `1.04/1.04/1.04/1.04`. This confirms the rejected gated BF16 reduce/output
  candidate was not targeting the dominant cost. Exactly one next candidate is
  selected: a decode-only boundary/stream residual attribution to split the
  remaining marker residual into GQA-entry gap, post-O-proj exit gap, and
  HQQ/graph stream debt before any new optimization.

- Measured and rejected a decode-only Gemma4 HQQ4/k4v4 gated BF16
  reduce/output candidate. The candidate inspected the existing boundary
  `gqa_attention_k4v4_tiled_g` -> `gqa_attention_polar4_reduce_g` writing
  FP32 `d_gqa_out` -> `apply_gated_attn_bf16` writing BF16 `d_scratch` ->
  BF16 O projection, then tried to fuse the gated BF16 conversion into the
  k4v4 Polar4 reduce/output stage. The runtime gate was captured graph GQA
  decode only: `GRAPH_SEG_ROUTE_GQA`, `layer_idx == range_end`,
  `kv_format == 9`, effective `head_dim == 512`, gated attention,
  `gqa_tile_size == 256`, `gqa_max_tiles > 1`, and allocated tiled buffers.
  Candidate `./dev build` passed in `133s`. Timing-enabled Gemma k4
  `KRASIS_DECODE_GQA_SPLIT_CLOCKS=1 KRASIS_DECODE_MIXED_SEGMENT_CLOCKS=1 KRASIS_DECODE_GQA_PATH_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  passed `14/14` and `ALL TESTS PASSED`, but regressed to `5399.5 tok/s`
  prefill, `63.25 tok/s` internal decode, and `115.34 tok/s` HTTP. HCS stayed
  clean (`3840/3840`, `240/0` hit/miss, `0` cold DMA/DMA calls,
  `copy_failures=0`) and min free remained `11474 MB`. The GQA-path residual
  bucket did not improve consistently: `2.41/2.62/3.03/3.78 ms/tok` across
  49/99/249/511 rows versus the attribution marker `2.32/2.60/3.09/3.78`.
  The candidate was rejected at the timing gate; no timing-off Gemma speed or
  QCN guard was run. Only the candidate source was reverted, no fused
  gated-reduce symbols remain, and restore `./dev build` passed in `132s`.

- Added diagnostic-only Gemma4 HQQ4/k4v4 HD512 graph GQA path sub-split timing
  behind `KRASIS_DECODE_GQA_PATH_CLOCKS=1`. The instrumentation is active only
  with graph internal timing and the runtime gate `kv_format == 9`, effective
  `head_dim == 512`, a mixed `GRAPH_SEG_ROUTE_GQA` segment, and active tiled
  graph buffers; prefill, HCS policy, final segment behavior, QCN/non-HD512
  paths, and output math are unchanged. Build passed in `124s`, then
  `KRASIS_DECODE_GQA_SPLIT_CLOCKS=1 KRASIS_DECODE_MIXED_SEGMENT_CLOCKS=1 KRASIS_DECODE_GQA_PATH_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  passed `14/14` and `ALL TESTS PASSED`, with `5614.9 tok/s` prefill,
  `64.23 tok/s` decode, `115.66 tok/s` HTTP, HCS `3840/3840`, min free
  `11474 MB`, clean graph HCS (`240/0` hit/miss, zero cold DMA/DMA calls), and
  dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. The requested GQA
  path split showed projection `0.23/0.23/0.23/0.23 ms/tok`, norm+RoPE+KV
  `0.05/0.06/0.06/0.06`, attention/reduce `0.22/0.27/0.37/0.41`, O projection
  `0.17/0.17/0.17/0.17`, and residual `2.32/2.60/3.09/3.78` across
  49/99/249/511 rows. The selected next candidate is a decode-only HD512 k4v4
  graph path that fuses the post-attention gated BF16 conversion into the k4v4
  reduce/output stage while preserving output math and BF16 O-projection input;
  do not retry rejected GQA/thread/score/single-tile/all-HCS candidates.

- Added diagnostic-only Gemma4 HQQ4/k4v4 mixed graph-segment attribution behind
  `KRASIS_DECODE_MIXED_SEGMENT_CLOCKS=1`. The timing pass split the coarse
  `experts L{n-1} + route L{n} GQA` label into MoE expert compute, full GQA
  path, route/top-k work, segment residual, and final sync without changing
  runtime math, HCS residency, prefill, or graph correctness checks. Gemma k4
  passed `14/14` with `5592.5 tok/s` prefill, `64.24 tok/s` decode,
  `113.71 tok/s` HTTP, HCS `3840/3840`, `240/0` hit/miss, zero cold DMA/DMA
  calls, and min free `11474 MB`. The split showed the full GQA path is the
  row-growing component (`3.07/3.24/3.92/4.58 ms/tok` across 49/99/249/511)
  while tiled attention plus Polar4 reduce remain tiny. The selected next
  candidate is a decode-only HD512 k4v4 graph GQA path sub-split into
  projection, Q/K norm + RoPE + KV-cache write, attention/reduce, and
  O projection before any new optimization; this avoids retrying the rejected
  tiled-attention/thread/score/single-tile/all-HCS paths or the earlier prefill
  post-projection fusions.

- Measured and rejected a decode-only Gemma4 HQQ4/k4v4 all-HCS graph replay
  fast-path candidate. The candidate reused the existing graph-side
  `expert_classify_prepare` pointer-fill path and replayed captured graph
  segments back-to-back only under the tight runtime gate: graph-mode k4v4,
  effective HD512, tiled graph buffers active, GPU classify/pointer-table
  support present, complete HCS coverage for the current decode segment, and
  no cold DMA required. QCN is excluded by `head_dim=256`, k6 by `kv_format`,
  and prefill was not touched. The first timing attempt was stopped before a
  correctness/speed decision because the gate blocked on dynamic-HCS
  bookkeeping even though Gemma showed `3840/3840` resident experts and zero
  cold DMA; the gate was adjusted to allow dynamic-HCS bookkeeping only after
  complete all-resident coverage was proven, while heatmap collection remained
  blocked. The second timing-enabled Gemma run passed `14/14` and `ALL TESTS
  PASSED`, with `5622.5 tok/s` prefill, `65.38 tok/s` internal decode,
  `116.42 tok/s` HTTP, HCS `3840/3840`, `240/0` hit/miss, `0` cold DMA,
  `0` DMA calls, and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`.
  The candidate removed per-segment route sync/upload attribution
  (`inter: 0.00 ms`, classify/upload/launch `0.00 ms`) but moved the wait to
  the final sync. Residual GQA-route rows versus the split attribution marker
  were mixed/worse: 49-token `9.16` vs `9.08`, 99-token `9.36` vs `9.03`,
  249-token `9.71` vs `9.82`, and 511-token `10.41` vs `10.18`. Because the
  required residual bucket did not improve consistently and internal decode
  stayed below the accepted `65.89 tok/s` marker, the candidate was rejected at
  the timing gate. No timing-off Gemma speed or QCN guard was run. The
  all-HCS fast-path source was reverted and `./dev build` passed afterward to
  restore the accepted installed extension.

- Added diagnostic-only Gemma4 HQQ4/k4v4 graph decode split timing for the
  accepted HD512 tiled-GQA path. The final accepted measurement uses an
  env-gated `record_globaltimer_u64_g` marker kernel, enabled only by
  `KRASIS_DECODE_GQA_SPLIT_CLOCKS=1`, to split
  `gqa_attention_k4v4_tiled_g` from `gqa_attention_polar4_reduce_g` inside
  graph replay without changing output math, HCS residency, prefill, or graph
  correctness checks. Earlier CUDA-event instrumentation was rejected after
  graph replay failed with `cuEventElapsedTime` invalid-value errors, and an
  Nsight Systems attempt stalled before usable kernel rows. The final
  timing-enabled Gemma run passed `14/14` and `ALL TESTS PASSED`, with
  `5429.2 tok/s` prefill, `64.67 tok/s` internal decode, `117.31 tok/s` HTTP,
  HCS `3840/3840`, min free `11474 MB`, and clean graph HCS (`240/0` hit/miss,
  `0` cold DMA, `0` copy failures). The split showed the two measured k4v4
  kernels are small versus the coarse GQA-route segment: 49-token row
  `0.20 + 0.02 = 0.21 ms/tok` versus `9.30 ms/tok` GQA route, 99-token
  `0.24 + 0.02 = 0.25` versus `9.29`, 249-token `0.35 + 0.02 = 0.37` versus
  `10.18`, and 511-token `0.39 + 0.02 = 0.40` versus `10.58`. The selected
  next candidate is therefore not another attention/reducer micro-kernel
  change: target the all-HCS-resident graph replay residual by removing the
  per-layer CPU route sync/upload loop when runtime HCS coverage is complete.
  No timing-off speed run or QCN guard was run for this attribution-only pass.

- Measured and rejected a decode-only Gemma4 HQQ4/k4v4 single-tile
  reducer-bypass candidate. The exact gate was graph-mode k4v4 decode with
  effective `hd == 512`, `tile_size == 256`, `max_tiles > 1`, allocated tiled
  buffers, and runtime device-side `num_tiles == 1`; prefill, HCS residency,
  graph correctness checks, final logits, k6, non-HD512 paths, and QCN
  `head_dim=256` were not intended to change. First `./dev build` failed
  before runtime testing because the added output pointer exceeded `cudarc`
  tuple launch arity; the launch plumbing was fixed with the existing raw
  parameter-vector style and the second build passed. Timing-enabled Gemma
  validation passed `14/14` and `ALL TESTS PASSED`, with benchmark summary
  `4904.0 tok/s` prefill, `64.09 tok/s` internal decode, `115.89 tok/s` HTTP,
  and min free `11474 MB`. HCS stayed clean, but GQA route timing did not
  improve consistently versus the accepted tiled timing marker, and internal
  decode regressed versus the accepted timing run (`64.94 tok/s`). Per the
  gate, no timing-off Gemma speed or QCN guard was run. The candidate was
  reverted and `./dev build` passed afterward to restore the accepted
  installed extension.

- Measured and rejected a decode-only Gemma4 HQQ4/k4v4 HD512 score-unroll
  candidate. The exact affected path was graph-mode k4v4 decode with effective
  `hd == 512`, `tile_size == 256`, `max_tiles > 1`, and existing tiled buffers;
  prefill, HCS residency, output math, graph segmentation, final logits, and
  non-HD512 paths were not changed. `./dev build` passed. Timing-enabled Gemma
  validation passed `14/14` with benchmark summary `4876.5 tok/s` prefill,
  `65.18 tok/s` internal decode, `114.41 tok/s` HTTP, and min free
  `11474 MB`; visible graph rows showed lower GQA route cost in representative
  rows while final stayed flat around `1.04-1.07 ms/tok`, so the timing-off
  speed gate and QCN guard were run. Timing-off Gemma validation passed
  network `14/14` but regressed against the accepted tiled-GQA marker:
  `5604.3 tok/s` prefill, `65.32 tok/s` decode, `116.10 tok/s` HTTP,
  HCS `3840/3840`, min free `11474 MB`, and `0` copy failures versus
  accepted `5613.8/65.89/118.74`. QCN guard completed with `6354.9 tok/s`
  prefill, `90.90 tok/s` decode, `153.41 tok/s` HTTP, HCS `15957/24576`,
  min free `896 MB`, and `0` copy failures; QCN `head_dim=256` is gated away
  from the HD512 branch. The candidate was rejected, the score-unroll branch
  was reverted, and `./dev build` passed to restore the accepted installed
  extension.

- Measured and rejected a decode-only Gemma4 HQQ4/k4v4 HD512 thread-count
  candidate. Latest tiled timing showed GQA route dominates graph-mode decode
  across 49/99/249/511-token rows (`9.20/9.52/10.11/10.65 ms/tok`) while the
  final segment stays flat around `1.04-1.05 ms/tok`, so the candidate tried
  launching the existing graph-mode k4v4 HD512 tiled GQA kernel with `512`
  attention threads instead of `256`. `./dev build` passed, and timing-enabled
  Gemma validation passed `14/14` with `5559.5 tok/s` prefill, `64.47 tok/s`
  internal decode, `117.56 tok/s` HTTP, and min free `11474 MB`; long-row GQA
  buckets improved (`249` `10.01` vs `10.11 ms/tok`, `511` `10.49` vs
  `10.65`), but the 49-token GQA bucket regressed and the final segment did
  not move. Timing-off Gemma validation passed `14/14` but regressed versus
  the accepted tiled-GQA marker: `5471.1 tok/s` prefill, `65.25 tok/s`
  internal decode, `118.26 tok/s` HTTP, HCS `3840/3840`, min free
  `11474 MB`, and `0` copy failures, versus accepted `5613.8/65.89/118.74`.
  QCN guard completed with `6348.3 tok/s` prefill, `88.73 tok/s` internal
  decode, `148.18 tok/s` HTTP, HCS `15957/24576`, min free `896 MB`, and `0`
  copy failures; QCN is gated away by `head_dim=256`. The candidate was
  reverted and `./dev build` passed afterward, restoring the accepted
  tiled-GQA decode source.

- Accepted a small decode-only Gemma4 HQQ4/k4v4 graph-mode tiled-GQA dispatch
  candidate after measured attribution showed the gap is GQA route/final
  segment replay time rather than HCS misses. The candidate is deliberately
  narrow: for graph-mode k4v4 HD512 decode with runtime tiled buffers present,
  use the existing tiled k4v4 GQA kernel plus Polar4 reduce path; otherwise keep
  the existing single-block k4v4 route kernel. Prefill code, HCS residency,
  output math, and correctness checks are not changed. `./dev build` passed.
  Timing-enabled validation
  `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  passed `14/14`, with summary `4691.3 tok/s` prefill, `64.94 tok/s`
  internal decode, `119.80 tok/s` HTTP, and min free `11474 MB`; HCS remained
  clean (`240/0` hit/miss, `0` cold DMA, `0` copy failures). Timing buckets
  were mixed but improved enough to run the speed gate: 49-token graph decode
  `15.23 ms/tok` vs prior `15.27`, 99-token `15.47` vs `15.63`, 249-token
  internal `16.12` vs `15.97`, and 511-token code-gen `16.61` vs `17.19`.
  Timing-off Gemma validation passed network `14/14` with `5613.8 tok/s`
  prefill, `65.89 tok/s` internal decode, `118.74 tok/s` HTTP, HCS
  `3840/3840`, min free `11474 MB`, and `0` copy failures. QCN guard
  `./dev speed-test` completed with `6351.4 tok/s` prefill, `87.72 tok/s`
  internal decode, `199.77 tok/s` HTTP, HCS `15957/24576`, min free `896 MB`,
  and `0` copy failures; QCN `head_dim=256`, so it is gated away from the new
  HD512 branch. Gemma decode remains far below the `200 tok/s` target.

- Ran a timing-enabled Gemma4 HQQ4/k4v4 q2 attribution pass on the current
  source before further optimization:
  `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`.
  The run passed `14/14` and `ALL TESTS PASSED`, with summary `5561.1 tok/s`
  prefill, `64.76 tok/s` internal decode, `116.67 tok/s` HTTP, and min free
  VRAM `11474 MB`. Graph-mode decode showed HCS is not the bottleneck:
  `100.00%` HCS hit rate, `240/0` hit/miss, `0.00 MB/tok` DMA, and `0` copy
  failures. The measured decode limiter is graph replay/sync wait around GQA
  route segments and the final segment: the 249-token graph row was
  `15.97 ms/tok`, with GPU compute `4.44 ms`, sync wait `11.23 ms`, GQA route
  `10.04 ms/tok`, and final sync about `1.04 ms`. The long code-gen row was
  `17.19 ms/tok` with sync wait `12.48 ms`. Long-prefill residuals remain in
  the five HD512 custom-tiled launches (`~2504.6 ms` over 5 calls on a
  14780-token inner row); KV append was only `3.2 ms` over 30 calls. No runtime
  code was changed in this attribution pass.

- Accepted the Gemma4 HQQ4 HD512 q2/wide full-attention specialization toward
  the RTX 5090 target (`10000 tok/s` prefill, `200 tok/s` decode, correctness
  required). `./dev build` passed first. Timing-enabled k4 validation
  `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  passed network `14/14` and showed `5563.7 tok/s` best prefill. Timing-off
  k4 validation `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` passed
  network `14/14` with `5613.0 tok/s` prefill
  (`2932.6`, `5613.0`, `4679.2`, `3794.3`, `3649.8`, `3572.9 tok/s` rows),
  `65.37 tok/s` decode, `117.61 tok/s` HTTP, HCS `3840/3840`, min free
  `11474 MB`, and `0` copy failures. The k6 guard
  `./dev test tests/gemma-4-4-hqq4-k6v6-a16.conf` passed network `14/14` with
  `4942.5 tok/s` prefill, `65.50 tok/s` decode, `117.91 tok/s` HTTP, HCS
  `3840/3840`, min free `11748 MB`, and `0` copy failures. Production QCN
  guard `./dev speed-test` passed with `6349.0 tok/s` prefill,
  `90.21 tok/s` decode, `199.75 tok/s` HTTP, HCS `15957/24576`, min free
  `928 MB`, and `0` copy failures. The Gemma prefill gain is accepted; Gemma
  decode remains far below target and Gemma min-free VRAM remains far above
  the default `600 MB` safety target because the tested Gemma configs already
  keep all `3840/3840` experts hot.

- Ran connectivity-only Zephyrus diagnosis after the pre-release matrix
  blocked. Existing notes still identify Zephyrus as `main@192.168.1.228`
  from `/home/main/Documents/BOX_3070.txt`. Local networking has an on-link
  route (`192.168.1.228 dev enp65s0 src 192.168.1.181`), but reachability
  fails below SSH: ping returned `Destination Host Unreachable`, neighbour
  lookup reported `192.168.1.228 dev enp65s0 FAILED`, ARP showed
  `(incomplete)`, and SSH still returned
  `ssh: connect to host 192.168.1.228 port 22: No route to host`. Full log:
  `benchmarks/20260613_093522_zephyrus_connectivity_diagnosis.log`. The
  pre-release matrix remains blocked; podman and model validation were not
  run.

- Began the pre-release environment validation path with no optimization
  changes. Existing docs/scripts identify the remote variants as Zephyrus
  installed-command Qwen3.5-35B-A3B HQQ4/k4v4 500 MB KV plus HQQ4+10% k4v4
  500 MB KV, and the podman variant as `krasis-run` with the Lore
  Qwen3.6-35B-A3B HQQ6+10% k6v6 config. Clean `./dev build` passed and was
  logged to `benchmarks/20260613_093124_prerelease_build.log`. The first
  formal Zephyrus preflight failed before any model run with
  `ssh: connect to host 192.168.1.228 port 22: No route to host`; full log:
  `benchmarks/20260613_093124_prerelease_zephyrus_preflight_failed.log`. Per
  the validation rule, the matrix stopped there and podman was not run as a
  substitute.

- Ran a final timing-off Gemma4 HQQ4 validation matrix on the current source
  state after the accepted k4v4 and k6v6 `head_dim=512` full-attention
  specializations. `./dev build` passed first. The k4v4 validation
  `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` passed network `14/14` with
  `4176.7 tok/s` prefill (`2824.5`, `4176.7`, `3013.0`, `2291.5`, `2283.2`,
  and `2214.2 tok/s` rows), `65.47 tok/s` decode, `117.99 tok/s` HTTP, HCS
  `3840/3840`, min free `11474 MB`, and `0` copy failures observed in the HCS
  rows. The k6v6 validation
  `./dev test tests/gemma-4-4-hqq4-k6v6-a16.conf` passed network `14/14` with
  `3864.2 tok/s` prefill (`1565.0`, `3864.2`, `2991.2`, `2865.7`, `2738.2`,
  and `2829.3 tok/s` rows), `65.69 tok/s` decode, `118.91 tok/s` HTTP, HCS
  `3840/3840`, min free `11748 MB`, and `0` copy failures observed in the HCS
  rows. Both remain above the accepted comparison points (`4056.2 tok/s` k4v4
  and `3750.8 tok/s` k6v6). No optimization candidate or QCN guard was run
  from this validation-only pass.

- Added and accepted a separate Gemma4 HQQ4/k6v6 stage-exact
  `head_dim=512` full-attention prefill specialization gate. Attribution showed
  k6v6 uses active FP8 stage-exact KV during prefill (`kv_format=1`,
  `decode_kv_format=7`, `prefill_kv_active=true`), while the attention fallback
  itself still consumes current BF16 `q/k/v` projection buffers before the
  stage-exact append. The existing k4v4 direct-cache gate remains unchanged;
  the new predicate is limited to Gemma4 HQQ4 k6v6 stage-exact full attention
  (`start_pos=0`, `window=0`, `head_dim=512`, `16` Q heads, `2` KV heads) and
  the same runtime opt-in shared-memory check. Timing-off
  `./dev test tests/gemma-4-4-hqq4-k6v6-a16.conf` passed network `14/14` and
  improved k6v6 prefill from the tracked `2268.3 tok/s` baseline to
  `3750.8 tok/s` best (`1579.0`, `3750.8`, `3013.3`, `2826.2`, `2735.9`, and
  `2864.0 tok/s` rows), with `65.68 tok/s` decode, `118.34 tok/s` HTTP, HCS
  `3840/3840`, min free `11748 MB`, and `0` copy failures. QCN guard
  `./dev speed-test` completed with `6357.4 tok/s` prefill, `88.91 tok/s`
  decode, `145.97 tok/s` HTTP, HCS `15957/24576`, min free `864 MB`, and `0`
  copy failures.

- Added timing-only Gemma4 HQQ4/k6v6 attribution by extending only the
  reporting gates that had been k4v4 direct-cache specific. The accepted
  `head_dim=512` specialization remains k4v4-only. Full
  `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k6v6-a16.conf --timing`
  passed `ALL TESTS PASSED` with timing-enabled benchmark summary
  `1647.9 tok/s` prefill, `64.31 tok/s` decode, `116.77 tok/s` HTTP, HCS
  `3840/3840`, min free `11748 MB`, and `0` copy failures. Timing rows were
  `1555.0`, `1647.9`, `1012.4`, `972.0`, `907.7`, and `892.2 tok/s`.
  The tracked timing-off k6 context remains baseline `2268.3 tok/s` and
  rejected candidate `2264.1 tok/s`. Attribution showed k6v6 prefill runs
  stage-exact (`kv_format=1`, `decode_kv_format=7`,
  `prefill_kv_active=true`), so the prior k6 HD512 gate extension did not
  match the real prefill path. The five `custom_no_fa2` full-attention layers
  are still `5/11/17/23/29`; on the 8419-token calibration row,
  `flash_attn_tiled_launch` owned `6313.9 ms` wall / `6314.0 ms` CUDA-event
  time over those five calls, while `kv_append_kernel` was only `2.1 ms` over
  30 calls. No optimization candidate or QCN guard was run.

- Rejected extending the Gemma4 HQQ4 `head_dim=512` full-attention
  specialization from k4v4 to k6v6. The existing k6v6 config
  `tests/gemma-4-4-hqq4-k6v6-a16.conf` was benchmarked timing-off first:
  baseline passed network `14/14` with `2268.3 tok/s` best prefill
  (`2268.3`, `1708.6`, `950.7`, `891.2`, `893.5`, `905.0 tok/s` rows),
  `65.82 tok/s` decode, `115.72 tok/s` HTTP, HCS `3840/3840`, min free
  `11748 MB`, and `0` copy failures. The k6v6 gate candidate passed network
  `14/14` but regressed slightly to `2264.1 tok/s` best prefill (`2264.1`,
  `1679.0`, `950.2`, `892.9`, `895.3`, `903.5 tok/s` rows), with
  `65.89 tok/s` decode, `118.24 tok/s` HTTP, HCS `3840/3840`, min free
  `11748 MB`, and `0` copy failures. The k6v6 gate was reverted and
  `./dev build` passed afterward, so the installed extension is back to the
  accepted k4v4-only gate. No QCN guard was run because no speed win was
  accepted.

- Added a gated Gemma4 HQQ4/k4v4 `head_dim=512` full-attention prefill
  specialization for the five unsupported-FA2 GQA layers. The new CUDA kernel
  keeps the existing causal fallback math but uses a fixed `head_dim=512` path
  with a larger `BC=32` KV tile, enabled only when runtime opt-in shared memory
  is sufficient and the layer is Gemma4 HQQ4/k4v4 direct single-chunk full
  attention (`start_pos=0`, `window=0`, `head_dim=512`, `16` Q heads, `2` KV
  heads, `prefill_kv_active=false`). Timing-off
  `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` passed network `14/14` and
  improved Gemma HQQ4/k4v4 prefill to `4056.2 tok/s` best (`1409.2`, `4056.2`,
  `3004.7`, `2219.5`, `2222.5`, `2211.8 tok/s` rows), with `65.31 tok/s`
  decode, `118.03 tok/s` HTTP, HCS `3840/3840`, min free `11474 MB`, and `0`
  copy failures. QCN guard `./dev speed-test` completed with `6365.4 tok/s`
  prefill, `87.28 tok/s` decode, `146.51 tok/s` HTTP, HCS `15957/24576`, min
  free `928 MB`, and `0` copy failures.

- Added timing-only `custom_no_fa2` / `custom_tiled` sub-step attribution for
  the Gemma4 HQQ4/k4v4 direct-cache GQA path. The instrumentation brackets the
  custom fallback wrapper only: entry, head-dim validation, layout/shared-memory
  math, cache-pointer selection, argument packing, pre-launch checkpoint, and
  the `flash_attn_tiled` launch. Full
  `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  passed network `14/14` with timing-enabled benchmark summary `2220.7 tok/s`
  prefill, `64.68 tok/s` decode, `116.62 tok/s` HTTP, HCS `3840/3840`, min
  free `11474 MB`, and `0` copy failures. Timing-row speeds were `4764`,
  `1806`, `990`, `729`, `689`, and `728 tok/s`. The host-side sub-steps were
  `0.0 ms`; `flash_attn_tiled_launch` owned the wait with event time
  `116.1 ms` on the 1K row, `2424.7 ms` on 5K, `9380.5 ms` on 10K,
  `13091.7 ms` on the 11,824-token calibration row, and
  `19171.5/20346.8/19175.0 ms` on the capped 14,780/14,780/14,779 rows. No
  optimization candidate or QCN guard was run.

- Added timing-only attention-branch attribution for the Gemma4 HQQ4/k4v4
  direct-cache GQA path. The instrumentation logs every Gemma HQQ4/k4 GQA call
  with layer, token count, attention type/window, branch, path, and fixed-FA2
  status, then adds before/after sync-debt checkpoints only around non-fixed
  attention branches. Full
  `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` passed network
  `14/14` with timing-instrumented benchmark summary `2097.3 tok/s` prefill,
  `64.77 tok/s` decode, `116.76 tok/s` HTTP, HCS `3840/3840`, min free
  `11474 MB`, and `0` copy failures. Timing-row speeds were `4909`, `1863`,
  `1047`, `728`, `689`, and `690 tok/s`. The five non-fixed calls are layers
  `5/11/17/23/29`, all `custom_no_fa2` / `custom_tiled` because
  `fa2_head_dim=false`. Their after-branch sync debt owns the long-row wait:
  `13075.6 ms` on the 11,824-token row, `8830.2 ms` on the 10K row, and
  `19179.9/20318.5/20313.0 ms` on the measured 14,780/14,780/14,779-token
  rows. No optimization candidate or QCN guard was run.

- Added timing-only append-gap sync-debt bisection for the Gemma4 HQQ4/k4v4
  direct-cache GQA path. The new `KRASIS_PREFILL_TIMING=1` checkpoints bracket
  the exact interval from fixed-length FA2 return/bookkeeping through attention
  branch exit, optional reference/trace hooks, append setup, k4 cache geometry,
  append timing-event record, and the direct-cache append launch. Full
  `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` passed network
  `14/14` with timing-instrumented benchmark summary `1630.4 tok/s` prefill,
  `66.10 tok/s` decode, `115.73 tok/s` HTTP, HCS `3840/3840`, min free
  `11474 MB`, and `0` copy failures. The old timing-off speed markers remain:
  prior unreproduced best `2303.8 tok/s`, latest timing-off best `1653.2 tok/s`.
  Timing-row speeds in this diagnostic run were `4803`, `1862`, `1048`, `729`,
  `687`, and `728 tok/s`. The 11,824-token attribution row showed
  `after_fa2_bookkeeping=0.0 ms`, then `after_attention_branch=13080.6 ms`,
  while all following reference/trace/append setup/event/launch checkpoints were
  ~`0.0 ms`; append itself was `2.5 ms` wall / `2.2 ms` event. The 14,780-token
  rows showed the same pattern (`after_attention_branch` `19162.0-20399.1 ms`).
  Since fixed FA2 had 25 calls but `after_attention_branch` had 30 calls, the
  producer is narrowed to unbracketed non-fixed-FA2 attention branch/layer work,
  not trace/reference hooks, append setup, append timing events, or the append
  kernel. No optimization candidate or QCN guard was run.
- Added timing-only sync-debt bisection for the Gemma4 HQQ4/k4v4 direct-cache
  GQA path. The new `KRASIS_PREFILL_TIMING=1` checkpoints force a stream sync
  at Gemma4 layer handoff, pre-GQA, GQA entry, after individual Q/K/V
  projections, after fixed-length FA2, immediately before append, and after O
  projection. On the 11,824-token row, the debt was absent at layer entry,
  GQA entry, and after Q/K/V (`0.0 ms` each over 30 calls), small at pre-GQA
  (`6.7 ms`), and moved to `pre_append`: `13063.3 ms` over 30 calls before
  append was launched. The append itself was only `2.2 ms` wall / `2.0 ms`
  event. On the 14,780-token warmup row, `pre_append` was `19119.8 ms`, while
  append was `3.2/3.0 ms`. This confirms the old append attribution was a sync
  placement artifact and narrows the producer to the interval after the
  post-FA2 checkpoint and before append launch, not Q/K/V projection, GQA
  entry, prior Gemma4 handoff, or the append kernel. The run was stopped after
  attribution; no optimization candidate or QCN guard was run.
- Added timing-only CUDA event instrumentation around the Gemma4 HQQ4/k4v4 GQA
  queue boundary after direct append timing ruled out the append kernel body.
  The diagnostic run bracketed Q/K/V projection, QKV norm, RoPE, fixed-length
  FA2, direct k4/v4 append, O projection, and the explicit stream-sync waits
  while preserving existing wall buckets. On the 11,824-token row, projection
  was `181.3 ms` wall / `174.6 ms` event, QKV norm `11.7/11.5`, RoPE
  `6.8/6.6`, FA2 `137.6/137.4`, O projection `92.6/92.4`, but direct append
  was `13066.9 ms` wall with only `1.9 ms` append-kernel event time. On the
  14,780-token warmup row the same pattern held: append wall `20260.6 ms`,
  append event `2.9 ms`, while projection/FA2/O were close to their event
  times. Conclusion: the 13-20s debt is not produced by Q/K/V projection,
  norm/RoPE, FA2, O projection, or the append kernel body; it appears as an
  explicit stream-sync wait at the append boundary. The timing run was stopped
  after attribution and no optimization candidate or QCN guard was run.
- Added timing-only CUDA event instrumentation for the accepted Gemma4
  HQQ4/k4v4 direct-cache `kv_append` route. The diagnostic run showed the old
  wall/sync `kv_append` bucket was a synchronization attribution artifact, not
  a slow append kernel: on the 11,824-token row, wall `kv_append` was
  `13070.1 ms`, but `kv_append_kernel` was only `2.0 ms` over 30 calls
  (`13068.1 ms` wall-minus-kernel). On a 14,780-token warmup row, wall
  `kv_append` was `19114.7 ms` while `kv_append_kernel` was `3.0 ms`. No
  replacement k4/v4 append kernel was built because the kernel body is not the
  measured bottleneck. A timing-off validation run after the instrumentation
  passed network `14/14` with `65.41 tok/s` decode, `117.73 tok/s` HTTP, HCS
  `3840/3840`, `0` copy failures, and `11474 MB` minimum free VRAM, but did
  not produce a new speed win: best prefill was `1653.2 tok/s` because the 1K
  row did not reproduce the earlier `2303.8 tok/s` best. The longer rows were
  comparable to the accepted baseline, so this remains attribution-only
  instrumentation rather than an accepted speed change.
- Tested a deeper Gemma4 HQQ4/k4v4 projection-backend variant after the
  post-projection fusion attempts failed. A timing-enabled attribution run
  confirmed the long-row bottleneck is still KV append/staging: `11824` tokens
  spent `13082.4 ms` in `kv_append`, while HQQ projection internals were much
  smaller (`237.0 ms` Marlin float-zp, `5.2 ms` group sums, `10.4 ms`
  correction GEMM, `18.4 ms` correction add). A base-only HQQ4 Marlin
  projection experiment skipped the HQQ4 two-scale/intercept repair and passed
  network `14/14`, but regressed timing-off prefill to `1742.1 tok/s` best
  with visibly degraded code generation, versus the accepted HQQ4/k4v4
  stage-exact skip at `2303.8 tok/s`. The experimental backend path was
  reverted and `./dev build` passed afterward; no new runtime default was
  accepted.
- Ran a larger Gemma4 HQQ4/k4v4 fused cache-prep candidate after the isolated
  Q/K fusion failed. The candidate fused Q norm+half-split RoPE, K
  norm+half-split RoPE+final k4 cache write, and V no-scale RMSNorm+final v4
  cache write for the already validated single-chunk direct-cache HQQ4/k4v4
  route. It built and passed `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf`
  with network `14/14`, decode `65.13 tok/s`, HTTP `117.36 tok/s`, HCS
  `3840/3840`, and `0` copy failures, but regressed prefill to `1710.7 tok/s`
  best (`1552.5`, `1710.7`, `969.3`, `713.5`, `667.0`, `668.8` row speeds)
  versus the current accepted HQQ4/k4v4 skip at `2303.8 tok/s`. The fused
  kernels and gate were reverted, `./dev build` passed afterward, and no new
  runtime default was accepted.
- Ran a Gemma4 fused HQQ/GQA kernel pass. Rejected a fused Q/K RMSNorm plus
  half-split RoPE prefill kernel because it preserved startup but severely
  regressed timing-off HQQ4/k4v4 prefill: early rows were `1659.5`, `1633.2`,
  `958.4`, `673.8`, `668.5`, and `669.0 tok/s` before the run was stopped,
  versus the accepted HQQ4/k4v4 skip at `2303.8 tok/s`. Rejected broadening
  the stage-exact skip from Gemma4 HQQ4/k4v4 to HQQ8/k4v4 after the timed
  prefill row stalled with GPU0 pegged and no benchmark row output in the
  bounded run. Both candidates were reverted, `./dev build` passed afterward,
  and the only active runtime speed diff remains the previously validated
  HQQ4/k4v4 single-chunk stage-exact skip.
- Ran a Gemma4 HQQ prefill mode optimization pass. Rejected persistent BF16
  materialization from HQQ4 because it passed network `14/14` but regressed
  best prefill to `1707.7 tok/s` and reduced min free VRAM to `9330 MB`.
  HQQ8 `native-fused-marlin` improved HQQ8/k4v4 prefill from `1590.0` to
  `1734.0 tok/s` with `65.26` decode, `119.35` HTTP, network `14/14`, and
  HCS `3840/3840`. HQQ8 `native-fused-marlin-v2` was the best HQQ8 format:
  `1855.3` prefill, `66.67` decode, `119.10` HTTP, network `14/14`, and HCS
  `3840/3840`. Rejected HQQ8 `symmetric-marlin` because it was slower
  (`1652.3` prefill) and produced a visibly weaker multi-turn sample despite
  passing the short suite. No new runtime default was accepted because even the
  best HQQ8 mode remains below the current HQQ4/k4v4 skip (`2303.8`) and far
  below Gemma BF16/k4v4 (`5196.5`).
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
- Gemma4 fixed HQQ attention modes are now supported for non-ring compressed
  KV. The model constructor accepts Gemma4 `attention_quant` values `bf16`,
  `hqq4`, `hqq6`, and `hqq8`, while continuing to reject mixed/auto HQQ modes,
  non-INT4 routed experts, and KV formats outside BF16/k6v6/k4v4. Gemma4 HQQ
  registration now uses split Q/K/V descriptors for GQA layers instead of the
  generic fused-QKV HQQ descriptor path, because Gemma4 has mixed
  sliding/full-attention geometry and the fused path failed startup
  calibration on full-attention layers. Validation covered all fixed-HQQ
  combinations with `./dev test`: HQQ8/k6v6 `1838.4` prefill / `66.56`
  decode, HQQ6/k6v6 `1632.5` / `63.85`, HQQ4/k6v6 `1706.5` / `65.39`,
  HQQ8/k4v4 `1590.0` / `65.42`, HQQ6/k4v4 `1628.2` / `64.43`, and HQQ4/k4v4
  `2150.7` / `65.25`; all passed `14/14` network prompts with HCS
  `3840/3840` and `0` copy failures. HQQ attention saves VRAM but is slower
  than the BF16 attention fast path in prefill. HQQ4 also produced a visibly
  weaker code-generation sample despite passing the short network test, so
  Gemma4 HQQ should not be treated as llama-witness validated.
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
- Extended Gemma4 CUDA graph decode to the validated non-ring `k4v4` path.
  The first broad k4 graph gate was rejected because pre-HCS calibration
  attempted graph replay and logged `materialized 0 routed experts`; the
  accepted gate now requires HCS residency before enabling k4 graph replay.
  Clean timing-off validation `./dev test tests/gemma-4-4-k4v4-a16.conf`
  improved k4v4 decode from `38.79` to `63.69 tok/s` and HTTP from `66.92`
  to `115.47 tok/s`, with `5192.4 tok/s` best prefill, `14/14` network
  prompts, HCS `3840/3840`, `0` copy failures, and `11030 MB` minimum free
  VRAM. The QCN `./dev speed-test` guard completed afterward at `6856.8`
  prefill, `88.67` decode, `149.12` HTTP, HCS `15957/24576`, `0` copy
  failures, and `896 MB` minimum free VRAM.
- QCN `./dev speed-test` guard after the Gemma4 HQQ change completed at
  `6902.6 tok/s` prefill, `89.37 tok/s` decode, `149.29 tok/s` HTTP, HCS
  `15957/24576`, `0` copy failures, and `928 MB` minimum free VRAM.
- Improved Gemma4 HQQ4/k4v4 prefill by skipping stage-exact temporary KV only
  when runtime scratch budgeting proves the prompt fits as a single chunk. The
  accepted gate is deliberately narrow: Gemma4, fixed HQQ4 GQA, non-ring k4v4
  decode KV, uniform per-layer KV capacity, and measured single-chunk fit.
  Other Gemma HQQ modes and all non-Gemma paths keep the existing stage-exact
  path until separately validated. Timing diagnostics showed HQQ4/k4v4 prefill
  is still dominated by GQA/KV append (`11824` tokens: `13091.5 ms` KV append
  inside `13521.6 ms` GQA), not HQQ projection. Rejected
  `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1` because it regressed prefill to
  `1702.7 tok/s`. The accepted final `./dev test
  tests/gemma-4-4-hqq4-k4v4-a16.conf` improved best prefill from `2150.7` to
  `2303.8 tok/s`, with `65.10 tok/s` decode, `117.49 tok/s` HTTP, `14/14`
  network prompts, HCS `3840/3840`, `0` copy failures, and `11474 MB` minimum
  free VRAM. QCN `./dev speed-test` guard passed afterward at `6356.9`
  prefill, `90.14` decode, `148.29` HTTP, HCS `15957/24576`, `0` copy
  failures, and `928 MB` minimum free VRAM.
- Investigated extending the Gemma4 fast-path prefill and graph-decode
  optimizations beyond the accepted HQQ4/k4v4 gate. The BF16/k4v4 baseline
  `./dev test tests/gemma-4-4-k4v4-a16.conf` measured `5196.5 tok/s`
  prefill, `64.36 tok/s` decode, `115.43 tok/s` HTTP, network `14/14`, HCS
  `3840/3840`, and `11030 MB` minimum free VRAM. Rejected
  `KRASIS_KV_STAGE_EXACT=0` for BF16/k4v4: it reduced long-prefill transient
  memory from `5586 MB` to `4306 MB`, but regressed speed to `5059.5` prefill,
  `63.68` decode, and `114.45` HTTP. Decode graph timing confirmed the
  post-HCS BF16/k4v4 path is already in graph mode; a representative 79-token
  block was `15.59 ms/tok`, with graph segment CUDA time dominated by GQA
  route segments (`9.31 ms/tok`) and final (`1.05 ms/tok`) with no cold DMA.
  Rejected `KRASIS_GPU_ROUTE_SYNC=1` as a graph-decode improvement because it
  passed network `14/14` but regressed to `5051.3` prefill, `62.64` decode,
  and `113.71` HTTP. No new BF16 fast-path runtime change was accepted from
  these probes.
- Ran a focused Gemma4 BF16/k4v4 KV-staging optimization pass with
  prefill/GQA timing enabled. Baseline instrumentation confirmed the long
  calibration row is dominated by stage-exact temporary FP8 KV append:
  `11824` tokens spent `12829.3 ms` in `kv_append` inside `13007.6 ms` GQA,
  with `1270.2 MB` temporary FP8 K/V for 30 active layers. Rejected a 2D
  token-by-KV-segment launch for the generic FP8 append kernels because it was
  unchanged/slower (`12836.7 ms` `kv_append`, `13219.5 ms` total long row).
  Rejected a CUDA `__nv_fp8x2_e4m3` vectorized append candidate: timing
  instrumentation improved the long row to `12505.5 ms` `kv_append`, but the
  timing-off validation regressed BF16/k4v4 best prefill to `5033.4 tok/s`
  with `63.23 tok/s` decode and `115.50 tok/s` HTTP. Network still passed
  `14/14`, HCS stayed `3840/3840`, copy failures were `0`, and min free VRAM
  was `11030 MB`. The vectorized candidate was reverted, and no new
  KV-staging runtime change was accepted.
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
