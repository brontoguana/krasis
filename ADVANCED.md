# Advanced Configuration

## Install Pre-release

```bash
curl -sSf https://raw.githubusercontent.com/brontoguana/krasis/main/install.sh | bash -s -- prerelease
```

This installs the latest pre-release build. Normal `install.sh` (without the `prerelease` flag) always installs the latest stable release.

## Running Krasis

Krasis has two entry points:

- `krasis` — the installed command (use for production, release testing)
- `./dev` — the development entry point (handles conda, auto-rebuild, GPU cleanup)

Never run Python scripts directly. Always use one of these commands.

BF16 validation policy:

- BF16-heavy configs are validation-only. Use them to prove correctness or isolate quantization from logic bugs.
- Production runs must use the normal Rust serving path with quantized configs.
- `gpu_expert_bits = 16` is not a production mode.

### Dev Commands

| Command | Description |
|---------|-------------|
| `./dev build` | Rebuild Rust extension (maturin develop --release) |
| `./dev run <config> [flags]` | Launch server from a test config |
| `./dev benchmark <config>` | Run standard benchmark and exit |
| `./dev release-test <model>` | Run full release test (release config matrix, produces markdown report) |
| `./dev test <config>` | Short model test (benchmark + network tests) |
| `./dev test <config> --thorough` | Thorough test (+ stress + large prompts) |
| `./dev network <port> [--large] [--quick]` | Run network tests against a running server |
| `./dev perplexity <config>` | Run perplexity eval (WikiText-2) and exit |
| `./dev awq-calibrate <config>` | Deprecated AWQ calibration command; disabled in the active runtime surface |
| `./dev kill` | Kill all krasis/GPU processes and reset |

Add `--timing` to `run` or `benchmark` for a per-layer decode timing breakdown. This adds ~30-50% overhead so do not use it for speed benchmarks — only for profiling.

`KRASIS_BENCHMARK_GPU_TELEMETRY=1` enables diagnostic SM-clock,
temperature, and power sampling during decode rows. It is default-off because
speed benchmarks must not run with instrumentation enabled.

## Config Files

The preferred way to run Krasis is with a config file:

```bash
krasis --config path/to/config.conf
./dev run qcn            # resolves to testconfigs/qcn.conf
./dev benchmark qcn      # same
./dev run dsv4           # DeepSeek-V4-Flash-0731 production config
```

Config files use `KEY=VALUE` format. CLI flags override config file values.

`dsv4`, `deepseek-v4`, and `deepseek-v4-flash-0731` resolve to the validated
DeepSeek-V4-Flash-0731 INT4-expert, BF16-attention/KV config. GPU selection,
layer partitioning, HCS residency, and VRAM budgets remain runtime-measured;
the named config does not encode a particular GPU or fixed residency budget.

## Tool Use

The OpenAI-compatible `/v1/chat/completions` endpoint accepts `tools` in both
streaming and non-streaming requests. Krasis derives the native output grammar
from the loaded chat template, renders tool definitions and prior tool turns
with that template, and translates generated calls back to OpenAI structured
`tool_calls`. Typed arguments, multiple calls in one response, assistant
`tool_calls` history, and `tool`-role results are supported.

| Model family | Native output grammar | Krasis tool-use status |
|---|---|---|
| DeepSeek-V4-Flash | DSML `tool_calls` / typed `invoke` parameters | Supported |
| Qwen3 base / Qwen3-235B | JSON inside `<tool_call>` | Supported |
| Qwen3-Coder-Next | Function/parameter XML inside `<tool_call>` | Supported |
| Qwen3.5 / Qwen3.6 | Function/parameter XML inside `<tool_call>` | Supported |
| Ornith 35B / 397B | Function/parameter XML inside `<tool_call>` | Supported |
| Step-3.7-Flash | Function/parameter XML inside `<tool_call>` | Supported |
| Nemotron-3 Nano / Super | Function/parameter XML inside `<tool_call>` | Supported |
| GLM-4.7 / GLM-5.2 | Function name plus `arg_key` / `arg_value` XML | Supported; model runtime remains preview |
| Gemma 4 | Native `call:name{...}` grammar | Supported |
| MiniMax M2.x | Native `minimax:tool_call` grammar | Supported |
| DeepSeek-V2 / V2-Lite / VL2 | No tool grammar in the shipped fallback template | Not supported by the current template |

Detection is template-contract based rather than a list of model names. This
means a checkpoint with an absent or changed grammar fails visibly on a tools
request instead of being passed through as raw markup or parsed as a different
family. Malformed or truncated tool blocks remain ordinary assistant text and
never cause a parser panic.

## Server Flags

### Core

| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | — | Config file (KEY=VALUE format), CLI flags override |
| `--model-path PATH` | — | HuggingFace model directory (safetensors + config.json) |
| `--num-gpus N` | all | Number of GPUs to use |
| `--selected-gpus IDX` | all | Comma-separated GPU indices (e.g. `0,2`) |
| `--pp-partition STR` | auto | Layer partition across GPUs (e.g. `24,24`) |
| `--host ADDR` | 0.0.0.0 | Server bind address |
| `--port PORT` | 8012 | Server port |
| `--ssh-tunnel TARGET` | off | Reverse SSH tunnel target (`user@host` or `user@host:ssh_port`). Remote `127.0.0.1:<server port>` forwards to local Krasis over SSH with key-only batch mode. |
| `--ssh-key-path PATH` | off | Optional identity file for `--ssh-tunnel`; passed to `ssh -i` with `IdentitiesOnly=yes`. |

### Quantization

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu-expert-bits` | 4 | GPU Marlin expert bits: `4` or `8` |
| `--cpu-expert-bits` | 4 | CPU decode expert bits: `4` or `8` |
| `--attention-quant` | bf16 direct, hqq6 launcher | Attention weight precision: interactive launcher presets are HQQ4, HQQ4+10% (`hqq46_auto`), HQQ6, and HQQ6+10% (`hqq68_auto`); `hqq8`, `hqq46`, and `bf16` remain explicit advanced modes |
| `--shared-expert-quant` | int8 | Shared expert quant: `int8` or `bf16` |
| `--dense-mlp-quant` | int8 | Dense MLP quant: `int8` or `bf16` |
| `--lm-head-quant` | int8 | LM head quant: `int8` or `bf16` |
| `--vision-quant` / `--step-vision-quant` | int4 | Lazy vision tower quantization for image requests: `int4` default, or `bf16` for validation |
| `--vision-group-size` / `--step-vision-group-size` | 128 | Vision INT4 row group size: `32`, `64`, or `128` |
| `--kv-dtype` | k6v6 | KV cache format: `k6v6` Quality, `k4v4` Ultra Compact, or `bf16` Full Precision |

AWQ attention and Polar4 KV are deprecated and disabled for new runs. Their
implementation remains in the tree for historical reference, but active
configs should use HQQ attention plus `k6v6`, `k4v4`, or `bf16` KV.

When BF16 is selected for experts or major components, treat that run as validation-only rather than production.

### Memory & Caching

| Flag | Default | Description |
|------|---------|-------------|
| `--kv-cache-mb N` | 1000 | KV cache size in MB |
| `--max-context-tokens N` | 0 | Explicit runtime context cap; `0` uses the model-declared limit. KV allocation, request limits, prefill scratch sizing, and heatmap compatibility use the effective cap. |
| `--hcs` / `--no-hcs` | on | Hot Cache Strategy for expert pinning |
| `--multi-gpu-hcs` | off | Pin HCS experts across all GPUs |
| `--hcs-host-cache-mode MODE` | source | Soft HCS host storage: `source`, `mirror`, or `auto` |
| `--dynamic-hcs` / `--no-dynamic-hcs` | on | Dynamic HCS: protect the high-ranked heatmap prefix and reserve a recency-adaptive tail |
| `--dynamic-hcs-tail-blocks N` | 2 | Advanced dynamic HCS recency-tail size, measured in activated-expert blocks; valid range `1..5` |
| `--vram-safety-margin N` | 600 | Reserved VRAM in MB below which warnings fire |
| `--stream-attention` | off | Stream attention weights from CPU (for very large models) |
| `--force-load` | — | Override RAM safety checks and load anyway |
| `--force-rebuild-cache` | — | Delete existing expert caches and rebuild from safetensors |
| `--build-cache` | — | Build expert caches (if missing) and exit without starting server |
| `--heatmap-path PATH` | — | Path to expert_heatmap.json for HCS init |
| `--approved-heatmap-mode MODE` | auto | Approved route-heatmap lookup: `auto`, `off`, or `require` |
| `--approved-heatmap-manifest-url URL` | GitHub manifest | Override the approved route-heatmap manifest URL |

Dynamic HCS uses the same physical HCS residency table as the heatmap cache.
It does not create a second cache or allow duplicate expert residency across a
heatmap region and a recency region. The default keeps the heatmap prefix and
reserves two activated-expert blocks for recency promotion; use
`--dynamic-hcs-tail-blocks 1..5` for model-specific tuning, or
`--no-dynamic-hcs` to run static heatmap HCS only.

When `--heatmap-path` is not supplied, `--approved-heatmap-mode auto` checks the
approved heatmap manifest and uses a matching checksum-verified artifact from
the local cache or GitHub. Approved heatmaps provide only the expert route
ranking; Krasis still calibrates VRAM locally and sizes HCS residency at
startup. If the manifest or listed artifact is not downloadable in `auto` mode,
Krasis logs the fallback and runs the quick local startup heatmap. Use `off` to
force the quick local startup heatmap, or `require` to fail startup unless an
approved artifact is available for the current model/router signature and
validated runtime.

`./dev approved-heatmap-build` uses calibrated HCS residency while it captures
the larger reusable route corpus. A compatible `--resume-from`,
`--bootstrap-from`, or config `CFG_HEATMAP_PATH` seeds residency before the
first prompt. Without one, only the first prompt runs cold; Krasis then rebuilds
the normal reclaimable HCS pool from cumulative captured routes before later
prompts. `--residency-refresh-every N` controls that rebuild cadence and
defaults to every prompt. Bootstrap counts affect residency only and are never
added to the new artifact; resume counts remain part of the cumulative output.

`--hcs-host-cache-mode source` is the default RAM-saving mode. It skips the
duplicate soft HCS host mirror and reloads soft HCS chunks from the Marlin host
cache, reducing system RAM at the cost of slower reloads. Use `mirror` to opt
back into the old faster-reload path with duplicated pre-packed host chunks.
`auto` keeps mirror mode when system RAM is sufficient and switches to source
mode when the measured available RAM cannot safely hold the soft mirror.

### Prefill & Decode

| Flag | Default | Description |
|------|---------|-------------|
| `--layer-group-size N` | 2 | MoE layers to load per group during prefill |
| `--gpu-prefill-threshold N` | 300 | Minimum tokens to use GPU prefill |
| `--krasis-threads N` | 40 | CPU threads for expert computation |
| `--gguf-path PATH` | — | GGUF file for CPU experts (instead of native cache) |

Experimental decode environment variables (default off; single-GPU graph decode
path). Previous-token prefetch and adaptive cold-drop require host-visible
legacy route synchronization; split expert launch supports legacy and ordinary
GPU route synchronization but fails closed with the separate all-hot no-sync
graph contract:

| Env var | Default | Description |
|------|---------|-------------|
| `KRASIS_PREFETCH=1` | off | Previous-token route prefetch: stages predicted cold experts on a dedicated copy stream one token ahead so demand cold DMA shrinks. Staged experts are consumed only if their copies already completed (consume-if-ready); late slots fall back to demand copies. Issuance is capped per token by a byte budget derived from a measured H2D bandwidth probe and the previous token's wall time and demand traffic |
| `KRASIS_PREFETCH_DEPTH=N` | 4 | Prefetch lookahead depth in MoE layers (staging VRAM = depth × top-k × expert size) |
| `KRASIS_PREFETCH_BUDGET_OFF=1` | off | Disable the per-token prefetch byte budget (A/B diagnostics only) |
| `KRASIS_PREFETCH_GATE=0` | on | Disable the demand-first temporal gate. When the gate is on (default with prefetch), each prefetch issuance waits GPU-side on the boundary's demand cold-DMA event, so prefetch bytes transfer during the segment's compute window instead of contending on the copy engine with the demand copies the graph is waiting on |
| `KRASIS_SPLIT_EXPERT_LAUNCH=1` | off | Launch hot/staged experts before the cold-DMA wait so hot compute overlaps demand copies. Exact full/hot/cold weight masks preserve every routed contribution; compatible with legacy and ordinary GPU route synchronization. Latent-MoE graphs fail closed because their expert-input projection executes inside the captured segment and is not available for a safe pre-launch |
| `KRASIS_CPU_TAIL_RACE=1` | off | After any adaptive cold-mass pruning, let a persistent Rust worker attempt the lowest-ranked surviving demand-cold expert directly from the existing INT4 Marlin host allocation while the GPU copies the other cold experts. A completed CPU result replaces that expert's weight H2D and GPU GEMV; a miss falls back to the normal GPU path without waiting. Requires split launch and legacy host-visible route synchronization; speculative decode, prefetch/APFL, GPU route sync, non-INT4, latent, biased, ungated, clamped, and non-SiLU experts fail visibly. CPU activations use the existing INT16-quantized Marlin-native kernel, so this is a separate approximate quality-gated mode even when cold-mass pruning is off |
| `KRASIS_CPU_TAIL_WORKERS=2` | off | Experimental second CPU-tail worker. Requires `KRASIS_CPU_TAIL_RACE=1` and a `KRASIS_CPU_TAIL_CALIBRATION_JSON` artifact whose matched live-DMA two-team optimizer recommends a non-overlapping two-team split. The runtime revalidates both CPU sets against the current affinity mask and fails visibly on absent, overlapping, or stale placement evidence. At most two experts are claimed per cold queue; unset or `1` preserves single-worker behavior |
| `KRASIS_CPU_TAIL_TRANSPOSED=1` | off | Temporary CPU-tail architecture experiment; also requires `KRASIS_CPU_TAIL_RACE=1`. After startup HCS selection, duplicate every non-resident routed expert into a CPU-transposed INT4 layout while leaving the Marlin host cache and every GPU/prefill/HCS path unchanged. Required bytes are derived from the actual selected set and tensors, checked against runtime `MemAvailable` plus proportional headroom, and allocation fails visibly rather than building a partial tier. Later-evicted startup residents use Marlin fallback; promotions retain unused duplicates. This can consume hundreds of GiB and is not a production cache mode |
| `KRASIS_MARLIN_AUTOTUNE=1` | off | Measure batched w13 GEMV ksplit candidates (median of repeated timed blocks) on the real loaded expert shape at graph capture; overrides the occupancy formula only when a candidate beats the formula's own median by more than the margin |
| `KRASIS_MARLIN_AUTOTUNE_MARGIN_PCT=N` | 5 | Minimum relative win (percent) over the formula candidate before an autotune override is installed |
| `KRASIS_ADAPTIVE_COLD_DROP=1` | off | Enable approximate demand-cold expert pruning. Only routed experts that would require a demand DMA are eligible; surviving router weights are not renormalized. Requires both percentage variables below. Results depend on runtime HCS residency, so this mode can vary with VRAM and cache state |
| `KRASIS_ADAPTIVE_COLD_DROP_PROTECT_PCT=N` | — | Protect the leading `N` percent of router ranks in every layer from pruning (for example, `75` protects the leading 75% of a layer's top-k positions) |
| `KRASIS_ADAPTIVE_COLD_DROP_MASS_PCT=N` | — | Maximum fraction of that layer's total routed weight that may be dropped, expressed as a percentage. Eligible cold routes are considered from lowest weight upward and admitted only while this per-layer cap is respected |
| `KRASIS_ADAPTIVE_COLD_DROP_SHADOW_PROTECT_PCTS=A,B,...` | off | Shadow-only rank-protection sweep. Requires the shadow mass list below; records projected drops/bytes/mass without changing routes or outputs |
| `KRASIS_ADAPTIVE_COLD_DROP_SHADOW_MASS_PCTS=A,B,...` | off | Shadow-only per-layer routed-mass sweep. Actual and shadow modes are mutually exclusive |

Adaptive cold drop currently supports normal single-token Rust MoE decode and
the legacy host-visible route-sync CUDA-graph path. It fails visibly rather
than silently degrading when combined with GPU route sync or speculative
decode. This is an explicit quality/performance tradeoff and is never enabled
automatically.

The CPU-tail race is also explicit and experimental. It uses only the existing
Marlin cache, uploads a successful expert's BF16 output into the captured
expert-output slot, and reports attempts, wins, deadline misses, saved demand
bytes, and CPU/input-copy timing at request completion. Its single pinned
output buffer is protected by a CUDA completion event; a later layer uses
ordinary GPU DMA rather than waiting when that buffer is still in flight.
The separate transposed-tier flag is a temporary RAM-duplication experiment:
the tier is Rust-owned, freed with the decode store, and reports its expert
coverage, bytes, conversion wall, per-layout attempts/wins, and per-layout
worker time. It does not change the on-disk cache format.

The interactive launcher exposes this as **Adaptive cold-mass pruning**. It
defaults to `Off`; Left/Right cycles through `Off`, `75/3`, `75/5`, `75/8`,
and `75/10`. Saved launcher configs use
`CFG_ADAPTIVE_COLD_MASS_PRUNING="off|75/3|75/5|75/8|75/10"`. The server
translates a selected preset into the three Rust environment variables above;
invalid presets fail during argument parsing. Server launches without this
high-level config key continue to honor explicitly supplied low-level
environment variables for diagnostics.

### Speculative Decoding

| Flag | Default | Description |
|------|---------|-------------|
| `--draft-model PATH` | — | Draft model for speculative decoding (e.g. `~/.krasis/models/Qwen3-0.6B`) |
| `--draft-k N` | 3 | Tokens to draft per speculative round |
| `--draft-context N` | 512 | Context window for draft model warmup |

### Inference Options

| Flag | Default | Description |
|------|---------|-------------|
| `--temperature F` | 0.6 | Sampling temperature |
| `--enable-thinking` / `--no-enable-thinking` | on | Enable thinking/reasoning mode |

### Benchmarking & Testing

| Flag | Default | Description |
|------|---------|-------------|
| `--benchmark` | — | Run benchmark before launching server |
| `--benchmark-only` | — | Run benchmark and exit (no server) |
| `--timing` | — | Enable per-layer decode timing instrumentation |
| `--stress-test` | — | Run stress test (diverse prompts) and exit |
| `--perplexity` | — | Run perplexity evaluation and exit |
| `--note TEXT` | — | Description note written to log file header |

## Per-Component Quantization Summary

Krasis lets you quantize each component independently. The defaults are a good starting point — increase precision if you need better quality, decrease if you need to fit in less VRAM/RAM.

| Component | Options | Default |
|-----------|---------|---------|
| GPU experts | INT4, INT8 | INT4 |
| CPU experts | INT4, INT8 | INT4 |
| Attention | HQQ4, HQQ4+10%, HQQ6, HQQ6+10%, HQQ8, BF16 | BF16 |
| Shared expert | INT8, BF16 | INT8 |
| Dense MLP | INT8, BF16 | INT8 |
| LM head | INT8, BF16 | INT8 |
| Lazy vision towers | INT4, BF16 | INT4 |
| KV cache | k6v6, k4v4, BF16 | k6v6 |

Embeddings, norms, routing gates, and vision norms/positional plumbing are always kept at BF16. Vision towers are lazy-loaded on the first image request, staged on GPU only for image embedding generation, then released back to CPU.

HQQ attention artifacts live under the normal model cache tree and the runtime restores staged prefill/decode descriptors from that cache. AWQ attention is deprecated and disabled for new runs; do not use it for production validation. BF16 is full precision with no calibration needed.
