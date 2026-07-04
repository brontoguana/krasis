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

## Config Files

The preferred way to run Krasis is with a config file:

```bash
krasis --config path/to/config.conf
./dev run qcn            # resolves to testconfigs/qcn.conf
./dev benchmark qcn      # same
```

Config files use `KEY=VALUE` format. CLI flags override config file values.

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
