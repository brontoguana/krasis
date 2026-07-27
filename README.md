# Krasis

Krasis is an LLM runtime for running large MoE models on NVIDIA consumer GPUs.
It is built around fast GPU prompt processing, GPU-executed decode, and HCS
expert residency management so models much larger than VRAM can still run
locally.

The current runtime is no longer the early Python-hot-path prototype. The
serving path is Rust/CUDA focused: Python is used for launcher/setup/model
loading work, while the performance-sensitive runtime path uses Rust/CUDA
orchestration, CUDA kernels, cached quantized weights, and measured VRAM
budgeting.

## Install

The current release is
[`v1.0.16`](https://github.com/brontoguana/krasis/releases/tag/v1.0.16).

**Native Windows:** [Download the Krasis Windows installer](https://github.com/brontoguana/krasis/releases/download/v1.0.16/KrasisSetup-1.0.16-win64.exe).
It installs Krasis for the current user, includes its own Python runtime, and
adds `Krasis` to the Start Menu.

**Linux or WSL2:**

```bash
curl -sSf https://raw.githubusercontent.com/brontoguana/krasis/main/install.sh | bash -s -- prerelease
```

See [all releases](https://github.com/brontoguana/krasis/releases) for older
versions and individual wheel/source assets.

You can [contact me here](https://forms.gle/ue4nvyvNNHtUZ7MQ7), but for bugs,
setup problems, model requests, or feature requests please open a GitHub issue.

If you want to monitor Krasis during runs, [check out ktop](https://github.com/brontoguana/ktop).

![Krasis Server](krasis_server_3.png)

## What Krasis Does

- Runs multi-hundred-billion-parameter MoE models from BF16 safetensors on
  commodity NVIDIA GPU systems.
- Uses full GPU prefill for fast prompt processing.
- Uses GPU-executed decode with HCS managing hot/cold expert residency between
  VRAM and CPU RAM.
- Builds cached INT4/INT8 expert formats and HQQ attention caches under
  `~/.krasis`.
- Supports compact KV cache modes including `k6v6` Quality and `k4v4` Ultra
  Compact.
- Provides an interactive launcher, OpenAI-compatible API, chat client,
  reproducible benchmarks, and GitHub-release based installation.

## Major Changes Since The Previous Stable Release

The current release line is a major change from `v0.1.64`, the previous stable
Krasis release. Highlights:

- Runtime hot-path work moved out of Python and into Rust/CUDA for serving,
  decode orchestration, timing, HCS operations, and benchmark-critical paths.
- Added HQQ attention support, including HQQ4, HQQ6, HQQ8, auto mixed profiles,
  cache build/rebuild support, and HQQ benchmark/validation lanes.
- Added compact KV cache formats, including `k6v6` and `k4v4`, with `k6v6`
  as the quality-oriented launcher default and `k4v4` for tighter VRAM budgets.
- Added full Ampere support for the current production path. HQQ attention and
  compact KV cache modes were built with Ampere compatibility in mind and do
  not require FP8-capable hardware.
- Expanded validated model coverage across Qwen3-Coder-Next, Qwen3/3.5/3.6,
  Ornith, Step-3.7-Flash, Gemma 4, and Nemotron MoE families.
- Added and hardened HCS expert residency management: measured startup
  calibration, prompt-conditioned reload, dynamic recency tail, per-stage
  budgets, soft-tier reload caps, and safe eviction/reload paths.
- Added runtime VRAM safety systems: short/long prefill/decode calibration,
  measured scratch budgets, pressure detection, idle pressure drain, and hard
  exit protection before CUDA enters an unsafe OOM state.
- Added full GitHub release wheel packaging for Python 3.10, 3.11, 3.12, and
  3.13, with vendored CUDA sidecars injected into wheels.
- Added a native Windows installer with a private Python runtime, interactive
  launcher, and a maximized Start Menu entry.
- Added `krasis update` and `krasis prerelease` maintenance commands.
- Added an interactive curated Hugging Face downloader flow in the launcher.
- Added reverse SSH tunnel support for exposing a local Krasis server to a
  remote machine through SSH without opening public ports.
- Added repeatable benchmark and release-test commands, benchmark log archival,
  llama-witness based correctness validation, and richer diagnostics.
- Removed Session messenger integration and other stale prototype-era surfaces.
- Cleaned terminal/log output so human console lines are clean while prefixed
  records go to log files.
- Deprecated AWQ and Polar4 for new production runs. Current production
  surfaces use HQQ attention plus `k6v6`, `k4v4`, or BF16 KV depending on the
  memory/quality target.

## Benchmarks

Selected current timing-disabled results. `Decode` is the internal engine
measurement; `HTTP round trip` includes local client/server HTTP overhead.

| Hardware | Model | Params | Attention + KV | Prefill | Decode | HTTP round trip |
|---|---|---:|---|---:|---:|---:|
| RTX PRO 6000 96 GB | Step-3.7-Flash | 201.4B | INT4/HQQ4/k4v4 | 5,261.0 tok/s | 55.40 tok/s | 112.82 tok/s |
| RTX PRO 6000 96 GB | Ornith-1.0-397B | 397B | INT4/HQQ4/k4v4 | 2,354.5 tok/s | 23.58 tok/s | 41.73 tok/s |
| RTX PRO 6000 96 GB | Qwen3-Coder-Next | 80B | INT4/HQQ4/k4v4 | 11,211.1 tok/s | 91.34 tok/s | 161.82 tok/s |
| RTX 5090 32 GB | Nemotron-3-Super-120B-A12B | 123.6B | INT4/HQQ4/k4v4 | 1,852.2 tok/s | 41.87 tok/s | 50.76 tok/s |
| RTX 5090 32 GB | Qwen3.5-397B-A17B | 397B | INT4/HQQ4/k4v4 | 973.8 tok/s | 10.04 tok/s | 18.71 tok/s |
| RTX 5090 32 GB | Nemotron-3-Nano-30B-A3B | 31.6B | INT4/HQQ4/k4v4 | 8,583.9 tok/s | 151.76 tok/s | 325.36 tok/s |

See the complete [benchmark table](STATS-BENCHMARKS.md), the associated
[quality results](STATS-QUALITY.md), and the
[reproducible benchmark index](benchmarks/BENCHMARKS.md). Approximate
adaptive-pruning results are not used in this table.

## Tradeoffs And Requirements

- Krasis currently targets NVIDIA GPUs with CUDA, including Ampere and newer
  architectures. The production HQQ attention and compact KV cache modes do not
  require FP8 support.
- Input models should be BF16 safetensors from Hugging Face or another local
  safetensors source.
- First run is slower because Krasis builds optimized local caches. Later runs
  reuse those caches.
- Disk usage must cover the source model plus Krasis cache artifacts under
  `~/.krasis`.
- System RAM should be sized for the selected quantized cache and HCS backing
  store. Larger models need substantial RAM even when GPU VRAM is limited.
- Production runs should use quantized INT4/INT8 expert caches and HQQ
  attention. BF16-heavy modes are validation/debug modes, not normal deployment
  targets.

## Quick Start

### Requirements

- Native x86-64 Windows, Linux (including Ubuntu 24.04+), or WSL2
- Python 3.10+ on Linux/WSL; native Windows uses the release-pinned private
  Python included by the installer
- NVIDIA GPU with CUDA drivers installed
- Rust is only needed for source builds, not normal wheel installs
- Enough disk/RAM for the source model and generated Krasis caches

### 1. Install Krasis

Linux/WSL:

```bash
curl -sSf https://raw.githubusercontent.com/brontoguana/krasis/main/install.sh | bash -s -- prerelease
```

This creates a managed environment at `~/.krasis/venv`, installs Krasis,
symlinks commands into `~/.local/bin`, and updates PATH for the current shell.
No sudo is required for the Krasis install itself. Omit `prerelease` when
installing the latest stable release.

Native Windows:

[Download `KrasisSetup-1.0.16-win64.exe`](https://github.com/brontoguana/krasis/releases/download/v1.0.16/KrasisSetup-1.0.16-win64.exe).
The installer creates a per-user install under
`%LOCALAPPDATA%\Programs\Krasis`, installs and validates a release-pinned
private Python/Krasis/PyTorch runtime, and adds `Krasis` to the Start Menu
folder. It never uses or modifies a system Python. `Krasis` opens the native
interactive launcher in a maximized, resizable console. Models and caches
still live under
`%USERPROFILE%\.krasis`. The first install downloads the pinned CUDA PyTorch
wheel and can take several minutes.

Native Windows packages Marlin, FlashAttention, and FLA sidecars for supported
Ampere and newer NVIDIA architectures.

### 2. Install CUDA Dependencies

```bash
krasis-setup
```

This installs runtime CUDA/PyTorch dependencies when needed. It is usually only
required once per machine.

### 3. Download A Model

Run:

```bash
krasis
```

Then use the interactive launcher to choose from Krasis-supported Hugging Face
models, or put BF16 safetensors manually under `~/.krasis/models/`.

Manual download example:

```bash
huggingface-cli download Qwen/Qwen3-Coder-Next \
    --local-dir ~/.krasis/models/Qwen3-Coder-Next
```

### 4. Run

```bash
krasis
```

The launcher walks through model selection, GPU selection, quantization/runtime
options, and server startup. Settings are saved under `~/.krasis/config`.

## Updating

On Linux or WSL:

```bash
# Latest stable release
krasis update

# Latest pre-release
krasis prerelease

# Uninstall Krasis, keeping model files
curl -sSf https://raw.githubusercontent.com/brontoguana/krasis/main/install.sh | bash -s -- --uninstall
```

On native Windows, download and run the desired stable or prerelease installer
from the [Krasis releases page](https://github.com/brontoguana/krasis/releases).

## WSL2

Krasis works on WSL2. By default WSL often limits available memory, which is
usually too small for large MoE models. Create or edit:

```text
C:\Users\<YourUsername>\.wslconfig
```

Example:

```ini
[wsl2]
memory=120GB
```

Adjust the value to leave memory for Windows, then restart WSL from PowerShell:

```powershell
wsl --shutdown
```

## Usage

### Interactive Launcher

```bash
krasis
```

The launcher provides:

- model selection from local models
- curated Hugging Face model download for supported models
- GPU selection, including selected GPU indices
- quantization, HQQ attention, KV cache, HCS, and VRAM safety settings
- optional reverse SSH tunnel target
- benchmark/run choices

### Non-Interactive Launch

```bash
# Use saved config
krasis --non-interactive

# Use a config file
krasis --config tests/qcn-k4v4-hqq8-int4-benchmark.conf

# Override selected values
krasis --non-interactive --model-path /path/to/model --selected-gpus 0,2 --benchmark
```

Common options:

- `--attention-quant hqq6` or `hqq8`
- `--kv-dtype k6v6`, `k4v4`, or `bf16`
- `--gpu-expert-bits 4` or `8`
- `--vram-safety-margin 600`
- `--dynamic-hcs` / `--no-dynamic-hcs`
- `--ssh-tunnel user@host`
- `--ssh-key-path ~/.ssh/id_ed25519`

For the full option surface, run:

```bash
krasis --help
```

### Chat Client

```bash
krasis chat
krasis chat --prompt "Explain HCS in one paragraph"
krasis chat --file prompts.txt
krasis chat --port 8013
krasis chat --url http://host:8012
```

The standalone command also remains available:

```bash
krasis-chat
```

### API

Krasis exposes an OpenAI-compatible chat endpoint:

```text
http://localhost:8012/v1/chat/completions
```

Useful endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/timing`

### Benchmarks

Use the fixed speed-regression entry point for repeatable Qwen3-Coder-Next
speed checks:

```bash
./dev speed-test
```

Run a standard benchmark for a config:

```bash
./dev benchmark tests/qcn-k4v4-hqq8-int4-benchmark.conf
```

Run a benchmark from the installed command:

```bash
krasis --config tests/qcn-k4v4-hqq8-int4-benchmark.conf --benchmark
```

### Source Build

For development builds:

```bash
git clone https://github.com/brontoguana/krasis.git
cd krasis
./dev build
./dev run qcn
```

The `./dev` entry point handles environment setup and is preferred for local
development commands.

## Advanced Documentation

See [ADVANCED.md](ADVANCED.md) for detailed config options, quantization modes,
HQQ cache controls, HCS controls, benchmarking commands, and API details.

## License

SSPL-1.0

Krasis is free to use, modify, and distribute.

If you want to support the project or offer Krasis as part of a commercial
product or a hosted/managed service, please [get in touch](https://forms.gle/ue4nvyvNNHtUZ7MQ7).
