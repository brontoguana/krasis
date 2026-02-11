# Krasis

Rust + PyO3 MoE runtime for large mixture-of-experts LLMs. Replaces the KTransformers C++ LLAMAFILE backend with AVX2 INT4/INT8 kernels, NUMA-aware expert dispatch, and unified weight format.

## What It Does

Runs 350B+ parameter MoE models (Kimi K2.5, Qwen3-235B, DeepSeek V2) on commodity hardware: 3x 16GB GPUs + 1TB system RAM. GPUs handle attention, norms, and routing; Krasis handles the expensive MoE expert computation on CPU.

## Architecture

```
SGLang (Python)
├── HTTP API, batching, scheduling
├── Attention (FlashInfer MLA/GQA), norms, routing
├── GPU prefill: INT4 Marlin fused_marlin_moe kernel (large prompts)
└── CPU decode: Krasis AVX2 INT4 kernel (token-by-token)

Krasis (Rust + PyO3)
├── ONE weight format: GPU-native Marlin INT4 (disk = RAM = GPU, no conversion)
├── First run: BF16 safetensors → INT4 quantize → Marlin repack → disk cache
├── Every run: load Marlin cache → RAM. GPU DMA copies directly. CPU reads directly.
├── AVX2 Marlin-native CPU kernel for decode
├── Expert-level + intra-expert parallelism (rayon)
├── NUMA-aware weight placement + thread pinning
├── Zero-allocation scratch pool, NTA prefetch
└── PyO3 bridge: KrasisEngine with async submit/sync
```

## Supported Models

| Model | Experts | Status |
|-------|---------|--------|
| **Kimi K2.5** | 384 routed + 1 shared, top-8 | Working (SGLang PP=3, ~1.25 tok/s decode) |
| **Qwen3-235B-A22B** | 128 routed, top-8 | Working (KTransformers PP=3, 4.21 tok/s) |
| **DeepSeek V2-Lite** | 64 routed + 2 shared, top-6 | Working (standalone, 3.3 tok/s) |

## Hardware Requirements

- **CPU**: x86-64 with AVX2+FMA (AMD EPYC, Intel Xeon). AVX512 not required.
- **GPU**: NVIDIA with compute capability 8.0+ (Ampere/Ada). 16GB+ VRAM per GPU.
- **RAM**: ~500-600 GB for 350B+ models (3 PP ranks × ~190 GB each).

## Building

```bash
# Build Rust library
cargo build --release

# Install Python package (into your venv)
pip install -e .

# Or build + install manually
CPUINFER_CPU_INSTRUCT=NATIVE ./install.sh build --manual
```

## Usage

### With SGLang (recommended)

```bash
# Set environment
export KRASIS_BACKEND=1
export KRASIS_MODEL_PATH=/path/to/model

# Launch SGLang with Krasis MoE backend
python -m sglang.launch_server \
    --model /path/to/model \
    --kt-cpuinfer 21 \
    --pp-size 3 \
    --quantization w8a8_int8 \
    --kv-cache-dtype fp8_e4m3 \
    --disable-cuda-graph
```

See `run_kimi_krasis.sh` for a complete launch script.

## Key Features

- **GPU-native Marlin INT4 — the ONLY weight format** — same bytes on disk, in RAM, on GPU. No conversion anywhere. CPU and GPU both read the same Marlin-packed data.
- **Streaming cache build** — first-run quantization streams one expert at a time (~128 MB peak)
- **Zero-conversion GPU prefill** — DMA copy Marlin weights from RAM to GPU, run fused_marlin_moe instantly
- **FP8 KV cache** — 2x VRAM savings with negligible precision loss
- **INT8 non-expert weights** — halves attention VRAM via per-channel quantization
- **skip_shared_experts** — prevents double computation when host handles shared experts on GPU

## Documentation

- [STATUS.md](STATUS.md) — Feature tracking and performance results
- [CHANGELOG.md](CHANGELOG.md) — Detailed change history with test results
- [DESIGN.md](DESIGN.md) — Architecture design document

## License

Apache 2.0
