# Krasis

Hybrid LLM runtime — minimal VRAM, always-on GPU prefill, optimised CPU inference.

Krasis replaces the KTransformers C++ backend with a single Rust `.so` that serves both GPU and CPU from one unified Marlin INT4 weight format in system RAM. No GGUF, no dual-format weight duplication.

## Architecture

- **SGLang** handles HTTP API, batching, scheduling, attention (FlashInfer), KV cache
- **Krasis** handles MoE expert computation: GPU prefill via `fused_marlin_moe` CUDA kernel, CPU decode via native AVX2 INT4 kernel
- Single Marlin INT4 format in RAM, mmap'd — GPU reads directly, CPU un-permutes activation vector (14KB, fits in L1) and scans weights sequentially
- NUMA-aware expert placement and thread dispatch

## Status

Early development. Testing against DeepSeek-V2-Lite (15.7B, same DeepSeekV2 architecture as Kimi K2.5).

## Building

```bash
cargo build --release
```

## License

MIT
