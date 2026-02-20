# Krasis Benchmark Results

## Standard Benchmarks — 2026-02-20 (10K prompt, HCS mode)

Config: 10K token prompt, FP8 KV cache, INT8 attention/shared_expert/dense_mlp/lm_head, 48 CPU threads.
All configs use EP (Expert Parallelism for multi-GPU, not Pipeline Parallelism).

| Model | Config | Prefill (tok/s) | TTFT (s) | Decode (tok/s) | ms/tok | Status | Log |
|-------|--------|----------------:|----------:|---------------:|-------:|--------|-----|
| DeepSeek-V2-Lite | 1gpu INT8/INT8 | 1882.8 | 5.32 | 3.04 | 328.8 | PASS | [log](suite_logs/) |
| DeepSeek-V2-Lite | 2gpu INT4/INT4 | 1623.1 | 6.16 | 6.22 | 160.9 | PASS | [log](suite_logs/) |
| Qwen3-Coder-Next | 1gpu INT8/INT8 | 696.4 | 14.36 | 5.93 | 168.6 | PASS | [log](suite_logs/) |
| Qwen3-Coder-Next | 1gpu INT4/INT4 | 979.6 | 10.21 | 7.89 | 126.8 | PASS | [log](Qwen3-Coder-Next_native_1gpu_int4gpu_int4cpu.log) |
| Qwen3-Coder-Next | 2gpu INT4/INT4 | 880.2 | 11.36 | 8.15 | 122.8 | FAIL* | [log](Qwen3-Coder-Next_native_2gpu_int4gpu_int4cpu.log) |
| gpt-oss-120b | 1gpu INT8/INT8 | 516.1 | 19.38 | 3.59 | 278.7 | PASS | [log](suite_logs/) |
| gpt-oss-120b | 2gpu INT4/INT4 | 825.7 | 12.11 | 5.17 | 193.6 | PASS | [log](suite_logs/) |
| Qwen3-235B-A22B | 1gpu INT8/INT8 | - | - | - | - | OOM | - |
| Qwen3-235B-A22B | 2gpu INT4/INT4 | - | - | - | - | OOM | - |

### Notes

- **Qwen3-235B-A22B OOM**: 94 MLA attention layers exceed 16GB VRAM on a single GPU with PP=1. Would need 3+ GPUs or larger VRAM.
- **gpt-oss-120b**: First successful benchmark run after Marlin w2 padding fix and CUDA graph dispatch fix.
- **QCN 1gpu INT4/INT4 979.6 tok/s**: Fastest prefill — all experts + attention on one GPU, no cross-GPU DMA.
- **QCN 2gpu INT4/INT4 FAIL***: Prefill and decode speeds are valid, but decode output is garbage (cross-GPU HCS expert corruption). P2P bounce fix resolved the OOM crash but not the data corruption.
