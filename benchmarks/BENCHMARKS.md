# Krasis Benchmark Results

## Suite Run — 2026-02-20 (10K prompt, PP=1, HCS mode)

Config: 10K token prompt, FP8 KV cache, INT8 attention/shared_expert/dense_mlp/lm_head, 48 CPU threads.
All configs use PP=1 (Expert Parallelism for multi-GPU, not Pipeline Parallelism).

| Model | Config | Prefill (tok/s) | TTFT (s) | Decode (tok/s) | ms/tok | Status |
|-------|--------|----------------:|----------:|---------------:|-------:|--------|
| DeepSeek-V2-Lite | 1gpu INT8/INT8 | 1882.8 | 5.32 | 3.04 | 328.8 | PASS |
| DeepSeek-V2-Lite | 2gpu INT4/INT4 | 1623.1 | 6.16 | 6.22 | 160.9 | PASS |
| Qwen3-Coder-Next | 1gpu INT8/INT8 | 696.4 | 14.36 | 5.93 | 168.6 | PASS |
| Qwen3-Coder-Next | 2gpu INT4/INT4 | 819.5 | 12.20 | 7.94 | 125.9 | PASS |
| gpt-oss-120b | 1gpu INT8/INT8 | 516.1 | 19.38 | 3.59 | 278.7 | PASS |
| gpt-oss-120b | 2gpu INT4/INT4 | 825.7 | 12.11 | 5.17 | 193.6 | PASS |
| Qwen3-235B-A22B | 1gpu INT8/INT8 | - | - | - | - | OOM |
| Qwen3-235B-A22B | 2gpu INT4/INT4 | - | - | - | - | OOM |

Full logs: [suite_logs/](suite_logs/)

### Notes

- **Qwen3-235B-A22B OOM**: 94 MLA attention layers exceed 16GB VRAM on a single GPU with PP=1. Previously ran with PP=2/PP=3 (layers split across GPUs). Would need 3+ GPUs or larger VRAM to run with PP=1.
- **gpt-oss-120b**: First successful benchmark run after Marlin w2 padding fix and CUDA graph dispatch fix.
- **QCN 2gpu 819.5 tok/s**: Consistent with previous 849 tok/s baseline (10K prompt, PP=1, HCS mode).
