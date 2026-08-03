# DeepSeek-V4-Flash-0731 gathered sparse-score GEMM component gate

- Date: 2026-08-02
- Model/config: `DeepSeek-V4-Flash-0731`, `tests/deepseek-v4-flash-0731-bf16.conf`
- Candidate: `KRASIS_PREFILL_DEEPSEEK_V4_GATHERED_SPARSE_SCORES=1`
- Instrumentation: `KRASIS_PREFILL_TIMING=1`; these are component diagnostics, not speed-benchmark results.
- Approved heatmap: required and loaded from the 2026-08-02 0731 local manifest.

At 1,000 input tokens, the gathered BF16-input/FP32-output strided-batched GEMM measured 29.8-30.1 ms of sparse-score CUDA-event time across all 43 layers. The original score kernel measured 4,341.7 ms at 2,043 tokens in the preceding attribution run and was the dominant prefill stage. The candidate therefore passes the component gate by a wide margin; the timing-disabled adjacent A/B remains the end-to-end acceptance gate.

The candidate retained 6,440/11,008 HCS experts, reported 1,136 MiB minimum free VRAM, completed all standard benchmark requests, and released the RTX 6000 fully after teardown. The A4500 service remained HTTP 200.
