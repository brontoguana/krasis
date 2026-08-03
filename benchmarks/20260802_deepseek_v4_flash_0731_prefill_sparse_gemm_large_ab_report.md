# DeepSeek-V4-Flash-0731 gathered sparse-score GEMM large-prompt A/B

Both servers used the same source, 0731 config, approved 80-prompt heatmap, HCS population, and deployment state. The candidate differed only by `KRASIS_PREFILL_DEEPSEEK_V4_GATHERED_SPARSE_SCORES=1`. Timing instrumentation was disabled.

| Prompt tokens | Control prefill | Candidate prefill | Delta | Control seconds | Candidate seconds | Candidate min free VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| 2,043 | 166.7 tok/s | 254.3 tok/s | +52.5% | 12.25 | 8.04 | 650 MiB |
| 8,623 | 201.5 tok/s | 441.6 tok/s | +119.2% | 42.80 | 19.53 | 578 MiB |
| 23,348 | 191.9 tok/s | 465.1 tok/s | +142.4% | 121.66 | 50.20 | 650 MiB |
| 62,403 | 145.5 tok/s | 313.4 tok/s | +115.4% | 429.02 | 199.12 | 650 MiB |

All 18 network, canonical large-prompt, and multi-turn checks passed in both members. The candidate is not yet accepted: the first 8,623-token request triggered a one-time 578 MiB low-water, 22 MiB below the required 600 MiB safety margin. Later larger requests returned to 650 MiB, localizing the issue to lazy first-use workspace/allocation rather than steady-state prompt scaling. The runtime allocation cause must be fixed and retested before quality gates or promotion.
