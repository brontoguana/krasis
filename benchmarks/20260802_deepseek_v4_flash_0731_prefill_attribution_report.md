# DeepSeek-V4-Flash-0731 prefill attribution

Command surface: one `KRASIS_PREFILL_TIMING=1 ./dev run` server using the
required approved heatmap, exercised by `./dev network 18220 --large` with the
canonical Gutenberg prompts. This is an instrumented component run, not speed
benchmark evidence. CUDA-event stage totals are the attribution source; the
timing-disabled adjacent baseline remains the speed source.

| Prompt tokens | Total event time | MoE | Sparse scores | Sparse output | Indexer | Remaining stages |
|---:|---:|---:|---:|---:|---:|---:|
| 2,043 | 11,372.2 ms | 5,022.4 (44.2%) | 4,341.7 (38.2%) | 1,651.6 (14.5%) | 114.9 (1.0%) | 241.6 (2.1%) |
| 8,623 | 42,958.1 ms | 6,823.4 (15.9%) | 23,699.6 (55.2%) | 9,709.1 (22.6%) | 1,331.0 (3.1%) | 1,394.9 (3.2%) |
| 23,348 | 121,583.8 ms | 7,826.0 (6.4%) | 72,700.4 (59.8%) | 28,660.9 (23.6%) | 8,489.3 (7.0%) | 3,907.2 (3.2%) |
| 62,403 | 428,186.8 ms | 15,347.4 (3.6%) | 234,047.1 (54.7%) | 111,918.1 (26.1%) | 56,409.9 (13.2%) | 10,464.3 (2.4%) |

The fourteen stage events account for the reported prefill total within 13 ms
at every prompt size. Stage calls are 43 for one-chunk prompts and 86 for the
62,403-token two-chunk prompt, matching the config-derived 43-layer main stack.
Sparse scoring is the largest long-prompt component and consumes 54.7-59.8% of
prefill event time at 8,623-62,403 tokens, so it passes the component threshold
for the first prefill optimization attempt.

All four large-prompt requests passed. The instrumented 62,403-token row
reported 650 MiB minimum free VRAM versus the 600 MiB safety margin. One
stochastic multi-turn generation check failed to repeat `42`, then the next
turn correctly recalled all three stored facts; this is retained in the raw
network log and is not used as component evidence.

Measurement-integrity note: the legacy generic timing footer prints a
hardcoded `48 layers` label and zero generic MoE percentages for the V4 branch.
Those legacy labels are invalid for this 43-layer model. The new V4 stage table
reports runtime call counts and valid CUDA-event totals; the legacy label will
be corrected before the next profiling run.
