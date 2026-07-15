# Krasis Benchmark Stats

Benchmarks use `./dev benchmark` with Krasis timing instrumentation disabled. Peak
system RAM is recorded with external `/usr/bin/time -v` around the benchmark
command, so it is a process max-RSS measurement. HTTP round trip is the
client-side HTTP path and should not be confused with internal decode speed.

Initial scope: measured supported models at `INT4/HQQ4/k4v4` and
`INT4/HQQ6/k6v6`.

| Hardware | Model | Params | Active params | Attention + KV | Prefill | Decode | HTTP round trip | Peak system RAM | HCS | Min free VRAM | Logs |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1x RTX 5090 32GB, AMD EPYC 7742 | Qwen3.6-35B-A3B | 35.5B text | 3.0B | INT4/HQQ4/k4v4 | 10,670.1 tok/s | 117.20 tok/s | 241.78 tok/s | 23.5 GB max RSS (22.4 GiB) | 10240/10240 (100.0%) | 10,186 MB | [report](benchmarks/20260714_qwen36_hqq4_k4v4_mmap_evict_benchmark_report.log), [stdout](benchmarks/20260714_qwen36_hqq4_k4v4_mmap_evict_benchmark_stdout.log), [peak](benchmarks/20260714_qwen36_step37_mmap_evict_rerun_peak_benchmark_hqq4_hqq6.log) |
| 1x RTX 5090 32GB, AMD EPYC 7742 | Qwen3.6-35B-A3B | 35.5B text | 3.0B | INT4/HQQ6/k6v6 | 9,693.6 tok/s | 115.50 tok/s | 234.29 tok/s | 23.7 GB max RSS (22.6 GiB) | 10240/10240 (100.0%) | 9,882 MB | [report](benchmarks/20260714_qwen36_hqq6_k6v6_mmap_evict_benchmark_report.log), [stdout](benchmarks/20260714_qwen36_hqq6_k6v6_mmap_evict_benchmark_stdout.log), [peak](benchmarks/20260714_qwen36_step37_mmap_evict_rerun_peak_benchmark_hqq4_hqq6.log) |
| 1x RTX 5090 32GB, AMD EPYC 7742 | Qwen3-Coder-Next | 80B class | n/a | INT4/HQQ4/k4v4 | 7,028.5 tok/s | 87.90 tok/s | 150.75 tok/s | 44.1 GB max RSS (42.0 GiB) | 16362/24576 (66.6%) | 878 MB | [report](benchmarks/20260715_qcn_hqq4_k4v4_stats_benchmark_report.log), [stdout](benchmarks/20260715_qcn_hqq4_k4v4_stats_benchmark_stdout.log), [peak](benchmarks/20260715_remaining_stats_benchmark_hqq4_hqq6.log) |
| 1x RTX 5090 32GB, AMD EPYC 7742 | Qwen3-Coder-Next | 80B class | n/a | INT4/HQQ6/k6v6 | 6,628.4 tok/s | 88.22 tok/s | 201.94 tok/s | 45.8 GB max RSS (43.7 GiB) | 16119/24576 (65.6%) | 856 MB | [report](benchmarks/20260715_qcn_hqq6_k6v6_stats_benchmark_report.log), [stdout](benchmarks/20260715_qcn_hqq6_k6v6_stats_benchmark_stdout.log), [peak](benchmarks/20260715_remaining_stats_benchmark_hqq4_hqq6.log) |
| 1x RTX 5090 32GB, AMD EPYC 7742 | Qwen3.5-35B-A3B | 35B class | 3B class | INT4/HQQ4/k4v4 | 10,700.7 tok/s | 116.43 tok/s | 259.67 tok/s | 23.8 GB max RSS (22.7 GiB) | 10240/10240 (100.0%) | 10,186 MB | [report](benchmarks/20260715_q35b_hqq4_k4v4_stats_benchmark_report.log), [stdout](benchmarks/20260715_q35b_hqq4_k4v4_stats_benchmark_stdout.log), [peak](benchmarks/20260715_remaining_stats_benchmark_hqq4_hqq6.log) |
| 1x RTX 5090 32GB, AMD EPYC 7742 | Qwen3.5-35B-A3B | 35B class | 3B class | INT4/HQQ6/k6v6 | 9,314.5 tok/s | 113.61 tok/s | 225.17 tok/s | 23.9 GB max RSS (22.8 GiB) | 10240/10240 (100.0%) | 9,882 MB | [report](benchmarks/20260715_q35b_hqq6_k6v6_stats_benchmark_report.log), [stdout](benchmarks/20260715_q35b_hqq6_k6v6_stats_benchmark_stdout.log), [peak](benchmarks/20260715_remaining_stats_benchmark_hqq4_hqq6.log) |
| 1x RTX 5090 32GB, AMD EPYC 7742 | Qwen3.5-122B-A10B | 122B class | 10B class | INT4/HQQ4/k4v4 | 3,692.5 tok/s | 31.80 tok/s | 53.15 tok/s | 78.2 GB max RSS (74.6 GiB) | 4428/12288 (36.0%) | 856 MB | [report](benchmarks/20260715_q122b_hqq4_k4v4_stats_benchmark_report.log), [stdout](benchmarks/20260715_q122b_hqq4_k4v4_stats_benchmark_stdout.log), [peak](benchmarks/20260715_remaining_stats_benchmark_hqq4_hqq6.log) |
| 1x RTX 5090 32GB, AMD EPYC 7742 | Qwen3.5-122B-A10B | 122B class | 10B class | INT4/HQQ6/k6v6 | 3,389.2 tok/s | 28.83 tok/s | 53.45 tok/s | 82.9 GB max RSS (79.1 GiB) | 4185/12288 (34.1%) | 862 MB | [report](benchmarks/20260715_q122b_hqq6_k6v6_stats_benchmark_report.log), [stdout](benchmarks/20260715_q122b_hqq6_k6v6_stats_benchmark_stdout.log), [peak](benchmarks/20260715_remaining_stats_benchmark_hqq4_hqq6.log) |
| 1x RTX 5090 32GB, AMD EPYC 7742 | Qwen3-235B-A22B | 235B class | 22B class | INT4/HQQ4/k4v4 | 1,385.3 tok/s | 5.64 tok/s | 9.91 tok/s | 134.2 GB max RSS (127.9 GiB) | 2080/12032 (17.3%) | 946 MB | [report](benchmarks/20260715_q235_hqq4_k4v4_stats_benchmark_report.log), [stdout](benchmarks/20260715_q235_hqq4_k4v4_stats_benchmark_stdout.log), [peak](benchmarks/20260715_remaining_stats_benchmark_hqq4_hqq6.log) |
| 1x RTX 5090 32GB, AMD EPYC 7742 | Qwen3-235B-A22B | 235B class | 22B class | INT4/HQQ6/k6v6 | 768.8 tok/s | 5.16 tok/s | 9.38 tok/s | 144.3 GB max RSS (137.6 GiB) | 1820/12032 (15.1%) | 936 MB | [report](benchmarks/20260715_q235_hqq6_k6v6_stats_benchmark_report.log), [stdout](benchmarks/20260715_q235_hqq6_k6v6_stats_benchmark_stdout.log), [peak](benchmarks/20260715_remaining_stats_benchmark_hqq4_hqq6.log) |
| 1x RTX 5090 32GB, AMD EPYC 7742 | Step-3.7-Flash | 201.4B total / 199.4B text | 13.9B | INT4/HQQ4/k4v4 | 2,618.4 tok/s | 23.05 tok/s | 36.11 tok/s | 108.4 GB max RSS (103.4 GiB) | 2784/12096 (23.0%) | 884 MB | [report](benchmarks/20260714_step37_hqq4_k4v4_mmap_evict_benchmark_report.log), [stdout](benchmarks/20260714_step37_hqq4_k4v4_mmap_evict_benchmark_stdout.log), [peak](benchmarks/20260714_qwen36_step37_mmap_evict_rerun_peak_benchmark_hqq4_hqq6.log) |
| 1x RTX 5090 32GB, AMD EPYC 7742 | Step-3.7-Flash | 201.4B total / 199.4B text | 13.9B | INT4/HQQ6/k6v6 | 1,874.5 tok/s | 20.93 tok/s | 32.82 tok/s | 112.8 GB max RSS (107.5 GiB) | 2624/12096 (21.7%) | 956 MB | [report](benchmarks/20260714_step37_hqq6_k6v6_mmap_evict_benchmark_report.log), [stdout](benchmarks/20260714_step37_hqq6_k6v6_mmap_evict_benchmark_stdout.log), [peak](benchmarks/20260714_qwen36_step37_mmap_evict_rerun_peak_benchmark_hqq4_hqq6.log) |
| 1x RTX 5090 32GB, AMD EPYC 7742 | Gemma-4-26B-A4B-it | 26B class | 4B class | INT4/HQQ4/k4v4 | 5,598.9 tok/s | 63.57 tok/s | 119.90 tok/s | 21.3 GB max RSS (20.3 GiB) | 3840/3840 (100.0%) | 12,084 MB | [report](benchmarks/20260715_gemma4_hqq4_k4v4_cached_stats_benchmark_report.log), [stdout](benchmarks/20260715_gemma4_hqq4_k4v4_cached_stats_benchmark_stdout.log), [peak](benchmarks/20260715_gemma4_cached_rerun_benchmark_hqq4_hqq6.log) |
| 1x RTX 5090 32GB, AMD EPYC 7742 | Gemma-4-26B-A4B-it | 26B class | 4B class | INT4/HQQ6/k6v6 | 1,610.0 tok/s | 62.16 tok/s | 114.42 tok/s | 21.3 GB max RSS (20.3 GiB) | 3840/3840 (100.0%) | 11,788 MB | [report](benchmarks/20260715_gemma4_hqq6_k6v6_cached_stats_benchmark_report.log), [stdout](benchmarks/20260715_gemma4_hqq6_k6v6_cached_stats_benchmark_stdout.log), [peak](benchmarks/20260715_gemma4_cached_rerun_benchmark_hqq4_hqq6.log) |
| 1x RTX 5090 32GB, AMD EPYC 7742 | Nemotron-3-Super-120B-A12B | 123.6B total | 12.4B | INT4/HQQ4/k4v4 | 1,852.2 tok/s | 41.87 tok/s | 50.76 tok/s | 65.7 GB max RSS (62.7 GiB) | 7038/20480 (34.4%) | 906 MB | [report](benchmarks/20260714_nemotron_super_hqq4_k4v4_local_heatmap_mmap_evict_benchmark_report.log), [stdout](benchmarks/20260714_nemotron_super_hqq4_k4v4_local_heatmap_mmap_evict_benchmark_stdout.log), [peak](benchmarks/20260714_nemotron_super_nano_local_heatmap_mmap_evict_rerun_peak_benchmark_hqq4_hqq6.log) |
| 1x RTX 5090 32GB, AMD EPYC 7742 | Nemotron-3-Super-120B-A12B | 123.6B total | 12.4B | INT4/HQQ6/k6v6 | 1,899.8 tok/s | 30.47 tok/s | 50.96 tok/s | 63.1 GB max RSS (60.2 GiB) | 4922/20480 (24.0%) | 894 MB | [report](benchmarks/20260714_nemotron_super_hqq6_k6v6_local_heatmap_mmap_evict_benchmark_report.log), [stdout](benchmarks/20260714_nemotron_super_hqq6_k6v6_local_heatmap_mmap_evict_benchmark_stdout.log), [peak](benchmarks/20260714_nemotron_super_nano_local_heatmap_mmap_evict_rerun_peak_benchmark_hqq4_hqq6.log) |
| 1x RTX 5090 32GB, AMD EPYC 7742 | Nemotron-3-Nano-30B-A3B | 31.6B total | 3.2B | INT4/HQQ4/k4v4 | 8,583.9 tok/s | 151.76 tok/s | 325.36 tok/s | 19.4 GB max RSS (18.5 GiB) | 2944/2944 (100.0%) | 11,722 MB | [report](benchmarks/20260714_nemotron_nano_hqq4_k4v4_local_heatmap_mmap_evict_benchmark_report.log), [stdout](benchmarks/20260714_nemotron_nano_hqq4_k4v4_local_heatmap_mmap_evict_benchmark_stdout.log), [peak](benchmarks/20260714_nemotron_super_nano_local_heatmap_mmap_evict_rerun_peak_benchmark_hqq4_hqq6.log) |
| 1x RTX 5090 32GB, AMD EPYC 7742 | Nemotron-3-Nano-30B-A3B | 31.6B total | 3.2B | INT4/HQQ6/k6v6 | 8,748.2 tok/s | 127.75 tok/s | 274.83 tok/s | 19.4 GB max RSS (18.5 GiB) | 2944/2944 (100.0%) | 10,454 MB | [report](benchmarks/20260714_nemotron_nano_hqq6_k6v6_local_heatmap_mmap_evict_benchmark_report.log), [stdout](benchmarks/20260714_nemotron_nano_hqq6_k6v6_local_heatmap_mmap_evict_benchmark_stdout.log), [peak](benchmarks/20260714_nemotron_super_nano_local_heatmap_mmap_evict_rerun_peak_benchmark_hqq4_hqq6.log) |

Notes:

- Qwen3.6 parameters are counted from the loaded safetensors. Text weights are
  35.5B parameters; active parameters exclude the LM head.
- Separate Qwen3.6 HQQ4/HQQ6/HQQ8 approved route heatmaps exist. These two
  benchmark rows were rerun after incremental mmap cache-page eviction landed
  and loaded the matched HQQ4/HQQ6 heatmaps
  `qwen36_35b_hqq4_p00006` and `qwen36_35b_hqq6_p00006`; quick startup heatmap
  collection was skipped.
- QCN, Qwen3.5-35B, Qwen3.5-122B, Qwen3-235B, and Gemma4 rows use their
  model-name parameter scale until exact safetensor-count notes are added for
  those checkpoints. Their HQQ4/HQQ6/HQQ8 approved route heatmaps were built
  on 2026-07-14/15 and uploaded in commits `9dc6ae5` and `a5a7577`.
  Benchmark rows loaded matched HQQ4/HQQ6 heatmaps in require mode and skipped
  quick startup heatmap collection.
- Qwen3-Coder-Next, Qwen3.5-122B, and Qwen3-235B min free VRAM stayed close to
  the default 600 MB safety margin after HCS pressure eviction, which indicates
  the measured runs were using their available HCS budget aggressively. Qwen3.5
  35B and Gemma4 fit all routed experts on the RTX 5090, so their min free VRAM
  is much higher.
- Gemma4 still logged `Building GPU INT4 Marlin expert cache (one-time)` on
  repeated launches despite the cache file existing. The stats rows are valid,
  but cache reuse for Gemma4 startup is a follow-up issue.
- Step-3.7 parameters are counted from the loaded safetensors. Full checkpoint
  weights are 201.4B parameters; text weights plus LM head are 199.4B. Active
  parameters are text-path estimates from actual tensor shapes, using the
  configured 8-of-288 routed experts per MoE layer.
- Separate Step-3.7 HQQ4/HQQ6/HQQ8 approved route heatmaps exist and were
  uploaded in commit `61a175f`. These two benchmark rows were rerun after
  incremental mmap cache-page eviction landed and loaded the matched HQQ4/HQQ6
  heatmaps `step37_flash_hqq4_p00006` and
  `step37_flash_hqq6_p00006`; quick startup heatmap collection was skipped.
- Nemotron Super/Nano parameters are counted from loaded safetensors. Active
  parameters are estimated from actual routed-expert tensor sizes using the
  configured top-k experts per routed layer: Super uses 22-of-512 across 40
  routed layers; Nano uses 6-of-128 across 23 routed layers.
- Separate Nemotron Super and Nano HQQ4/HQQ6/HQQ8 route heatmaps were built
  locally on 2026-07-14 and uploaded in commit `9a7b8a2`. These benchmark rows
  were rerun after incremental mmap cache-page eviction landed, using the
  matched HQQ4/HQQ6 heatmaps in require mode; quick startup heatmap collection
  was skipped.
