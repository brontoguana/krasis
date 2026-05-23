# Changelog

## Unreleased

- Improved startup long VRAM calibration probe selection. The adaptive chooser
  now combines Rust prefill scratch growth with compact-KV stage-exact staging
  cost from the loaded model dimensions, then jumps directly to a predicted
  safe long probe with a validation reserve instead of stepping upward through
  small probes.

## 1.0.3 - 2026-05-23

- Made startup long VRAM calibration adaptive. Krasis now measures the short
  startup prefill first, then ramps the long calibration probe upward from
  observed low-water VRAM instead of immediately probing 80% of the context/KV
  cap. This avoids startup OOMs on WSL/shared-GPU systems where Windows
  processes reduce live free VRAM before Krasis starts.

## 1.0.2 - 2026-05-20

- Fixed VRAM pressure handling after runtime below-safety lows. The monitor now
  latches below-safety pressure events until HCS has reacted, and the HCS drain
  path preserves the measured low-water deficit as extra idle headroom instead
  of clearing the event when idle VRAM has merely recovered above the nominal
  safety margin.

## 1.0.1 - 2026-05-19

- Fixed setup compatibility for mixed Ampere and Blackwell systems. `krasis-setup`
  now selects the PyTorch CUDA wheel index from all visible GPU compute
  capabilities plus the installed driver runtime, so systems with RTX 50-series
  GPUs install CUDA 12.8 PyTorch wheels instead of CUDA 12.6 wheels that lack
  `sm_120` support.
- Added post-install validation for CUDA torch architecture support. If an
  existing CUDA torch build is visible but does not support one of the installed
  GPUs, setup reinstalls the correct wheel and the launcher reports the same
  unsupported-architecture condition clearly.

## 1.0.0 - 2026-05-19

- First stable Krasis release after the 0.1.x prerelease line.
- Moves the performance-sensitive runtime path to Rust/CUDA-focused execution,
  with full GPU prefill and GPU-executed decode.
- Adds HQQ attention cache support, compact KV cache modes, HCS expert
  residency management, measured VRAM budgeting, release wheels with sidecar
  assets, launcher model/download/tunnel workflows, and expanded benchmark and
  release validation.
- Current public benchmark coverage includes Qwen3-Coder-Next, Qwen3.6-35B,
  Qwen3.5-122B, and Qwen3-235B runs across the RTX 5090 and RTX A4500.

## 2026-02-28 — Decode optimisation: serial route matmul + AVX2 sigmoid

Baseline: 152.7 ms/tok (6.55 tok/s) at 12 threads on 5900X (WSL2, DDR4 dual-channel)
After: 119.0 ms/tok (8.40 tok/s) — 22% latency reduction, 28% throughput improvement

### Changes kept
- **Serial route matmul**: Changed MoE routing from parallel to serial dispatch.
  Rayon thread wake-up latency (~30us x 11 threads x 40 layers) dominated for 256
  tiny dot products (8KB per expert). Saved 32ms/tok (moe_route: 35.1ms to 3.1ms).
- **AVX2 vectorized sigmoid**: Replaced scalar sigmoid in SiLU activation with
  8-wide AVX2 fast exp approximation. Marginal (~1ms), but cleaner code with no downside.

### Changes tested and reverted
- Sub-tile splitting: cache line contention from two threads reading same tile data (+7%)
- Flattened MoE dispatch: cross-expert task mixing destroyed L2 cache locality (+22%)
- No nested MoE parallelism: only 8/12 threads active, lost work-stealing benefit (+10%)
- MADV_COLLAPSE (THP alternative): WSL2 returns EINVAL, Hyper-V doesn't support it
- Page table warming: pre-faulting PTEs polluted L1/L2, net slower (+5%)
- PREFETCHNTA (non-temporal prefetch on weights): ~2% improvement but within WSL2 noise

### Key findings
- 24 threads (SMT) is 2.3x slower than 12 threads (physical cores only) on 5900X
- THP does not work in WSL2 (AnonHugePages stays 0 despite madvise calls)
- TLB misses from 4KB pages on a 67GB model account for most of the remaining gap
  to theoretical bandwidth (~11 GB/s effective vs ~38 GB/s practical STREAM)
- MADV_COLLAPSE would likely work on native Linux and is worth retesting there

## 2026-02-28 — Log management and run notes

- On server start, existing `krasis.log` is archived to `logs/krasis_YYYYMMDD_HHMMSS.log` (timestamped from file mtime)
- Fresh `krasis.log` started for each run
- New `--note` parameter writes a run description header at the top of each log
- `logs/` directory gitignored (except `.gitkeep`)
