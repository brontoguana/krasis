# Krasis Benchmark Results

## Verification - 2026-06-14 (Gemma4 k4 restore clean speed after q2-BC32 K/V-alias rejection)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Accepted experimental decode flags `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1`
and `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` were the only enabled Krasis env
overrides. Prefill/decode attribution clocks and rejected q2 prototype envs
were explicitly unset. This verified the restored source after rejecting the
q2-BC32 K/V-alias prototype.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| k4 restore clean speed | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`; summary `5619.6` prefill, `92.43` internal decode, `155.69` HTTP. HCS stayed clean at `3840/3840`, min free decode VRAM `11474 MB`, and every Dynamic HCS row had `copy_failures=0`. The log contained no prefill attribution or q2 candidate markers. | restored baseline confirmed close to accepted `5594.9/92.42`; no optimization candidate started | [full log](20260614_1405_gemma4_hqq4_k4v4_restore_clean_speed.log) |

Notes:
- The verification gate and exact tmux command were recorded in
  `krasis-internal/DEBUGLOG.md` before launch.
- `./dev kill` was needed after success because the benchmark server remained
  as a zombie child during shutdown; cleanup left no tmux/server process and
  GPUs idle.
- This run confirms the q2-BC32 K/V-alias revert restored k4 timing-off speed
  near the accepted clean marker.

## Rejected - 2026-06-14 (Gemma4 HD512 q2-BC32 K/V-alias prototype)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Accepted experimental decode flags `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1`
and `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` remained enabled. This implemented the
design-only follow-up behind a new explicit gate,
`KRASIS_PREFILL_HD512_Q2_BC32_KV_ALIAS=1`, keeping default q2-BC16 and the
existing capacity-blocked `KRASIS_PREFILL_HD512_Q2_BC32=1` path separate.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| q2-BC32 K/V-alias build | `./dev build` | HQQ4 | k4v4/k6v6 path | Build passed; final-source build log reports `OK Build complete` and `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=133`. | build passed | [full log](20260614_1339_gemma4_hqq4_hd512_q2_bc32_kv_alias_build.log) |
| k4 q2-BC32 K/V-alias attribution | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 KRASIS_PREFILL_TIMING=1 KRASIS_PREFILL_HD512_KERNEL_CLOCKS=1 KRASIS_PREFILL_HD512_Q2_BC32_KV_ALIAS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`; summary `5270.0` prefill, `92.42` internal decode, `160.15` HTTP. HCS stayed clean at `3840/3840`, min free decode VRAM `11474 MB`, and every Dynamic HCS row had `copy_failures=0`. Selection explicitly reported `hd512_q2_mode=bc32_kv_alias`, `hd512_tile_cols=32`, `hd512_q_heads_per_block=2`. Representative long HD512 q2 rows at `11824` tokens were `373.1-374.4 ms/layer` with K load `93.9-94.2`, QK `28.2-28.3`, softmax `58.7-58.9`, V load `93.9-94.2`, PV `97.4-97.8`, final write `0.1`. | attribution gate passed; proceed to clean k4 speed | [full log](20260614_1346_gemma4_hqq4_k4v4_hd512_q2_bc32_kv_alias_attr_timing.log) |
| k4 q2-BC32 K/V-alias clean speed | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 KRASIS_PREFILL_HD512_Q2_BC32_KV_ALIAS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`; summary `4817.6` prefill, `92.38` internal decode, `160.30` HTTP. HCS stayed clean at `3840/3840`, min free decode VRAM `11474 MB`, and every Dynamic HCS row had `copy_failures=0`. The log contained no prefill attribution markers. | rejected: clean prefill regressed versus accepted k4 clean baseline `5594.9`; candidate reverted | [full log](20260614_1351_gemma4_hqq4_k4v4_hd512_q2_bc32_kv_alias_clean_speed.log) |
| restore after q2-BC32 K/V-alias rejection | `./dev build` | HQQ4 | k4v4/k6v6 path | Restored-source build passed; final-source build log reports `OK Build complete` and `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=132`. | restored source verified | [full log](20260614_1357_restore_after_hd512_q2_bc32_kv_alias_reject_build.log) |

Notes:
- The implementation gate was recorded in `krasis-internal/DEBUGLOG.md`
  before source edits.
- The attribution split reduced the measured HD512 q2 kernel phase total, but
  the required clean timing-off run regressed prefill from the accepted k4
  clean marker `5594.9` to `4817.6`, so the candidate failed the full metric
  gate.
- No k6v6 or QCN guard was run because the clean k4 speed gate failed.
- Revert removed the alias env, symbols, dispatch, and `v_tile_load` timing
  layout. The existing q2-BC32 capacity-blocked path and HD512 timing clocks
  remain.
- Cleanup via `./dev kill` left no tmux/server process and GPUs idle.

## Implementation - 2026-06-14 (Gemma4 HD512 q2-BC32 prototype capacity gate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Accepted experimental decode flags `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1`
and `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` remained enabled. This implemented the
planned env-gated q2-BC32 prefill prototype behind
`KRASIS_PREFILL_HD512_Q2_BC32=1`; default env-off q2-BC16 dispatch remains
unchanged.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| q2-BC32 build | `./dev build` | HQQ4 | k4v4/k6v6 path | Build passed; final-source build log reports `OK Build complete` and `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=128`. | build accepted | [full log](20260614_1316_gemma4_hqq4_hd512_q2_bc32_build.log) |
| k4 q2-BC32 attribution | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 KRASIS_PREFILL_TIMING=1 KRASIS_PREFILL_HD512_KERNEL_CLOCKS=1 KRASIS_PREFILL_HD512_Q2_BC32=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | The required first attribution run did not reach benchmark/correctness because the env-gated shared-memory check failed visibly before launch: q2-BC32 requires `106496` bytes opt-in shared memory and the selected RTX 5090 reports `101376` bytes. No hidden fallback to q2-BC16 occurred. | hardware-capacity blocked on this GPU; no clean speed run | [full log](20260614_1320_gemma4_hqq4_k4v4_hd512_q2_bc32_attr_timing.log) |

Notes:
- The metric gate was recorded in `krasis-internal/DEBUGLOG.md` before source
  edits.
- The failure is the planned visible failure mode for insufficient runtime
  shared memory under `KRASIS_PREFILL_HD512_Q2_BC32=1`, not a speed result.
- No k4 clean speed, k6v6, or QCN guard was run because k4 attribution could
  not start on this hardware.
- Cleanup via `./dev kill` left no tmux/server process and GPUs idle.

## Attribution - 2026-06-14 (Gemma4 HQQ4/k6v6 focused dual-norm + GPU-softcap split)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Accepted experimental decode flags `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1`
and `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` remained enabled. This was a
source-free attribution pass using existing prefill and decode clocks only;
the heavy per-block HD512 kernel clock and all rejected candidate envs were
explicitly unset.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Focused k6v6 attribution | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 KRASIS_PREFILL_TIMING=1 KRASIS_DECODE_FINAL_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 KRASIS_DECODE_GQA_OTHER_CLOCKS=1 KRASIS_DECODE_GQA_HD256_ATTN_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k6v6-a16.conf --timing` | HQQ4 | k6v6 | Full run passed correctness `14/14`, `ALL TESTS PASSED`; summary `5537.2` prefill, `64.04` internal decode, `116.30` HTTP. HCS stayed clean at `3840/3840`, min free decode VRAM `11748 MB`, zero cold DMA/DMA calls, and every Dynamic HCS row had `copy_failures=0`. Representative prefill rows: `4999` tokens component timing `652.2 ms` / `7665 tok/s`, with HD512 `custom_tiled` launch `307.4 ms` over 5 calls, GQA projection `74.5 ms`, FA2 `28.5 ms`, O projection `39.2 ms`, MoE `128.4 ms` with W1+act `77.7`, W2 `29.8`, scatter `14.3`; capped `10524` token rows were `2048.6-2050.3 ms` / `5133-5137 tok/s`, with HD512 `custom_tiled` launch `1285.4-1287.2 ms`, GQA projection `157.8-157.9`, FA2 `112.3`, O projection `82.8-83.1`, and MoE `257.3-257.5`. Internal decode rows were `64.1/62.3/60.8` tok/s for `49/99/249` tokens; graph timing showed total `15.43/15.89/16.28 ms/tok`, GPU compute `4.42-4.60`, sync wait `10.63-11.55`, MoE expert `3.34`, GQA path `3.07/3.34/3.87`, route/topk `2.86`, final segment `1.05`, and no cold DMA. | attribution only; no distinct non-rejected `>0.2 ms/tok` target selected | [full log](20260614_1255_gemma4_hqq4_k6v6_dual_norm_softcap_attr_timing.log) |

Notes:
- The metric gate and full tmux command were recorded in
  `krasis-internal/DEBUGLOG.md` before launch.
- Compared with clean timing-off markers, k6v6 remains close on prefill
  (`5537.2` attributed / `5243.5` clean versus k4 clean `5594.9`) but slower
  on decode (`64.04` attributed / `65.08` clean versus k4 clean `92.42`).
- The k6v6 long prefill cap is `10524` tokens, so the capped long-row absolute
  timings are not directly comparable to k4's `14780`-token rows. The dominant
  prefill bucket is still the HD512 `custom_tiled` fallback, which points to
  the same broad HD512 attention redesign already identified rather than a
  narrow k6v6-specific candidate.
- Decode MoE and GQA path costs are broadly similar to the k4 timing rows; the
  remaining route/topk and graph-route sync cost does not expose a distinct
  safe implementation path. Obvious dense, W2, W13, GQA, and LM-head variants
  are already rejected or require larger design work.
- No source changes were made and no performance candidate was started.
- `./dev kill` was required after completion because the wrapper left the
  benchmark server resident; cleanup cleared the server, tmux session, and GPU
  memory.

## Speed - 2026-06-14 (Gemma4 HQQ4/k6v6 clean dual-norm + GPU-softcap baseline)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Accepted experimental decode flags `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1`
and `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` were enabled. Known attribution clocks
and rejected candidate envs were explicitly unset, and `--timing` was omitted.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Clean timing-off Gemma k6v6 speed | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 ./dev test tests/gemma-4-4-hqq4-k6v6-a16.conf` | HQQ4 | k6v6 | Full run passed correctness `14/14`, `ALL TESTS PASSED`; summary `5243.5` prefill, `65.08` internal decode, `119.07` HTTP. HCS stayed clean at `3840/3840`, min free decode VRAM `11748 MB`, zero cold DMA/DMA calls, and every Dynamic HCS row had `copy_failures=0`. Benchmark prefill rows were `2721.4` at `1000` tokens, `5243.5` at `4999`, `4251.6` at `10000`, and `4319.1/4554.7/4337.8` on capped `10524/10524/10524` rows. Internal decode rows were `65.08/64.09/63.71` tok/s for `50/100/250`; network rows were `119.07/82.55/67.66`. Startup built and loaded the Gemma Marlin expert cache for this k6v6 run before benchmark warmup/timing. | accepted clean k6v6 baseline for comparison before HD512 redesign planning | [full log](20260614_1247_gemma4_hqq4_k6v6_dual_norm_softcap_clean_speed.log) |

Notes:
- The metric gate and full tmux command were recorded in
  `krasis-internal/DEBUGLOG.md` before launch.
- The run used the built `./dev test` path. No source changes were made and no
  performance candidate was started.
- `./dev kill` was required after completion because the wrapper left the
  benchmark server resident; cleanup cleared the server, tmux session, and GPU
  memory.

## Speed - 2026-06-14 (Gemma4 HQQ4/k4v4 clean dual-norm + GPU-softcap baseline)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Accepted experimental decode flags `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1`
and `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` were enabled. Known attribution clocks
and rejected candidate envs were explicitly unset, and `--timing` was omitted.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Clean timing-off Gemma speed | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`; summary `5594.9` prefill, `92.42` internal decode, `165.46` HTTP. HCS stayed clean at `3840/3840`, min free `11474 MB`, zero cold DMA/DMA calls, and `copy_failures=0`. Benchmark prefill rows were `1871.5` at `1000` tokens, `5594.9` at `4999`, `4676.0` at `10000`, and `3662.3/3594.9/3786.2` on capped `14780/14780/14779` rows. Internal decode rows were `92.42/90.29/84.98` tok/s for `50/100/250`; network rows were `165.46/112.44/92.09`. Startup rebuilt the Gemma Marlin expert cache, but this occurred before warmup/benchmark timing and does not make the speed rows attribution-enabled. | accepted clean experimental baseline; replaces anomalous `4931.0/92.31` marker | [full log](20260614_1238_gemma4_hqq4_k4v4_dual_norm_softcap_clean_speed.log) |

Notes:
- The metric gate and full tmux command were recorded in
  `krasis-internal/DEBUGLOG.md` before launch.
- The run used the built `./dev test` path. No source changes were made.
- `./dev kill` was required after completion because the wrapper left the
  benchmark server resident; cleanup cleared the server and GPU memory.

## Attribution - 2026-06-14 (Gemma4 HQQ4/k4v4 HD512 custom-tiled kernel split)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Accepted experimental decode flags `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1`
and `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` remained enabled. This was a
timing-only source change to split the HD512 q2 `custom_tiled` fallback kernel
that dominated capped long prefill rows in the previous prefill attribution.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Initial build | `./dev build` | HQQ4 | k4v4 | Build failed with Rust `E0308` because the timing report passed an `Option<f64>` event duration where the report helper expected `f64`. | fixed measurement plumbing before running | [build log](20260614_1219_gemma4_hqq4_k4v4_prefill_hd512_kernel_attr_build.log) |
| Fixed build | `./dev build` | HQQ4 | k4v4 | Build passed, `duration_s=129`. | timing-only attribution build accepted for run | [build log](20260614_1221_gemma4_hqq4_k4v4_prefill_hd512_kernel_attr_fix_build.log) |
| HD512 q2 kernel attribution | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 KRASIS_PREFILL_TIMING=1 KRASIS_PREFILL_HD512_KERNEL_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`; summary `3865.8` prefill, `92.17` internal decode, `157.09` HTTP. HCS stayed clean at `3840/3840`, min free `11474 MB`, zero cold DMA/DMA calls, and `copy_failures=0`. The prefill number is not a speed-regression marker because this run enabled per-block kernel clocks with globaltimer/atomic attribution. Capped long-row representative q2 kernel rows at `14780` tokens showed each of the five HD512 layers at `627.0-627.9 ms`, `7392` blocks, `3418800` tiles, with phase split per layer: Q load `1.2 ms`, K/V tile load `177.7-177.9 ms`, QK score `44.4-44.5 ms`, softmax/rescale/probability write `156.8-157.1 ms`, PV accumulation `246.7-247.0 ms`, final write `0.1 ms`, residual `0.0 ms`. Aggregate `flash_attn_tiled_launch` for the 5 calls was `3137.4 ms`. | attribution only; no performance candidate attempted | [full log](20260614_1224_gemma4_hqq4_k4v4_prefill_hd512_kernel_attr_timing.log) |

Notes:
- The metric gate was recorded in `krasis-internal/DEBUGLOG.md` before the
  source change.
- The source change is timing-only and opt-in via
  `KRASIS_PREFILL_HD512_KERNEL_CLOCKS=1`; default prefill kernel selection is
  unchanged.
- The split identifies the real dominant phases: PV accumulation, K/V tile
  load, and softmax/rescale/probability write. Q load and final write are
  negligible.
- No safe narrow performance candidate was attempted from this pass. The
  dominant phases point at a larger HD512 attention kernel redesign rather than
  an isolated swap or launch-level fix.
- The test wrapper completed but left the benchmark server resident; `./dev
  kill` cleared the process and GPU memory afterward.

## Attribution - 2026-06-14 (Gemma4 HQQ4/k4v4 prefill focus after decode baseline)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Accepted experimental decode flags `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1`
and `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` remained enabled. This was a
source-free attribution pass using existing `KRASIS_PREFILL_TIMING=1` Rust
prefill clocks after decode optimization paused on rejected buckets and
post-FFN fusion needing a larger design.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Prefill attribution | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`; summary `5562.6` prefill, `92.53` internal decode, `160.33` HTTP. HCS stayed clean at `3840/3840`, min free `11474 MB`, zero cold DMA/DMA calls, and `copy_failures=0`. The low prior timing-off prefill marker `4931.0` was not reproduced. Benchmark prefill rows: `2860.1` at `1000` tokens, `5562.6` at `4999`, `4660.3` at `10000`, and `3796.9/3794.6/3767.9` on capped `14780/14780/14779` rows. Long-row attribution showed custom-tiled HD512 fallback launch `2503.4-2505.0 ms` over 5 calls, GQA projection `222.2-222.3 ms`, FA2 `214.2-217.4 ms`, O projection `117.4-122.8 ms`, HQQ `marlin_float_zp` `291.1-291.3 ms`, MoE `353.4-353.7 ms` with W1+act `217.4-217.6`, W2 `79.8`, scatter `42.4`, and residual other `14.7-15.1`. KV append was only `3.4-3.5 ms` with `~0.3 ms` wall-minus-kernel. | attribution only; no new safe non-rejected prefill candidate selected | [full log](20260614_1202_gemma4_hqq4_k4v4_prefill_attr_timing.log) |

Notes:
- The metric gate was recorded in `krasis-internal/DEBUGLOG.md` before the
  run. No Krasis source change was made for this pass.
- The run rebuilt and reloaded the Gemma Marlin expert cache during launch
  (`12.1 GB`, `37s` total startup impact); that is not part of the measured
  prefill benchmark rows.
- Long rows are capped around `14,780` tokens by the runtime scratch/chunk
  choice, so the nominal `20K/35K/50K` labels are not true full-length rows in
  this configuration.
- The old KV append wall-debt signature is absent in this run: long-row
  `kv_append` is `3.4-3.5 ms`, close to the `3.1-3.2 ms` kernel event.
- The largest throughput-relevant prefill cost is still the known
  `custom_tiled` HD512 fallback. It is a real target, but this source-free
  attribution pass did not identify a new scoped safe implementation path, and
  prior narrow custom-tiled/GQA attempts have already been mixed or rejected.
  No performance candidate was attempted.
- Min free VRAM is far above the 600 MB safety margin because Gemma already
  has full HCS residency (`3840/3840`), so there is no additional HCS residency
  to gain in this run.

## Attribution - 2026-06-14 (Gemma4 HQQ4/k4v4 MoE hidden write/post-FFN split)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Accepted experimental flags `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and
`KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` remained enabled. This was an
attribution-only pass focused on splitting the MoE hidden write/post-FFN span
after the prior post-output split isolated it at `0.49-0.54 ms/tok`.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Initial build | `./dev build` | HQQ4 | k4v4 | Build passed, `duration_s=128`. | timing-only instrumentation build | [build log](20260614_1140_gemma4_hqq4_k4v4_moe_hidden_attr_build.log) |
| Aborted hidden attribution | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Stopped during warmup after the new hidden sub-span totals exceeded the parent hidden span, producing negative residuals. Root cause was measurement-only: the new hidden sub-span accumulators were initialized but not reset in the per-request timing reset paths. | invalid attribution; fixed measurement before using data | [partial log](20260614_1144_gemma4_hqq4_k4v4_moe_hidden_attr_timing.log) |
| Fixed build | `./dev build` | HQQ4 | k4v4 | Build passed, `duration_s=128`. | reset bug fixed; attribution rerun | [build log](20260614_1150_gemma4_hqq4_k4v4_moe_hidden_attr_fix_build.log) |
| MoE hidden write attribution | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`; summary `5507.8` prefill, `79.59` internal decode, `175.39` HTTP. HCS stayed clean at `3840/3840`, min free `11468 MB`, zero cold DMA/DMA calls, and `copy_failures=0`. Representative internal decode rows showed hidden write/post-FFN total `0.58-0.68 ms/tok`, `post_ffn_norm2` `0.21-0.23`, dense+MoE add `0.05-0.07`, `post_ffn_norm` `0.21-0.23`, residual add `0.05-0.07`, layer scalar `0.05-0.07`, endpoint `0.02-0.03`, and residual `~0.00`. | attribution only; no safe optimization candidate selected | [full log](20260614_1154_gemma4_hqq4_k4v4_moe_hidden_attr_timing.log) |

Notes:
- The metric gate was recorded before the source change in
  `krasis-internal/DEBUGLOG.md`.
- The source change is timing-only: it widens the Gemma MoE-route clock layout
  and adds markers around `post_ffn_norm2`, dense+MoE add,
  `post_ffn_norm`, residual add, optional layer scalar, and endpoint overhead.
  It does not change math, calibration, HCS, graph capture machinery, or
  non-Gemma default behavior.
- The first attribution attempt was rejected as invalid measurement data. The
  fix preserved the instrumentation and added the missing reset assignments
  for the new hidden sub-span accumulators.
- Each RMSNorm is a real `>0.2 ms/tok` sub-bucket, but this split does not
  expose a distinct safe implementation path. A performance candidate would
  need a separately gated post-FFN fusion design around dependent RMSNorm/add
  work; none was attempted here.
- Add/scalar/endpoint pieces are below the useful threshold.
- No timing-off speed test or QCN guard was run because this pass added
  attribution-only instrumentation and did not attempt a performance
  candidate.

## Attribution - 2026-06-14 (Gemma4 HQQ4/k4v4 MoE post-output split)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Accepted experimental flags `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and
`KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` remained enabled. This was an
attribution-only pass focused on splitting the previously grouped MoE
`post-output` span after focused MoE attribution showed it at
`0.50-0.57 ms/tok`. The source change is timing-only: the Gemma graph
MoE-route clock layout now splits shared gate, shared W13/reduce, shared W2,
routed scaling, hidden write/post-FFN work, and endpoint overhead. It does not
alter math, calibration, HCS, graph capture, or non-Gemma default behavior.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Build | `./dev build` | HQQ4 | k4v4 | Build passed, `duration_s=127`. | timing-only instrumentation accepted for attribution | [build log](20260614_1125_gemma4_hqq4_k4v4_moe_post_attr_build.log) |
| MoE post-output attribution | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`; summary `5293.8` prefill, `80.69` internal decode, `186.31` HTTP. HCS stayed clean at `3840/3840`, min free `11470 MB`, zero cold DMA/DMA calls, and `copy_failures=0`. Representative rows showed MoE out `0.06-0.08 ms/tok`, post-output `0.55-0.63`, shared gate/W13/reduce/W2 all `0.00`, routed scale `0.02-0.03`, hidden write/post-FFN `0.49-0.54`, endpoint `0.04-0.06`, and post residual `~0.00`. | attribution only; no optimization candidate selected | [full log](20260614_1130_gemma4_hqq4_k4v4_moe_post_attr_timing.log) |

Notes:
- The metric gate was recorded before the source change in
  `krasis-internal/DEBUGLOG.md`.
- The split shows Gemma's `post-output` cost is not shared-expert work:
  shared gate/W13/reduce/W2 are effectively zero in this config.
- The only newly isolated `>0.2 ms/tok` piece is hidden write/post-FFN work
  at about `0.49-0.54 ms/tok`, but that piece still groups two RMSNorms, two
  adds, and optional layer-scalar writeback. It needs a further focused split
  before there is a distinct safe implementation path.
- Routed scaling is only `0.02-0.03 ms/tok`; endpoint overhead is only
  `0.04-0.06 ms/tok`; routed weighted combine remains `0.06-0.08 ms/tok`.
- The completed test left a tmux/server process and GPU memory resident;
  `tmux kill-session` plus `./dev kill` cleared it, leaving both GPUs idle.
- No timing-off speed test or QCN guard was run because this pass added
  attribution-only instrumentation and did not attempt a performance candidate.

## Attribution - 2026-06-14 (Gemma4 HQQ4/k4v4 focused MoE split)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Accepted experimental flags `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and
`KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` remained enabled. This was an
attribution-only pass focused on MoE after route/final attribution. No source
change was made; the run used existing MoE-route and W2 preload clocks to
inspect W13, W13 reduce, activation/SiLU, W2 load/compute/store, weighted
accumulation, post/endpoint work, and graph/launch overhead without retrying
paired W2, W2 FP32 shared activation, W13 direct, activation precompute/reuse,
or rejected dense/GQA/final candidates.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Focused MoE attribution | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`; summary `5005.2` prefill, `81.13` internal decode, `187.11` HTTP. HCS stayed clean at `3840/3840`, min free `11470 MB`, zero cold DMA/DMA calls, and `copy_failures=0`. Representative rows showed MoE expert `3.36-3.53 ms/tok`, W13 `1.08-1.12`, W13 reduce `0.05-0.07`, activation+W2 `1.66-1.69`, activation `1.19`, W2 prep `0.02`, W2 compute/output `0.41`, load `0.31-0.35`, SiLU/multiply `0.62-0.66`, shared store `0.21-0.23`, sync `0.04`, weighted accumulation `0.06-0.08`, graph/launch `0.04-0.06`, post-output `0.50-0.57`, and repeated work about `53.4-53.6` with `1408` blocks/segment. | attribution only; no safe new candidate selected | [full log](20260614_1116_gemma4_hqq4_k4v4_moe_focus_attr_timing.log) |

Notes:
- The metric gate was recorded before the run in
  `krasis-internal/DEBUGLOG.md`; no source edit was made.
- W13 and W13 reduce are real cost, but the scoped W13 direct-BF16 path was
  already rejected and this split did not reveal another distinct safe W13
  implementation path.
- Activation/SiLU and W2 preload/store remain the largest MoE sub-buckets, but
  the distinct safe-looking paths already tried there are rejected:
  activation precompute/reuse, paired-output W2 tiles, and FP32 shared
  activation. The remaining ideas would need new attribution or correctness
  reasoning before becoming safe candidates.
- Weighted accumulation, W13 reduce, and graph/launch are below the useful
  `>0.2 ms/tok` threshold. The `post-output` interval is above that threshold
  but is a grouped shared/scale/post-FFN output span rather than a single
  isolated kernel, so it needs a separate focused split before it can justify
  an optimization candidate.
- The completed test left a tmux/server process and GPU memory resident;
  `tmux kill-session` plus `./dev kill` cleared it, leaving both GPUs idle.
- Min free VRAM is far above the 600 MB safety margin, but Gemma already has
  `3840/3840` expert coverage, so there is no missed HCS residency to reclaim.
- No timing-off speed test or QCN guard was run because this was
  attribution-only with no source candidate.

## Attribution - 2026-06-14 (Gemma4 HQQ4/k4v4 route/final focused split)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Accepted experimental flags `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and
`KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` remained enabled. This was an
attribution-only pass after moving off non-active GQA. No source change was
made; the run used existing route-prep/final clocks only, plus MoE-route
context, to inspect route-prep dense/overhead and final LM-head BF16 cuBLAS
without retrying rejected dense gate+up, dense-down custom, or final LM-head
Marlin INT8 candidates.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Focused route/final attribution | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 KRASIS_DECODE_FINAL_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`; summary `5618.9` prefill, `83.63` internal decode, `192.76` HTTP. HCS stayed clean at `3840/3840`, min free `11466 MB`, zero cold DMA/DMA calls, and `copy_failures=0`. Representative internal decode rows showed route-prep dense pre `0.25-0.26`, dense gate `0.34-0.35`, dense up `0.34-0.35`, dense activation `0.06-0.07`, dense down `0.37-0.38`, dense post norms `0.42-0.43`, router norm `0.02-0.03`, remaining overhead `0.22-0.25`, final RMSNorm `0.01`, final LM-head BF16 cuBLAS `0.92`, GPU softcap `0.00`, D2H logits `0.13`, graph residual `0.12`, and final segment/sync `1.05 ms/tok`. | attribution only; no safe new candidate selected | [full log](20260614_1108_gemma4_hqq4_k4v4_route_final_attr_timing.log) |

Notes:
- The metric gate was recorded before the run in
  `krasis-internal/DEBUGLOG.md`; no source edit was made.
- Route-prep has real cost, but the remaining non-rejected pieces are either
  below or barely at the useful-upside threshold, and the larger dense
  projection paths map to already rejected dense gate+up or dense-down custom
  work.
- Final LM-head BF16 cuBLAS remains the largest clean final bucket at about
  `0.92 ms/tok`, but the scoped Marlin INT8 replacement was already rejected
  with a severe regression and CUDA illegal address. No second LM-head
  candidate is justified from this split alone.
- The completed test left a tmux/server process and GPU memory resident;
  `tmux kill-session` plus `./dev kill` cleared it, leaving both GPUs idle.
- Accepted experimental baseline remains
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`
  at timing-off Gemma marker `4931.0/92.31/160.16`.

## Attribution - 2026-06-14 (Gemma4 HQQ4/k4v4 non-active GQA focused split)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Accepted experimental flags `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and
`KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` remained enabled. This was an
attribution-only pass before any further non-active GQA optimization, because
the previous GQA candidates made this bucket risky. The timing-only change
expanded existing non-active GQA debug markers to split projection, Q/K/V norm,
RoPE, KV-cache write, attention/reduce, BF16/gated conversion, O-input prep,
O projection, endpoint/gap, and existing HD256 score/weight-V/final-output
clocks. Calibration, HCS, production graph behavior, kernels/math, prefill,
and non-Gemma behavior were left unchanged.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Focused attribution build | `./dev build` | n/a | n/a | Build passed after adding timing-only non-active GQA detail clocks; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=128`. | proceed to focused attribution | [full log](20260614_1052_gemma4_hqq4_k4v4_gqa_other_detail_attr_build.log) |
| Focused non-active GQA attribution | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 KRASIS_DECODE_FINAL_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 KRASIS_DECODE_GQA_OTHER_CLOCKS=1 KRASIS_DECODE_GQA_HD256_ATTN_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`; summary `4950.5` prefill, `77.07` internal decode, `156.54` HTTP. HCS stayed clean at `3840/3840`, min free `11462 MB`, zero cold DMA/DMA calls, and `copy_failures=0`. Representative 49/99/249/511 rows showed non-active GQA projection `0.92/0.95/0.95/0.92`, Q/K/V norm `0.10/0.12/0.12/0.10`, RoPE `0.04/0.06/0.05/0.04`, KV write `0.11/0.13/0.12/0.11`, attention `0.68/0.85/1.30/1.88`, BF16 `0.04/0.05/0.05/0.04`, O input `0.02`, O projection `0.44/0.46/0.45/0.44`, endpoint `2.34/2.65/3.07/3.55`, and HD256 internal score/weight-V/final `0.18/0.42/0.04`, `0.20/0.56/0.04`, `0.33/0.89/0.04`, `0.48/1.33/0.04`. | attribution only; no safe candidate selected | [full log](20260614_1055_gemma4_hqq4_k4v4_gqa_other_detail_attr_timing.log) |

Notes:
- The metric gate was recorded before source edits in
  `krasis-internal/DEBUGLOG.md`.
- Newly separated Q/K/V norm, RoPE, and KV-write sub-buckets are each below
  the `>0.2 ms/tok` useful-upside threshold.
- The large projection bucket was already known and fused-QKV split-elision is
  rejected. The large HD256 attention weight/V bucket points at the same risky
  area as the rejected tile-cap, score-cache, and HD256-specialization attempts.
  O projection remains a real `~0.44-0.46 ms/tok` cost, but this split alone
  does not identify a scoped non-rejected GQA candidate that preserves the
  current HQQ GEMV contract. No optimization candidate was attempted.
- Accepted experimental baseline remains
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`
  at timing-off Gemma marker `4931.0/92.31/160.16`.

## Rejected - 2026-06-14 (Gemma4 HQQ4/k4v4 route-prep fused pre-norm copy candidate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Accepted experimental flags `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and
`KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` remained enabled. Current speed markers at
the start of the pass were default Gemma k4 `5613.8/65.89`, dual-norm
`5626.1/66.96`, dual-norm + GPU-softcap timing-off `4931.0/92.31`, target
`8000/120`. Fresh post-softcap attribution had already passed `14/14` with
`5597.4` prefill, `77.58` internal decode, `153.06` HTTP, HCS `3840/3840`,
min free `11464 MB`, and zero cold DMA/copy failures. Excluding all prior
rejected W2, W13, dense, GQA, and final-LM-head candidates, the non-rejected
measured route-prep dense pre-norm bucket was `0.23-0.26 ms/tok`. The rejected
candidate tried replacing only the graph route-prep `cuMemcpyDtoDAsync`
residual copy plus pre-FFN RMSNorm pair with existing `fused_add_rmsnorm`
using `first_layer=1`, under `KRASIS_DECODE_ROUTE_PREP_FUSED_PRE_NORM=1`.
Calibration, HCS, graph capture machinery, dense projections, GQA, prefill,
and non-Gemma behavior were left unchanged.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Candidate build | `./dev build` | n/a | n/a | Build passed for `KRASIS_DECODE_ROUTE_PREP_FUSED_PRE_NORM=1`; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=129`. | proceed to timing gate | [full log](20260614_1037_gemma4_hqq4_k4v4_route_fused_pre_norm_build.log) |
| Timing gate | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 KRASIS_DECODE_ROUTE_PREP_FUSED_PRE_NORM=1 KRASIS_DECODE_FINAL_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 KRASIS_DECODE_GQA_OTHER_CLOCKS=1 KRASIS_DECODE_GQA_HD256_ATTN_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`; summary `5611.7` prefill, `77.80` internal decode, `157.72` HTTP. HCS stayed clean at `3840/3840`, min free `11462 MB`, zero cold DMA/DMA calls, and `copy_failures=0`. The target dense pre-norm bucket moved only from fresh `0.23-0.26 ms/tok` to mostly `0.23` with occasional `0.21-0.22`; route prep/overhead stayed mostly `2.60 ms/tok` and regressed on some representative rows versus fresh `2.35/2.63/2.39`. The tiny internal-decode delta `77.58 -> 77.80 tok/s` was not accepted as a clear win. | rejected; no timing-off Gemma speed or QCN guard | [full log](20260614_1040_gemma4_hqq4_k4v4_route_fused_pre_norm_timing.log) |
| Restore accepted source after rejection | `./dev build` | n/a | n/a | Build passed after reverting only the fused pre-norm copy branch; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=127`; candidate env/source symbols are absent. | restore/build hygiene | [full log](20260614_1048_restore_after_route_fused_pre_norm_reject_build.log) |

Notes:
- The metric gate was recorded before source edits in
  `krasis-internal/DEBUGLOG.md`.
- This candidate was Gemma-route-prep relevant, env-gated, and
  dimension-derived from runtime hidden size/shared-memory capacity, but the
  measured improvement was below the `>0.2 ms/tok` useful-upside threshold and
  did not improve route prep or total decode enough to accept.
- Accepted experimental baseline remains
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`
  at timing-off Gemma marker `4931.0/92.31/160.16`.

## Rejected - 2026-06-14 (Gemma4 HQQ4/k4v4 final LM-head Marlin INT8 candidate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Accepted experimental flags `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and
`KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` remained enabled. The fresh attribution
was run before editing because removing CPU softcap changed the decode bucket
shape. Current speed markers at the start of the pass were default Gemma k4
`5613.8/65.89`, dual-norm `5626.1/66.96`, dual-norm + GPU-softcap timing-off
`4931.0/92.31`, target `8000/120`. Excluding paired W2, global activation
precompute/reuse, W13 direct, dense gate+up fusion, dense-down custom, GQA
tile-cap, score-cache, HD256 specialization, fused-QKV split-elision, and W2
FP32 shared activation, the largest scoped final bucket was LM head at about
`0.93 ms/tok`. The rejected candidate tried registering the existing Gemma
INT8-source final LM head through the existing Marlin INT8 setup path under
`KRASIS_DECODE_FINAL_LM_HEAD_MARLIN_INT8=1`. Calibration, HCS, graph capture,
prefill, and non-Gemma behavior were left unchanged.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Fresh post-softcap timing attribution | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 KRASIS_DECODE_FINAL_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 KRASIS_DECODE_GQA_OTHER_CLOCKS=1 KRASIS_DECODE_GQA_HD256_ATTN_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`; summary `5597.4` prefill, `77.58` internal decode, `153.06` HTTP. HCS stayed clean at `3840/3840`, min free `11464 MB`, zero cold DMA/DMA calls, and `copy_failures=0`. Representative post-softcap rows showed LM head `0.93 ms/tok`, D2H logits about `0.15-0.16`, host softcap `0.00`, route prep/overhead `2.35-2.65`, MoE activation+W2 about `1.66-1.69`, and weighted expert accumulation still below threshold. | selected final LM-head candidate | [full log](20260614_1018_gemma4_hqq4_k4v4_post_softcap_fresh_attr_timing.log) |
| Candidate build | `./dev build` | n/a | n/a | Build passed for `KRASIS_DECODE_FINAL_LM_HEAD_MARLIN_INT8=1`; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=7`. | proceed to timing gate | [full log](20260614_1029_gemma4_hqq4_k4v4_final_lm_head_marlin_int8_build.log) |
| Timing gate | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 KRASIS_DECODE_FINAL_LM_HEAD_MARLIN_INT8=1 KRASIS_DECODE_FINAL_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 KRASIS_DECODE_GQA_OTHER_CLOCKS=1 KRASIS_DECODE_GQA_HD256_ATTN_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Candidate failed before benchmark completion. Early timing rows showed LM head regressed from the fresh `~0.93 ms/tok` baseline to `9.40-9.52 ms/tok`, then graph replay failed with `final sync: CUDA_ERROR_ILLEGAL_ADDRESS` followed by boundary-sync fatal CUDA context error. HCS was still clean before failure (`3840/3840`, representative `240/0`, `copy_failures=0`). | rejected; no timing-off Gemma speed or QCN guard | [full log](20260614_1030_gemma4_hqq4_k4v4_final_lm_head_marlin_int8_timing.log) |
| Restore accepted source after rejection | `./dev build` | n/a | n/a | Build passed after reverting only the final LM-head Marlin INT8 candidate; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=8`; candidate env/source symbols are absent. | restore/build hygiene | [full log](20260614_1033_restore_after_final_lm_head_marlin_int8_reject_build.log) |

Notes:
- The metric gate was recorded before source edits in
  `krasis-internal/DEBUGLOG.md`.
- The candidate was Gemma-final relevant, env-gated, and dimension-derived
  from the source LM-head tuple and Marlin compatibility checks, but it failed
  both the local LM-head timing target and runtime safety.
- Accepted experimental baseline remains
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`
  at timing-off Gemma marker `4931.0/92.31/160.16`.

## Accepted - 2026-06-14 (Gemma4 HQQ4/k4v4 GPU final-logit softcap)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Baseline experimental flag `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` remained
enabled. Rejected paired-W2, W13 direct, dense gate+up, dense-down custom, and
rejected GQA variants were not retried. A focused final/sync attribution pass
split the remaining final/overhead bucket and showed host final-logit softcap
as the largest actionable sub-bucket, typically `3.86-4.32 ms/tok` with one
network row at `4.84 ms/tok`. The accepted candidate applies the configured
final-logit softcap to `d_logits` on GPU before D2H under
`KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`, then skips the CPU softcap loop only for
that gated Gemma graph-final path. The gate is runtime-derived from
`do_final`, `graph.kv_format == 9`, `Gemma4MoE`, finite positive
`final_logit_softcap`, and runtime `vocab_size`; no model/GPU/vocab hardcodes
were added. Calibration, HCS, graph replay mechanics, prefill, non-Gemma
behavior, and default behavior without the env flag were left unchanged.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Focused attribution build attempt | `./dev build` | n/a | n/a | Build failed after adding final/sync attribution because the report used an out-of-scope variable, `avg_sync_final` (`error[E0425]`). | fixed instrumentation bug | [full log](20260614_0948_gemma4_hqq4_k4v4_final_attr_build.log) |
| Focused attribution build2 | `./dev build` | n/a | n/a | Build passed after deriving final-sync timing from `graph.t_graph_final_sync_wait`; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=127`. | proceed to focused attribution | [full log](20260614_0952_gemma4_hqq4_k4v4_final_attr_build2.log) |
| Focused final/sync attribution | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 KRASIS_DECODE_GQA_OTHER_CLOCKS=1 KRASIS_DECODE_GQA_HD256_ATTN_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`; summary `5626.1` prefill, `58.90` internal decode, `119.52` HTTP. HCS stayed clean at `3840/3840`, min free `11464 MB`, zero cold DMA/DMA calls, and `copy_failures=0`. Final split showed final graph segment about `1.05 ms/tok`: final RMSNorm `0.01`, LM head `0.92`, graph residual `0.12`, D2H logits `0.12-0.13`, and host softcap generally `3.86-4.32 ms/tok` with one `4.84` row. | selected GPU softcap candidate | [full log](20260614_0955_gemma4_hqq4_k4v4_final_attr_timing.log) |
| Candidate build | `./dev build` | n/a | n/a | Build passed for `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=136`. | proceed to timing gate | [full log](20260614_0952_gemma4_hqq4_k4v4_final_softcap_gpu_build.log) |
| Timing gate | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_CLOCKS=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 KRASIS_DECODE_GQA_OTHER_CLOCKS=1 KRASIS_DECODE_GQA_HD256_ATTN_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`; summary `5618.5` prefill, `77.84` internal decode, `157.43` HTTP. HCS stayed clean at `3840/3840`, min free `11464 MB`, representative `240/0`, zero cold DMA/DMA calls, and `copy_failures=0`. Final split confirmed the target movement: host softcap dropped to `0.00 ms/tok`, GPU softcap rounded to `0.00 ms/tok`, D2H stayed about `0.13 ms/tok`, final graph segment stayed about `1.05 ms/tok`, and total internal decode improved versus focused timing baseline `58.90 -> 77.84 tok/s`. | passed timing gate; run timing-off speed and QCN guard | [full log](20260614_0958_gemma4_hqq4_k4v4_final_softcap_gpu_timing.log) |
| Timing-off Gemma speed | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`; summary `4931.0` prefill, `92.31` internal decode, `160.16` HTTP. HCS stayed clean at `3840/3840`, min free `11474 MB`, representative `240/0`, zero cold DMA/DMA calls, and `copy_failures=0`. Decode improved versus accepted experimental dual-norm speed baseline `66.96 -> 92.31 tok/s`; prefill was lower in this run even though the candidate does not touch prefill and the timing gate still measured `5618.5`, so the prefill discrepancy is recorded for follow-up rather than attributed to the softcap change. | accepted candidate as env-gated decode win | [full log](20260614_1004_gemma4_hqq4_k4v4_final_softcap_gpu_speed.log) |
| QCN guard | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 ./dev speed-test` | HQQ4 | k4v4 | Fixed QCN speed-test passed benchmark guard with `6685.1` prefill, `90.07` internal decode, `198.82` HTTP. HCS coverage `15957/24576` (`64.9%`), min free `896 MB`, above the `600 MB` safety margin; `copy_failures=0`. The new softcap gate is neutral for non-Gemma and remains inactive on QCN. | QCN guard passed | [full log](20260614_1011_qcn_speed_test_after_final_softcap_gpu.log) |

Notes:
- The metric gate was recorded before optimization edits in
  `krasis-internal/DEBUGLOG.md`.
- This is accepted behind `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` and the active
  experimental baseline `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1`; default runs
  remain unchanged unless the env flag is enabled.
- Current accepted experimental Gemma k4 marker with both flags enabled is
  `4931.0/92.31/160.16` from the timing-off run. For decode-only comparison,
  this is a large improvement from `5626.1/66.96/120.00`; the prefill variance
  needs a separate follow-up if prefill optimization is resumed.

## Rejected - 2026-06-14 (Gemma4 HQQ4/k4v4 MoE W2 FP32 shared-activation candidate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Baseline experimental flag `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` remained
enabled. The candidate targeted the measured MoE activation/W2 bucket after
the fresh broad timing attribution showed fused activation+W2 at
`1.67-1.69 ms/tok`, with preload sub-buckets load `~0.34`, SiLU/multiply
`~0.64`, and shared store `~0.21 ms/tok`. Weighted expert accumulation
remained below threshold at `0.06-0.08 ms/tok`. The candidate kept one W2
output tile per block, did not retry paired-output W2, and used an env-gated
FP32 shared activation buffer under `KRASIS_DECODE_MOE_W2_F32_ACT=1`.
Calibration, HCS, graph capture, prefill, final segment behavior, and
non-Gemma behavior were left unchanged.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Candidate build | `./dev build` | n/a | n/a | Build passed for the initial FP32 shared-activation candidate; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=137` | attempted timing gate | [full log](20260614_0909_gemma4_hqq4_k4v4_moe_w2_f32_act_build.log) |
| Initial timing attempt | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_MOE_W2_F32_ACT=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 KRASIS_DECODE_GQA_OTHER_CLOCKS=1 KRASIS_DECODE_GQA_HD256_ATTN_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Server failed before benchmark because the new PTX symbols were compiled and embedded but not added to the explicit `KERNEL_NAMES` load list: `RuntimeError: Kernel 'fused_silu_w2_batched_f32_act' not found`. | integration issue fixed before valid timing | [full log](20260614_0912_gemma4_hqq4_k4v4_moe_w2_f32_act_timing.log) |
| Candidate build after export-list fix | `./dev build` | n/a | n/a | Build passed after adding the candidate kernels to `KERNEL_NAMES`; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=129` | proceed to valid timing gate | [full log](20260614_0917_gemma4_hqq4_k4v4_moe_w2_f32_act_build2.log) |
| Timing gate | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_MOE_W2_F32_ACT=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 KRASIS_DECODE_GQA_OTHER_CLOCKS=1 KRASIS_DECODE_GQA_HD256_ATTN_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`, but failed the full metric gate: summary `4683.2` prefill, `58.41` internal decode, `99.72` HTTP versus fresh broad-clock baseline `4731.8/59.29/119.22`. HCS stayed clean at `3840/3840`, min free `11464 MB`, representative `240/0`, zero cold DMA/DMA calls, and `copy_failures=0`. Target fused activation+W2 improved from `1.69/1.67/1.69/1.67` to about `1.54/1.54/1.54/1.53 ms/tok`; internal activation+W2 improved from about `1.62` to `1.47-1.48`. Preload composition shifted to load `~0.44`, SiLU/multiply `~0.46-0.47`, store `~0.21-0.22`, sync `~0.06`. Total internal decode regressed, so the candidate was rejected. | rejected at timing gate; no timing-off Gemma speed or QCN guard | [full log](20260614_0920_gemma4_hqq4_k4v4_moe_w2_f32_act_timing2.log) |
| Restore accepted source after rejection | `./dev build` | n/a | n/a | Build passed after reverting only the FP32 shared-activation candidate; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=135`; `KRASIS_DECODE_MOE_W2_F32_ACT` and `fused_silu_w2_batched_f32_act*` symbols are absent | restore/build hygiene | [full log](20260614_0927_restore_after_moe_w2_f32_act_reject_build.log) |

Notes:
- The metric gate was recorded before editing in
  `krasis-internal/DEBUGLOG.md`.
- The candidate was Gemma MoE relevant, env-gated, dimension-derived from
  runtime `topk`, `intermediate`, `expert_hs`, and shared-memory capacity, and
  did not disable calibration/HCS/graph capture or change non-Gemma behavior.
- Rejection reason: the measured W2 bucket improved locally, but total
  internal decode regressed under the same broad attribution clock set, so the
  accepted experimental baseline remains
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1`.

## Rejected - 2026-06-14 (Gemma4 HQQ4/k4v4 GQA fused-QKV split-elision candidate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Baseline experimental flag `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` remained
enabled. A fresh broad timing-attribution pass was run before editing. It
showed weighted expert accumulation still only `0.06-0.08 ms/tok`, so no
weighted-add candidate was attempted under the requested `>0.2 ms/tok` upside
rule. Excluding paired W2, W13 direct, dense gate+up fusion, dense-down custom
GEMV, and rejected GQA tile-cap/score-cache/HD256-specialization variants, the
candidate targeted the remaining non-active GQA projection bucket by eliding
the fused-QKV K/V split copies under `KRASIS_DECODE_GQA_FUSED_QKV_SPLIT_ELIDE=1`.
Calibration, HCS, graph capture, prefill, final segment behavior, and
non-Gemma behavior were left unchanged.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Fresh timing attribution | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 KRASIS_DECODE_GQA_OTHER_CLOCKS=1 KRASIS_DECODE_GQA_HD256_ATTN_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`; summary `4731.8` prefill, `59.29` internal decode, `119.22` HTTP. HCS stayed clean at `3840/3840`, min free `11464 MB`, representative `240/0` hit/miss, zero cold DMA/DMA calls, and `copy_failures=0`. Fresh 49/99/249/511 rows: non-active GQA projection `0.95/0.93/0.95/0.95 ms/tok`, non-active endpoint `2.44/2.53/2.96/3.36`, sliding HD256 attention `0.70/0.86/1.22/1.61`, and weighted expert accumulation `0.08/0.06/0.08/0.06`. | select one candidate from fresh data | [full log](20260614_0842_gemma4_hqq4_k4v4_dual_norm_fresh_attr_timing.log) |
| Candidate build | `./dev build` | n/a | n/a | Build passed for the env-gated fused-QKV split-elision candidate; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=127` | proceed to timing gate | [full log](20260614_0852_gemma4_hqq4_k4v4_gqa_fused_qkv_elide_build.log) |
| Timing gate | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_GQA_FUSED_QKV_SPLIT_ELIDE=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 KRASIS_DECODE_GQA_OTHER_CLOCKS=1 KRASIS_DECODE_GQA_HD256_ATTN_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`, but failed the metric gate: summary `5605.5` prefill, `58.56` internal decode, `110.82` HTTP versus fresh broad-clock baseline `4731.8/59.29/119.22`. HCS stayed clean at `3840/3840`, min free `11464 MB`, representative `240/0`, zero cold DMA/DMA calls, and `copy_failures=0`. The target projection bucket did not improve: candidate `0.95/0.95/0.95/0.93 ms/tok` versus baseline `0.95/0.93/0.95/0.95`; non-active endpoint regressed to `2.44/2.59/3.04/3.56`, and total internal decode regressed. | rejected at timing gate; no timing-off Gemma speed or QCN guard | [full log](20260614_0855_gemma4_hqq4_k4v4_gqa_fused_qkv_elide_timing.log) |
| Restore accepted source after rejection | `./dev build` | n/a | n/a | Build passed after reverting only the fused-QKV split-elision candidate; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=127`; `KRASIS_DECODE_GQA_FUSED_QKV_SPLIT_ELIDE` and split-elision work-pointer symbols are absent | restore/build hygiene | [full log](20260614_0902_restore_after_gqa_fused_qkv_elide_reject_build.log) |

Notes:
- The metric gate was recorded before editing in
  `krasis-internal/DEBUGLOG.md`.
- Periodic checks: the candidate was Gemma GQA relevant, env-gated,
  dimension-derived from runtime fused-QKV descriptor rows, and did not disable
  calibration/HCS/graph capture or change non-Gemma behavior.
- Rejection reason: the measured target bucket did not move and total internal
  decode regressed, so the accepted experimental baseline remains
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1`.

## Rejected - 2026-06-14 (Gemma4 HQQ4/k4v4 MoE W13 direct-BF16 candidate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Baseline experimental flag `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` remained
enabled. The candidate targeted the measured MoE W13 bucket by adding an
env-gated direct-BF16 W13 graph path for runtime `w13_ksplits_batched == 1`,
skipping the one-split `reduce_ksplits_bf16_batched` launch. It avoided paired
W2, dense gate+up fusion, dense-down custom GEMV, GQA tile-cap, score-cache,
and HD256 specialization. Calibration, HCS, graph capture structure, prefill,
final segment behavior, and non-Gemma behavior were left unchanged.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Candidate build | `./dev build` | n/a | n/a | Build passed for the env-gated W13 direct-BF16 candidate; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=136` | proceed to timing gate | [full log](20260614_0827_gemma4_hqq4_k4v4_moe_w13_direct_build.log) |
| Timing gate | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_MOE_W13_DIRECT=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`, but failed the metric gate: summary `5626.1` prefill, `62.16` internal decode, `112.01` HTTP versus accepted dual-norm timing baseline `5628.2/64.75/112.01`. HCS stayed clean at `3840/3840`, min free `11466 MB`, representative `240/0` hit/miss, zero cold DMA/DMA calls, and `copy_failures=0`. The target W13+reduce rows stayed essentially unchanged at about `1.15-1.18 ms/tok` (`1.11+0.07`, `1.11+0.07`, `1.11+0.07`; 511 row `1.09+0.06`), and total internal decode regressed. | rejected at timing gate; no timing-off Gemma speed or QCN guard | [full log](20260614_0830_gemma4_hqq4_k4v4_moe_w13_direct_timing.log) |
| Restore accepted source after rejection | `./dev build` | n/a | n/a | Build passed after reverting only the W13 direct-BF16 candidate; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=136`; `KRASIS_DECODE_MOE_W13_DIRECT` and `marlin_gemv_int4_batched_direct_bf16` symbols are absent | restore/build hygiene | [full log](20260614_0836_restore_after_moe_w13_direct_reject_build.log) |

Notes:
- The metric gate was recorded before editing in
  `krasis-internal/DEBUGLOG.md`.
- Periodic checks: the candidate was Gemma MoE relevant, did not disable
  calibration/HCS/graph capture, did not change non-Gemma behavior, and used
  runtime dimensions/shared-memory capacity rather than model/GPU hardcodes.
- Rejection reason: removing the explicit one-split reduce launch did not
  reduce the measured W13+reduce bucket and regressed total internal decode, so
  the accepted experimental baseline remains
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1`.

## Rejected - 2026-06-14 (Gemma4 HQQ4/k4v4 route-prep dense-down custom BF16 GEMV candidate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Baseline experimental flag `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` remained
enabled. The candidate replaced only the Gemma route-prep dense-down BF16
GEMV with an env-gated custom BF16 GEMV under
`KRASIS_DECODE_ROUTE_PREP_DENSE_DOWN_CUSTOM=1`. It avoided the rejected dense
gate+up fusion path, paired-W2 work, GQA tile-cap, score-cache, and HD256
specialization. Calibration, HCS, graph capture, prefill, final segment
behavior, and non-Gemma behavior were left unchanged.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Candidate build | `./dev build` | n/a | n/a | Build passed for the env-gated dense-down custom BF16 GEMV candidate; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=137` | proceed to timing gate | [full log](20260614_0806_gemma4_hqq4_k4v4_route_dense_down_custom_build.log) |
| Timing gate | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_ROUTE_PREP_DENSE_DOWN_CUSTOM=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`, but failed the full metric gate: summary `5622.1` prefill, `62.12` internal decode, `112.43` HTTP versus accepted dual-norm timing baseline `5628.2/64.75/112.01`. HCS stayed clean at `3840/3840`, min free `11466 MB`, representative `240/0` hit/miss, zero cold DMA/DMA calls, and `copy_failures=0`. The dense-down sub-bucket improved from baseline `0.37/0.36/0.37 ms/tok` to `0.31/0.31/0.30`, but route-prep stayed about `2.55-2.56 ms/tok` on the internal rows and total internal decode regressed. | rejected at timing gate; no timing-off Gemma speed or QCN guard | [full log](20260614_0812_gemma4_hqq4_k4v4_route_dense_down_custom_timing.log) |
| Restore accepted source after rejection | `./dev build` | n/a | n/a | Build passed after reverting only the dense-down custom GEMV candidate; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=136`; `KRASIS_DECODE_ROUTE_PREP_DENSE_DOWN_CUSTOM` and `gemv_bf16_weight_bf16_input_bf16_warp8` symbols are absent | restore/build hygiene | [full log](20260614_0830_restore_after_dense_down_custom_reject_build.log) |

Notes:
- The metric gate was recorded before editing in
  `krasis-internal/DEBUGLOG.md`.
- Periodic checks: the candidate was Gemma route-prep relevant, did not disable
  calibration/HCS/graph capture, did not change non-Gemma behavior, and used
  runtime registered BF16 weight dimensions rather than model/GPU hardcodes.
- Rejection reason: the local dense-down sub-bucket improved, but the full
  timing gate requires no internal decode regression; the accepted experimental
  baseline remains `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1`.

## Rejected - 2026-06-14 (Gemma4 HQQ4/k4v4 HD256 GQA specialized single-kernel candidate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma.
Baseline experimental flag `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` remained
enabled. The candidate added a separate HD256 k4v4 single-kernel GQA
specialization under `KRASIS_DECODE_GQA_HD256_SPECIALIZED=1`; it did not use
the rejected tiled tile-cap path, the rejected score-cache path, or any W2
paired work. Calibration, HCS, graph capture, prefill, final segment behavior,
and non-Gemma paths were left unchanged.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Candidate build | `./dev build` | n/a | n/a | Build passed for the env-gated HD256 GQA specialization; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=138` | proceed to timing gate | [full log](20260614_0748_gemma4_hqq4_k4v4_hd256_spec_build.log) |
| Timing gate | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_GQA_HD256_SPECIALIZED=1 KRASIS_DECODE_GQA_HD256_ATTN_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed correctness `14/14`, `ALL TESTS PASSED`, but failed the performance gate: summary `5616.7` prefill, `61.87` internal decode, `110.76` HTTP versus the accepted dual-norm timing baseline `5628.2/64.75/112.01`. HCS stayed clean at `3840/3840`, min free `11464 MB`, representative `240/0` hit/miss, zero cold DMA/DMA calls, and `copy_failures=0`. Representative internal decode GQA path rows showed no consistent improvement: baseline `2.97/3.35/3.89 ms/tok` versus candidate `3.22/3.33/3.97`; the 511-token row stayed high (`4.50` vs baseline `4.66`) and total decode regressed. | rejected at timing gate; no timing-off Gemma speed or QCN guard | [full log](20260614_0752_gemma4_hqq4_k4v4_hd256_spec_timing.log) |
| Restore accepted source after rejection | `./dev build` | n/a | n/a | Build passed after reverting only the HD256 specialized-kernel candidate; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=136`; `KRASIS_DECODE_GQA_HD256_SPECIALIZED` and `gqa_attention_k4v4_hd256_single_g*` symbols are absent | restore/build hygiene | [full log](20260614_0807_restore_after_hd256_spec_reject_build.log) |

Notes:
- The metric gate was recorded before editing in
  `krasis-internal/DEBUGLOG.md`.
- The candidate was runtime/config gated by graph route segment, Gemma4 MoE
  graph presence, k4v4, `head_dim == 256`, sliding GQA, gated GQA, and active
  shared-memory capacity. It was rejected because specializing the dynamic
  head-dim path did not improve the measured GQA bucket and reduced total
  internal decode speed.

## Accepted - 2026-06-14 (Gemma4 HQQ4/k4v4 route-prep dual residual RMSNorm candidate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma/QCN.
The candidate is env-gated by `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and
runtime-gated to Gemma4 graph route-prep only. It fuses the two residual
RMSNorm passes that share `d_residual` and the same RMS denominator:
`pre_ffn_norm2` into the BF16 scratch path and router-input RMSNorm+scale into
`d_hidden`. It does not change calibration, HCS policy, graph capture,
prefill, final segment behavior, dense gate+up fusion, W2 paired-tile code, or
QCN/non-Gemma behavior.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Candidate build | `./dev build` | n/a | n/a | Build passed for the env-gated route-prep dual residual RMSNorm candidate; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=135` | proceed to timing gate | [full log](20260614_0720_gemma4_hqq4_k4v4_route_dual_norm_build.log) |
| Timing gate | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14`, `ALL TESTS PASSED`; benchmark summary `5628.2` prefill, `64.75` internal decode, `112.01` HTTP, HCS `3840/3840`, min free `11466 MB`. HCS stayed clean with representative `240/0` hit/miss, `0` cold DMA/DMA calls, and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. Representative 49/99/249/511 rows, ms/tok: route prep/overhead `2.63/2.41/2.63/2.63` versus baseline `2.82/2.82/2.82/2.53`; dense post norms `0.43/0.41/0.43/0.43`; router norm `0.03/0.02/0.03/0.03`; combined target `0.46/0.43/0.46/0.46` versus baseline dense-post+router `0.65/0.65/0.65/0.60` | timing gate accepted; proceed to timing-off Gemma validation and QCN guard | [full log](20260614_0724_gemma4_hqq4_k4v4_route_dual_norm_timing.log) |
| Timing-off Gemma validation | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | Full run passed `14/14`, `ALL TESTS PASSED`; benchmark summary `5626.1` prefill, `66.96` internal decode, `120.00` round trip. HCS `3840/3840`, min free `11474 MB`, representative HCS `240/0`, zero cold DMA/DMA calls, `copy_failures=0` | accepted as env-gated Gemma route-prep optimization; improves accepted k4 timing-off marker from `5613.8/65.89/118.74` to `5626.1/66.96/120.00` when enabled | [full log](20260614_0737_gemma4_hqq4_k4v4_route_dual_norm_speed.log) |
| QCN guard | `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 ./dev speed-test` | HQQ4 | k4v4 | Standard speed benchmark completed on Qwen3-Coder-Next: `6452.1` prefill, `87.61` internal decode, `149.48` round trip. HCS `15957/24576` loaded (`64.9%`), min free `928 MB`, `copy_failures=0`; runtime HCS pressure stayed above the `600 MB` safety margin | QCN/non-Gemma guard passed; env flag did not route QCN through the Gemma-only dual-norm path | [full log](20260614_0733_qcn_speed_test_after_route_dual_norm.log) |

Notes:
- The metric gate was recorded before editing in
  `krasis-internal/DEBUGLOG.md`.
- The gate is runtime-derived: captured graph route-prep, k4v4 session,
  `Gemma4MoE`, `layer_idx == range_end`, required norm pointers present, and
  hidden-size shared-memory capacity derived from `graph.gqa_max_smem_bytes`.
- This is not the rejected dense gate+up fusion path and not further W2 paired
  work.

## Rejected - 2026-06-14 (Gemma4 HQQ4/k4v4 fused W2 paired-output-tile candidate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma. Timing
instrumentation, MoE route graph clocks, MoE/W2 graph clocks, and
activation/preload detail clocks were enabled for the gate. No timing-off
`./dev speed-test` or QCN guard was run because correctness failed and the
target fused activation+W2 bucket regressed.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Candidate build | `./dev build` | n/a | n/a | Build passed for the decode-only Gemma INT4 fused-W2 paired-output-tile candidate; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=137` | proceed to timing gate | [full log](20260614_0654_gemma4_hqq4_k4v4_w2_pair2_build.log) |
| Timing gate | `KRASIS_DECODE_MOE_W2_PAIRED_TILES=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run failed correctness: `SUMMARY: 13/14 passed, 1 failed`, `multi_turn_t5_all` missing `Biscuit`, `1 TEST(S) FAILED`. Benchmark summary before the failure: `5454.8` prefill, `59.12` internal decode, `97.46` round trip. HCS stayed clean with representative `3840/3840`, `240/0` hit/miss, zero cold DMA/DMA calls, and `copy_failures=0`; decode min free stayed around `11470-11478 MB`. The paired path was active: `paired_tiles:on`, `physical_tiles/seg:88.0`, `blocks/seg:704.0`. Representative 49/99/249/511 rows, ms/tok: fused activation+W2 bucket `2.01/2.01/2.01/1.99`, activation `1.39/1.39/1.39/1.39`, W2 prep `0.01/0.01/0.01/0.01`, W2 GEMV/output `0.55/0.55/0.55/0.55`, internal total `1.94/1.94/1.95/1.94`, gate/up load `0.11/0.11/0.11/0.11`, SiLU/multiply `1.18/1.18/1.18/1.18`, shared store `0.09/0.09/0.09/0.09`, sync `0.02/0.02/0.02/0.02`, repeated block work about `17.9`, blocks/segment `704.0` | rejected at timing gate. The candidate halved physical W2 blocks versus the accepted `1408.0`, but doubled the per-block critical work enough to regress the wall bucket versus the accepted attribution baseline `1.69/1.69/1.69/1.67`, and it also failed accuracy. No timing-off speed or QCN guard | [full log](20260614_0700_gemma4_hqq4_k4v4_w2_pair2_timing.log) |
| Restore accepted source after rejection | `./dev build` | n/a | n/a | Build passed after reverting only the paired-output-tile candidate; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=135`; paired symbols/env gate are absent | restore/build hygiene | [full log](20260614_0710_restore_after_w2_pair2_reject_build.log) |

Notes:
- Candidate gate was runtime-derived and graph-only: `GRAPH_SEG_ROUTE_GQA`,
  `Gemma4MoE`, k4v4, routed gated SiLU INT4 experts, non-latent expert dims,
  `topk/intermediate/hidden` dimension checks, and active shared-memory
  capacity. It did not disable calibration, HCS, or graph capture.
- Reverted only the paired-tile candidate. Diagnostic W2/preload attribution
  remains available under `KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1`.

## Diagnostics - 2026-06-13 (Gemma4 HQQ4/k4v4 fused W2 activation/preload sub-attribution)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma. Timing
instrumentation, MoE route graph clocks, MoE/W2 graph clocks, and
activation/preload detail clocks were enabled. No timing-off speed run or QCN
guard was run because this was attribution only.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Build attempt 1 | `./dev build` | n/a | n/a | Build passed with initial activation/preload detail clocks; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=131` | first timing attempt exposed intrusive per-iteration clocks and was discarded | [full log](20260613_231629_gemma4_hqq4_k4v4_moe_w2_preload_attr_build.log) |
| Timing attempt 1 | `KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14`, but the per-iteration sampling inflated internal activation to `5-6 ms/tok` while the graph-level fused bucket stayed near `1.67-1.69 ms/tok` | invalid instrumentation; corrected to sample one representative activation element per block | [full log](20260613_231855_gemma4_hqq4_k4v4_moe_w2_preload_attr_timing.log) |
| Build attempt 2 | `./dev build` | n/a | n/a | Build passed after reducing sampling overhead; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=135` | proceed to corrected timing attempt | [full log](20260613_232338_gemma4_hqq4_k4v4_moe_w2_preload_attr_build2.log) |
| Timing attempt 2 | same as timing attempt 1 | HQQ4 | k4v4 | Full run passed `14/14`, but internal activation again exceeded the outer fused bucket because the graph clock base still used the old four-slot stride while the buffer used eight slots | invalid instrumentation; fixed clock-base stride | [full log](20260613_232606_gemma4_hqq4_k4v4_moe_w2_preload_attr_timing2.log) |
| Build attempt 3 | `./dev build` | n/a | n/a | Build passed after fixing the graph clock-base stride; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=127` | proceed to valid timing run | [full log](20260613_233023_gemma4_hqq4_k4v4_moe_w2_preload_attr_build3.log) |
| Valid attribution | `KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14`, `ALL TESTS PASSED`; benchmark summary `5602.3` prefill, `59.89` internal decode, `108.08` HTTP, HCS `3840/3840`, min free `11470 MB`. HCS stayed clean with representative `240/0` hit/miss, `0` cold DMA/DMA calls, and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. Representative 49/99/249/511 rows, ms/tok: activation+W2 fused bucket `1.69/1.69/1.69/1.67`, activation `1.20/1.20/1.20/1.20`, W2 prep `0.02/0.02/0.02/0.02`, W2 GEMV/output `0.41/0.41/0.41/0.41`, internal total `1.63/1.63/1.63/1.63`, weighted accumulation `0.08/0.08/0.08/0.06`, graph/launch overhead `0.06/0.06/0.06/0.05`. Activation/preload detail: gate/up load `0.33/0.33/0.33/0.33`, SiLU/multiply `0.65/0.65/0.65/0.65`, shared store `0.21/0.21/0.21/0.21`, sync `0.04/0.04/0.04/0.04`, residual `0.00`, repeated block work `53.87/53.88/53.94/53.94`, blocks/segment `1408.0` | attribution only. Largest wall-time sub-bucket is SiLU/multiply; repeated block work confirms activation/preload is repeated across every output-tile block. Exactly one next candidate selected: decode-only Gemma INT4 fused-W2 paired-output-tile block variant for graph MoE, gated by runtime hidden/output-tile count, top-k, intermediate size, and active shared-memory/register constraints, to reuse the block-local activated vector across two adjacent W2 output tiles without global activation precompute/reuse | [full log](20260613_233253_gemma4_hqq4_k4v4_moe_w2_preload_attr_timing3.log) |

Notes:
- New detail clocks are diagnostic-only under
  `KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1` and only activate with the existing
  Gemma k4v4 graph MoE/W2 timing gate.
- Normal decode without timing stays on the accepted `fused_silu_w2_batched`
  path. This pass did not change prefill, HCS policy, final segment behavior,
  QCN/non-Gemma paths, rejected dense gate+up paths, rejected GQA candidates, or
  output math.

## Rejected - 2026-06-13 (Gemma4 HQQ4/k4v4 activation precompute/reuse candidate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma. Timing
instrumentation and MoE route/W2 graph clocks were enabled for the gate. No
timing-off speed run or QCN guard was run because the target activation+W2
bucket did not improve.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Candidate build | `./dev build` | n/a | n/a | Build passed for the decode-only Gemma INT4 expert activation precompute/reuse candidate; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=137` | proceed to timing gate | [full log](20260613_225526_gemma4_hqq4_k4v4_act_precompute_build.log) |
| Timing gate | `KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; benchmark summary `5165.6` prefill, `64.24` internal decode, `109.83` HTTP, HCS `3840/3840`, min free `11470 MB`. HCS stayed clean with representative `240/0` hit/miss, `0` cold DMA/DMA calls, and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. Representative 49/99/249/511 rows, ms/tok: activation+W2 bucket `1.48/1.51/1.49/1.50`, activation `0.01/0.01/0.01/0.01`, W2 prep `1.07/1.08/1.08/1.08`, W2 GEMV/output `0.35/0.35/0.35/0.35`, internal total `1.44/1.45/1.44/1.44`, weighted accumulation `0.06/0.08/0.07/0.07`, graph/launch overhead `0.04/0.06/0.05/0.05` | rejected at timing gate. Kernel inspection correctly showed `SiLU(gate)*up` is recomputed once per output tile in `fused_silu_w2_batched`, but precomputing it once into global memory moved the measured cost into repeated W2 input preload/prep and did not reduce the full activation+W2 interval versus the accepted attribution marker `1.48/1.48/1.48/1.46`. No timing-off speed or QCN guard | [full log](20260613_225804_gemma4_hqq4_k4v4_act_precompute_timing.log) |
| Restore accepted source after rejection | `./dev build` | n/a | n/a | Build passed after reverting only the activation precompute/reuse candidate; candidate symbols are absent; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=135` | restore/build hygiene | [full log](20260613_230519_restore_after_act_precompute_reject_build.log) |

Notes:
- Verified before editing from the actual kernel: `fused_silu_w2_batched` is
  launched as `(ceil(expert_hs / 16), 1, topk)`, and every
  `(output tile, expert)` block recomputes the full activated vector into
  block-local shared memory.
- Candidate gate was Gemma graph MoE only: k4v4 `GRAPH_SEG_ROUTE_GQA`,
  `Gemma4MoE`, gated SiLU INT4 experts, non-latent expert dimensions,
  runtime `topk/intermediate/hidden` checks, and existing `d_batch_gate_ups`
  scratch capacity. It did not alter prefill, HCS policy, final segment
  behavior, QCN/non-Gemma paths, rejected dense gate+up paths, GQA candidates,
  or output math.

## Diagnostics - 2026-06-13 (Gemma4 HQQ4/k4v4 MoE activation/W2 attribution)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma. Timing
instrumentation and graph clock markers were enabled for attribution; no
timing-off speed run or QCN guard was run.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Diagnostic build attempt 1 | `./dev build` | n/a | n/a | Build failed with Rust `E0277` because the debug timed fused-W2 kernel had 13 launch parameters, exceeding cudarc's tuple `LaunchAsync` implementation | fixed diagnostic launch to use the existing raw pointer-vector launch form | [full log](20260613_223530_gemma4_hqq4_k4v4_moe_w2_attr_build.log) |
| Diagnostic build attempt 2 | `./dev build` | n/a | n/a | Build passed with debug-only MoE activation/W2 clocks; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=128` | proceed to attribution run | [full log](20260613_223732_gemma4_hqq4_k4v4_moe_w2_attr_build2.log) |
| Gemma4 HQQ4/k4v4 MoE activation/W2 attribution | `KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; benchmark summary `4915.2` prefill, `62.16` internal decode, `110.99` HTTP, HCS `3840/3840`, min free `11470 MB`. HCS stayed clean with representative `240/0` hit/miss, `0` cold DMA/DMA calls, and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. Representative 49/99/249/511 rows, ms/tok: MoE expert `3.30/3.30/3.31/3.16`, W13 `1.11/1.10/1.11/1.09`, W13 reduce `0.07/0.07/0.07/0.06`, fused activation+W2 bucket `1.48/1.48/1.48/1.46`, activation `1.05/1.06/1.06/1.05`, W2 prep `0.02/0.02/0.02/0.02`, W2 GEMV/output `0.34/0.34/0.34/0.34`, internal total `1.41/1.41/1.42/1.41`, weighted accumulation `0.08/0.08/0.08/0.06`, graph/launch overhead `0.06/0.06/0.06/0.05`, post-output `0.57/0.57/0.57/0.51` | attribution only. The largest proven sub-bucket is the repeated fused activation/preload phase, not W2 GEMV, weighted accumulation, or graph launch overhead. Exactly one next candidate selected: decode-only Gemma INT4 expert activation precompute/reuse path for graph MoE, gated by runtime top-k/intermediate/hidden dimensions and scratch capacity, so SiLU(gate)*up is computed once per routed expert and reused by the W2 GEMV instead of recomputed per output tile | [full log](20260613_224001_gemma4_hqq4_k4v4_moe_w2_attr_timing.log) |

Notes:
- The instrumentation is gated by `KRASIS_DECODE_MOE_W2_CLOCKS=1`, graph
  timing, k4v4 captured graphs containing `Gemma4MoE`, `GRAPH_SEG_ROUTE_GQA`,
  and the runtime gated INT4 expert path. Normal decode stays on
  `fused_silu_w2_batched`.
- The timed kernel records per-block critical-path clocks for start,
  activation end, W2 prep end, and kernel end, then compares internal total
  with the existing outer activation+W2 graph marker.
- No prefill, HCS policy, final segment behavior, QCN/non-Gemma path, rejected
  GQA candidate, dense gate+up candidate, or output math was changed.

## Rejected - 2026-06-13 (Gemma4 HQQ4/k4v4 dense gate+up dual-projection candidate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma. Timing
instrumentation and route-prep/MoE graph clocks were enabled for the gate. No
timing-off speed run or QCN guard was run because correctness failed.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Candidate build attempt 1 | `./dev build` | n/a | n/a | Build failed with Rust `E0133` because the optional cuBLAS symbol lookup needed an explicit unsafe block | fixed candidate wrapper and rebuilt | [full log](20260613_221418_gemma4_hqq4_k4v4_dense_gate_up_build.log) |
| Candidate build attempt 2 | `./dev build` | n/a | n/a | Build passed for the decode-only Gemma route-prep dense gate+up grouped-batched BF16 cuBLAS candidate; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=127` | proceed to timing gate | [full log](20260613_221555_gemma4_hqq4_k4v4_dense_gate_up_build2.log) |
| Timing gate | `KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Failed correctness during warmup graph replay: `sync routing[1]: CUDA_ERROR_ILLEGAL_ADDRESS`, followed by fatal decode/prefill boundary sync error. Partial one-token diagnostic showed HCS `3840/3840`, `240/0` hit/miss, `0` cold DMA/DMA calls, but the CUDA context was unsafe and no full benchmark summary was valid | rejected at correctness gate; no timing-off speed run or QCN guard | [full log](20260613_221837_gemma4_hqq4_k4v4_dense_gate_up_timing.log) |
| Restore accepted source after rejection | `./dev build` | n/a | n/a | Build passed after reverting only the grouped-batched dense gate+up candidate; candidate symbols are absent; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=127` | restore/build hygiene | [full log](20260613_222310_restore_after_dense_gate_up_reject_build.log) |

Notes:
- Candidate gate was captured graph route-prep only: `GRAPH_SEG_ROUTE_GQA`,
  `layer_idx == range_end`, `Gemma4MoE`, k4v4, BF16 dense gate/up weights with
  matching runtime geometry, shared BF16 `d_hidden` input, and existing
  adjacent `[gate | up]` output layout in `d_expert_gate_up`.
- The candidate did not alter prefill, HCS policy, final segment behavior,
  QCN/non-Gemma paths, rejected GQA candidates, or output layout. It was
  reverted because graph replay correctness failed.

## Diagnostics - 2026-06-13 (Gemma4 HQQ4/k4v4 route-prep sub-split attribution)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma. Timing
instrumentation and graph clock markers were enabled for attribution; no
timing-off speed run or QCN guard was run.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Diagnostic build | `./dev build` | n/a | n/a | Build passed with debug-only Gemma route-prep graph clocks; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=127` | proceed to attribution run | [full log](20260613_215750_gemma4_hqq4_k4v4_route_prep_attr_build.log) |
| Gemma4 HQQ4/k4v4 route-prep attribution | `KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; benchmark summary `5611.2` prefill, `63.87` internal decode, `109.14` HTTP, HCS `3840/3840`, min free `11464 MB`. HCS stayed clean with representative `240/0` hit/miss, `0` cold DMA/DMA calls, and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. Representative 49/99/249/511 rows, ms/tok: route prep/overhead `2.82/2.82/2.82/2.53`, pre-GQA norm `0.26/0.26/0.26/0.23`, post-GQA norm/add `0.26/0.26/0.26/0.24`, dense pre-norm `0.26/0.26/0.26/0.23`, dense gate `0.35/0.35/0.35/0.33`, dense up `0.35/0.35/0.35/0.33`, dense activation `0.06/0.06/0.06/0.05`, dense down `0.38/0.38/0.37/0.36`, dense post norms `0.42/0.42/0.42/0.39`, router-input norm `0.23/0.23/0.23/0.21`, remaining debug/graph overhead `0.25/0.25/0.25/0.17` | attribution only. The dense projection group is the largest proven route-prep target: gate+up+down are about `1.08 ms/tok` on 49/99 rows and `1.02 ms/tok` on the 511 row. Exactly one next candidate selected: decode-only Gemma dense gate+up dual-projection candidate for the route-prep graph path, preserving output math and leaving prefill, HCS, final, QCN/non-Gemma paths, and rejected GQA candidates untouched | [full log](20260613_220014_gemma4_hqq4_k4v4_route_prep_attr_timing.log) |

Notes:
- The instrumentation is gated by `KRASIS_DECODE_ROUTE_PREP_CLOCKS=1`,
  graph timing, k4v4 captured graphs containing `Gemma4MoE`, and target
  `GRAPH_SEG_ROUTE_GQA` layers.
- The route-prep flag forces the parent MoE/route clocks on so the sub-split
  shares the same route-prep denominator. Extra marker kernels mean this run is
  diagnostic only and should not be treated as a timing-off speed result.

## Diagnostics - 2026-06-13 (Gemma4 HQQ4/k4v4 MoE and route attribution)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma. Timing
instrumentation and graph clock markers were enabled for attribution; no
timing-off speed run or QCN guard was run.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Diagnostic build | `./dev build` | n/a | n/a | Build passed with debug-only MoE/route graph clocks; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=125` | proceed to attribution run | [full log](20260613_214141_gemma4_hqq4_k4v4_moe_route_attr_build.log) |
| Gemma4 HQQ4/k4v4 MoE/route attribution | `KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; benchmark summary `5598.9` prefill, `62.57` internal decode, `114.27` HTTP, HCS `3840/3840`, min free `11470 MB`. HCS stayed clean with representative `240/0` hit/miss, `0` cold DMA/DMA calls, and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. Representative 49/99/249/511 rows, ms/tok: MoE expert `3.49/3.49/3.49/3.41`, GQA path `3.09/3.35/3.94/4.63`, route/topk `3.09/3.09/3.09/2.96`. MoE split: W13 `1.11/1.11/1.11/1.09`, W13 reduce `0.07/0.07/0.07/0.06`, activation+W2 `1.66/1.66/1.66/1.65`, weighted output `0.08/0.08/0.08/0.07`, post/output `0.57/0.57/0.57/0.53`. Route split: logits `0.15/0.15/0.15/0.14`, top-k `0.31/0.31/0.31/0.30`, scale/classify `0.13/0.13/0.13/0.12`, prep/overhead `2.49/2.49/2.49/2.39`, CPU classify `0.01` | attribution only. Actual router logits/top-k/classify are small; the largest unresolved non-GQA bucket is route prep/overhead at about `2.4-2.5 ms/tok`, followed by MoE activation+W2 at about `1.65 ms/tok` and W13 at about `1.1 ms/tok`. Exactly one next candidate selected: decode-only Gemma route-prep sub-split attribution, separating pre-GQA norm/residual, post-GQA norm/add, dense gate/up/down, and router-input RMSNorm before any optimization | [full log](20260613_214409_gemma4_hqq4_k4v4_moe_route_attr_timing.log) |

Notes:
- The instrumentation is gated by `KRASIS_DECODE_MOE_ROUTE_CLOCKS=1`,
  graph timing, k4v4, `GRAPH_SEG_ROUTE_GQA`, and `Gemma4MoE` layers.
- No prefill, HCS policy, final segment behavior, QCN/non-Gemma path, output
  math, or speed-path behavior was changed.

## Rejected - 2026-06-13 (Gemma4 HQQ4/k4v4 sliding HD256 shared-score/cache candidate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma. Timing
instrumentation and HD256 attribution clocks were enabled for the gate; no
timing-off speed run or QCN guard was run because the target bucket did not
improve.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Candidate build | `./dev build` | n/a | n/a | Build passed for the decode-only sliding HD256 k4v4 shared-score/cache candidate; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=135` | proceed to timing gate | [full log](20260613_211248_gemma4_hqq4_k4v4_hd256_score_cache_build.log) |
| Initial timing gate | `KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 KRASIS_DECODE_GQA_OTHER_CLOCKS=1 KRASIS_DECODE_GQA_HD256_ATTN_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14`, `ALL TESTS PASSED`; summary `5338.5` prefill, `63.74` internal decode, `114.87` HTTP, HCS `3840/3840`, min free `11472 MB` | gate wiring corrected because the candidate shared-memory opt-in did not update the runtime capacity used by the gate | [full log](20260613_211550_gemma4_hqq4_k4v4_hd256_score_cache_timing.log) |
| Corrected timing gate | same timing command | HQQ4 | k4v4 | Full run passed `14/14`, `ALL TESTS PASSED`; summary `5617.2` prefill, `63.46` internal decode, `114.75` HTTP, HCS `3840/3840`, min free `11472 MB`, graph HCS clean with `240/0` hit/miss and zero cold DMA/DMA calls. Representative 49/99/249/511 rows, ms/tok: sliding HD256 attention graph `0.70/0.87/1.28/1.89`, internal `0.64/0.81/1.24/1.84`, score `0.18/0.21/0.32/0.48`, weight+V `0.42/0.57/0.88/1.33`, final `0.04/0.04/0.04/0.04` | rejected at timing gate. Correctness passed, but the required HD256 weight+V bucket did not improve versus the accepted marker `0.42/0.57/0.88/1.33`; internal decode also regressed versus accepted `65.89`. Per gate, no timing-off Gemma speed or QCN guard was run | [full log](20260613_212310_gemma4_hqq4_k4v4_hd256_score_cache_timing2.log) |
| Restore accepted source after rejection | `./dev build` | n/a | n/a | Build passed after reverting only the shared-score/cache candidate; no score-cache symbols remain; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=135` | restore/build hygiene | [full log](20260613_212911_restore_after_hd256_score_cache_reject_build.log) |

Notes:
- Candidate gate was CUDA graph GQA route decode only: `GRAPH_SEG_ROUTE_GQA`,
  `layer_idx == range_end`, k4v4, sliding `head_dim == 256`, gated GQA, and
  shared-score cache memory derived from
  `graph.kv_cache_len_for_layer(layer_idx)` and active GPU shared-memory
  capacity.
- The candidate preserved HCS state and correctness but did not move the
  measured target bucket, so it was reverted.

## Diagnostics - 2026-06-13 (Gemma4 HQQ4/k4v4 sliding HD256 single-kernel attention attribution)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma. Timing
instrumentation and graph clock markers were enabled for attribution; no
timing-off speed run or QCN guard was run.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Diagnostic build | `./dev build` | n/a | n/a | Build passed with debug-only sliding HD256 k4v4 single-kernel attention clocks; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=126` | proceed to corrected attribution run | [full log](20260613_205442_gemma4_hqq4_k4v4_hd256_attn_attr_build5.log) |
| Gemma4 HQQ4/k4v4 sliding HD256 attention attribution | `KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 KRASIS_DECODE_GQA_OTHER_CLOCKS=1 KRASIS_DECODE_GQA_HD256_ATTN_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; benchmark summary `5605.5` prefill, `63.80` internal decode, `115.13` HTTP, HCS `3840/3840`, min free `11472 MB`. HCS stayed clean with representative `240/0` hit/miss, `0` cold DMA/DMA calls, and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. Representative 49/99/249/511 rows, ms/tok: sliding HD256 attention graph interval `0.70/0.87/1.28/1.89`, internal kernel `0.64/0.81/1.24/1.84`, graph/marker overhead `0.06/0.06/0.04/0.05`, score/max `0.18/0.21/0.32/0.48`, weight+V accumulation `0.42/0.57/0.88/1.33`, final reduce/output `0.04/0.04/0.04/0.04`. Runtime tiles averaged `1.00/1.00/1.22/1.63` against allocated graph max `59.00`; diagnostic inactive-block count was `928.0/928.0/924.4/918.0` for the rejected tiled shape | attribution only. The current accepted hot path is `gqa_attention_k4v4_single_g`; there is no separate HD256 reduce launch. The rejected tile-cap branch targeted the same live non-active HD256 route layers, but it replaced an already single-kernel path with tiled graph work. The measured row-growing cost is inside the current single kernel, dominated by weight+V accumulation; launch/graph overhead and final reduce/output are small. Exactly one next candidate selected: decode-only HD256 k4v4 single-kernel shared-score/cache variant for sliding GQA, gated by runtime/config max attention length and active-GPU shared-memory capacity | [full log](20260613_205709_gemma4_hqq4_k4v4_hd256_attn_attr_timing4.log) |

Notes:
- Earlier build/timing attempts in this pass are retained as raw logs:
  `20260613_203134_gemma4_hqq4_k4v4_hd256_attn_attr_build.log` failed during
  instrumentation wiring; `20260613_203609...timing.log` and
  `20260613_204312...timing2.log` passed but the gate was too narrow to emit
  the requested split; `20260613_205021...timing3.log` emitted the split but
  used an invalid cumulative graph interval. The table above uses the corrected
  `timing4` run only.
- The inactive-block diagnostic is a tiled-branch shape estimate, not current
  accepted-path wasted work. Accepted source remains single-kernel for sliding
  HD256 k4v4 attention.

## Rejected - 2026-06-13 (Gemma4 HQQ4/k4v4 sliding HD256 tile-cap candidate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma. Timing
instrumentation and non-active GQA graph clocks were enabled for the gate; no
timing-off speed run or QCN guard was run because the candidate failed the
target-bucket timing gate.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Candidate build | `./dev build` | n/a | n/a | Build passed for the decode-only sliding HD256 k4v4 tile-cap graph attention candidate; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=125` | proceed to timing gate | [full log](20260613_201050_gemma4_hqq4_k4v4_hd256_tilecap_build.log) |
| Gemma4 HQQ4/k4v4 HD256 tile-cap timing gate | `KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 KRASIS_DECODE_GQA_OTHER_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; benchmark summary `5616.5` prefill, `63.97` internal decode, `114.51` HTTP, HCS `3840/3840`, min free `11472 MB`. HCS stayed clean with representative `240/0` hit/miss, `0` cold DMA/DMA calls, and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. Non-active attention/reduce across representative 49/99/249/511 rows was `0.76/0.96/1.46/2.16 ms/tok`, versus prior marker `0.75/0.95/1.47/2.16` | rejected at timing gate. Correctness passed, but the target non-active attention/reduce bucket did not improve. Per gate, no timing-off Gemma speed or QCN guard was run | [full log](20260613_201352_gemma4_hqq4_k4v4_hd256_tilecap_timing.log) |
| Restore accepted source after rejection | `./dev build` | n/a | n/a | Build passed after reverting only the sliding HD256 tile-cap candidate; candidate symbols/branches are absent; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=124` | restore/build hygiene | [full log](20260613_201820_restore_after_hd256_tilecap_reject_build.log) |

Notes:
- Candidate gate was k4v4 graph decode, route-layer GQA only, sliding
  `head_dim == 256`, tiled buffers present, and `tile_cap > 1` derived from
  `ceil(kv_cache_len_for_layer(layer_idx) / graph.gqa_tile_size)`, clamped to
  the allocated graph max tiles.
- The tiled branch did not reduce the measured sliding attention/reduce
  bucket, so the source was restored to the prior accepted graph dispatch.

## Diagnostics - 2026-06-13 (Gemma4 HQQ4/k4v4 non-active GQA geometry attribution)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma. Timing
instrumentation and graph clock markers were enabled for attribution; no
timing-off speed run or QCN guard was run.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Diagnostic build for non-active GQA split | `./dev build` | n/a | n/a | Build passed with non-active GQA graph clock markers added to the existing coverage instrumentation; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=127` | proceed to attribution run | [full log](20260613_195304_gemma4_hqq4_k4v4_other_gqa_attr_build.log) |
| Gemma4 HQQ4/k4v4 non-active GQA geometry attribution | `KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 KRASIS_DECODE_GQA_OTHER_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; benchmark summary `5608.8` prefill, `63.48` internal decode, `114.54` HTTP, HCS `3840/3840`, min free `11472 MB`. HCS stayed clean with representative `240/0` hit/miss, `0` cold DMA/DMA calls, and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. Coverage split across representative 49/99/249/511 rows, ms/tok: mixed GQA total `3.23/3.48/4.10/4.80`, active HD512 span `0.70/0.74/0.85/0.89`, other mixed GQA `2.53/2.73/3.25/3.92`, active endpoint/named markers `0.69/0.73/0.84/0.88`, active internal gap `0.00/0.00/0.00/0.00`, count coverage `17.2%`, active time coverage `21.6/21.4/20.7/18.4%`, named endpoint coverage `100.0%`. Non-active GQA split: projection `0.95/0.95/0.95/0.94`, norm/RoPE/KV `0.26/0.26/0.26/0.25`, attention/reduce `0.75/0.95/1.47/2.16`, BF16 conversion `0.05/0.05/0.05/0.05`, O-input prep `0.02/0.02/0.02/0.02`, O projection `0.45/0.45/0.45/0.45`, endpoint `2.49/2.69/3.21/3.88`, coverage gap `0.05/0.05/0.05/0.04`, `24` non-active segments | attribution only. The remaining mixed-GQA majority is Gemma's sliding GQA geometry (`head_dim=256`, `num_attention_heads=16`, `num_key_value_heads=8`, `sliding_window=1024`, gated k4v4), not hidden active HD512 work. Exactly one next optimization candidate selected: a decode-only sliding HD256 k4v4 graph attention tile-cap specialization using per-layer max attention length, to reduce inactive graph tile blocks in the row-growing attention/reduce bucket without changing math, HCS, prefill, final behavior, or QCN/non-HD512 paths | [full log](20260613_195613_gemma4_hqq4_k4v4_other_gqa_attr_timing.log) |

Notes:
- The active HD512 path is fully covered and remains a minority of mixed GQA
  time. The non-active sliding path accounts for `2.53 -> 3.92 ms/tok` across
  representative rows.
- Within the non-active sliding path, attention/reduce is the row-growing
  component (`0.75 -> 2.16 ms/tok`); projection and O projection are mostly
  fixed at about `0.95` and `0.45 ms/tok`.

## Diagnostics - 2026-06-13 (Gemma4 HQQ4/k4v4 mixed-segment coverage attribution)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma. Timing
instrumentation and graph clock markers were enabled for attribution; no
timing-off speed run or QCN guard was run.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Diagnostic build for coverage split | `./dev build` | n/a | n/a | Build passed with coverage accounting added to existing graph clock markers; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=123` | proceed to attribution run | [full log](20260613_180708_gemma4_hqq4_k4v4_coverage_attr_build.log) |
| Gemma4 HQQ4/k4v4 mixed-segment coverage attribution | `KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; benchmark summary `5602.5` prefill, `64.32` internal decode, `114.32` HTTP, HCS `3840/3840`, min free `11474 MB`. HCS stayed clean with `240/0` hit/miss, `0` cold DMA/DMA calls, and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. Coverage split across representative 49/99/249/511 rows, ms/tok: mixed GQA total `3.08/3.24/3.96/4.41`, active HD512 mixed span `0.70/0.72/0.86/0.95`, other mixed GQA `2.38/2.52/3.11/3.46`, active endpoint `0.69/0.71/0.85/0.94`, active named submarkers `0.69/0.71/0.85/0.94`, active internal gap `0.00/0.00/0.00/0.00`. Count coverage was stable at `17.2%`; active time coverage was `22.8/22.3/21.6/21.5%`; named endpoint coverage was `100.0%` | attribution only. The previous path residual was a coverage/denominator mismatch: active HD512 k4v4 markers fully cover the active path, but the active path is only about one fifth of mixed GQA time. Exactly one next candidate selected: decode-only non-active GQA layer geometry attribution for the remaining `other_mixed_gqa` time before any optimization | [full log](20260613_180941_gemma4_hqq4_k4v4_coverage_attr_timing.log) |

Notes:
- The active HD512 k4v4 path is not the current limiter; it is only
  `17.2%` of mixed GQA segment count and about `19-23.5%` of mixed GQA time
  across observed rows.
- The next pass should split the non-active GQA layers by geometry/runtime path
  rather than retrying HD512-specific kernel changes.

## Diagnostics - 2026-06-13 (Gemma4 HQQ4/k4v4 post-attention residual attribution)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma. Timing
instrumentation and graph clock markers were enabled for attribution; no
timing-off speed run or QCN guard was run.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Diagnostic build for boundary split | `./dev build` | n/a | n/a | Build passed with the boundary/stream residual report added to the existing graph clock markers; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=126` | proceed to attribution run | [full log](20260613_175047_gemma4_hqq4_k4v4_boundary_attr_build.log) |
| Gemma4 HQQ4/k4v4 boundary/stream residual attribution | `KRASIS_DECODE_GQA_SPLIT_CLOCKS=1 KRASIS_DECODE_GQA_BOUNDARY_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; benchmark summary `5566.7` prefill, `64.09` internal decode, `115.54` HTTP, HCS `3840/3840`, min free `11472 MB`. HCS stayed clean with `240/0` hit/miss, `0` cold DMA/DMA calls, and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. Boundary split across representative 49/99/249/511 rows, ms/tok: GQA-entry gap `0.00/0.00/0.00/0.00`, post-O-proj exit gap `0.00/0.00/0.00/0.00`, HQQ/graph stream debt `0.00/-0.00/-0.00/-0.00`. Active HD512 marked path was `0.70/0.75/0.82/0.87`, while the mixed GQA-path total was `3.09/3.34/3.85/4.60`; mixed MoE expert `3.33/3.34/3.24/3.25`, route/top-k `2.86/2.86/2.69/2.71`, final sync `1.04/1.04/1.04/1.04` | attribution only. The active HD512 k4v4 GQA path has no measurable entry, post-O, or stream-debt gap; the earlier path residual is a mixed-segment coverage/reporting artifact, not hidden post-attention work. Exactly one next candidate selected: decode-only coverage attribution for the mixed `GRAPH_SEG_ROUTE_GQA` label, separating active HD512 k4v4 path time from non-active/other-layer work before any new optimization | [full log](20260613_175323_gemma4_hqq4_k4v4_boundary_attr_timing.log) |
| Diagnostic build attempt 1 | `./dev build` | n/a | n/a | Build passed with the first seven-slot post-attention clock layout; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=124` | timing run exposed an over-narrow clock gate, so this build/run is recorded but not used for attribution | [full log](20260613_172655_gemma4_hqq4_k4v4_post_attn_attr_build.log) |
| Incomplete post-attention timing attempt | `KRASIS_DECODE_GQA_SPLIT_CLOCKS=1 KRASIS_DECODE_MIXED_SEGMENT_CLOCKS=1 KRASIS_DECODE_GQA_PATH_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`, summary `5608.6` prefill, `64.17` decode, `115.31` HTTP, but emitted no HD512 GQA path split because the instrumentation gate was narrower than the prior working path-clock gate. Cleanup required `./dev kill` after the completed server stayed alive | not used for attribution; gate corrected and rerun | [full log](20260613_172936_gemma4_hqq4_k4v4_post_attn_attr_timing.log) |
| Diagnostic build attempt 2 | `./dev build` | n/a | n/a | Build passed after restoring the prior working path-clock gate while keeping the seven-slot post-attention split; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=124` | proceed to attribution run | [full log](20260613_173518_gemma4_hqq4_k4v4_post_attn_attr_build2.log) |
| Gemma4 HQQ4/k4v4 post-attention residual attribution | `KRASIS_DECODE_GQA_SPLIT_CLOCKS=1 KRASIS_DECODE_MIXED_SEGMENT_CLOCKS=1 KRASIS_DECODE_GQA_PATH_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; benchmark summary `5622.8` prefill, `65.96` internal decode, `115.79` HTTP, HCS `3840/3840`, min free `11472 MB`. HCS stayed clean with `240/0` hit/miss, `0` cold DMA/DMA calls, and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. Post-attention split across representative 49/99/249/511 rows, ms/tok: `apply_gated_attn_bf16` `0.01/0.01/0.01/0.01`, O-proj input prep `0.00/0.00/0.00/0.00`, O projection `0.17/0.17/0.17/0.17`, marker residual `2.31/2.61/3.06/3.78`, final graph sync debt `1.04/1.04/1.04/1.04`. Full GQA path was `2.98/3.37/3.90/4.67`; attention/reduce was `0.21/0.27/0.37/0.42` | attribution only. The rejected gated BF16 reduce/output candidate is confirmed as the wrong target: `apply_gated_attn_bf16` and O-proj input prep are effectively negligible. Exactly one next candidate selected: a decode-only boundary/stream residual attribution that splits the remaining marker residual into GQA-entry gap, post-O-proj exit gap, and HQQ/graph stream debt before any new optimization | [full log](20260613_173800_gemma4_hqq4_k4v4_post_attn_attr_timing2.log) |

Notes:
- These passes did not change runtime behavior outside timing/debug mode.
- Boundary attribution shows the active HD512 k4v4 path is fully covered by
  the existing markers; the remaining coarse residual is in the mixed segment
  labelling/coverage, not in hidden post-attention kernels.

## Rejected - 2026-06-13 (Gemma4 HQQ4/k4v4 gated BF16 reduce/output candidate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma. Timing
instrumentation and GQA path clocks were enabled for the gate; no timing-off
speed run or QCN guard was run because the candidate failed the timing gate.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Candidate build | `./dev build` | n/a | n/a | Build passed for the decode-only HD512 k4v4 gated BF16 reduce/output candidate; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=133` | proceed to timing gate | [full log](20260613_171017_gemma4_hqq4_k4v4_gated_reduce_build.log) |
| Gemma4 HQQ4/k4v4 gated-reduce timing gate | `KRASIS_DECODE_GQA_SPLIT_CLOCKS=1 KRASIS_DECODE_MIXED_SEGMENT_CLOCKS=1 KRASIS_DECODE_GQA_PATH_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; benchmark summary `5399.5` prefill, `63.25` internal decode, `115.34` HTTP, HCS `3840/3840`, min free `11474 MB`. HCS stayed clean with `240/0` hit/miss, `0` cold DMA, `0` DMA calls, and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. HD512 k4v4 GQA path residual across representative 49/99/249/511 rows was `2.41/2.62/3.03/3.78 ms/tok` versus the attribution marker `2.32/2.60/3.09/3.78`; attention/reduce was `0.23/0.27/0.37/0.41`, and O projection stayed `0.17` | rejected at timing gate. Correctness passed, but the required GQA-path residual bucket did not improve consistently and internal decode regressed versus the accepted timing marker. Per gate, no timing-off Gemma speed or QCN guard was run | [full log](20260613_171248_gemma4_hqq4_k4v4_gated_reduce_timing.log) |
| Restore accepted source after rejection | `./dev build` | n/a | n/a | Build passed after reverting only the gated BF16 reduce/output candidate; no fused gated-reduce symbols remain; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=132` | restore/build hygiene | [full log](20260613_171752_restore_after_gated_reduce_reject_build.log) |

Notes:
- Candidate gate was captured graph GQA decode only: `GRAPH_SEG_ROUTE_GQA`,
  `layer_idx == range_end`, `kv_format == 9`, effective `head_dim == 512`,
  gated attention, `gqa_tile_size == 256`, `gqa_max_tiles > 1`, and allocated
  tiled buffers.
- The accepted source still contains only the diagnostic GQA path clocks from
  the prior attribution pass; no runtime optimization from this candidate was
  kept.

## Diagnostics - 2026-06-13 (Gemma4 HQQ4/k4v4 HD512 graph GQA path sub-split)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma. Timing
instrumentation and graph clock markers were enabled for attribution; no
timing-off speed run or QCN guard was run.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Diagnostic build | `./dev build` | n/a | n/a | Build passed with diagnostic-only HD512 k4v4 graph GQA path markers; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=124` | proceed to attribution run | [full log](20260613_165218_gemma4_hqq4_k4v4_gqa_path_attr_build.log) |
| Gemma4 HQQ4/k4v4 GQA path sub-split attribution | `KRASIS_DECODE_GQA_SPLIT_CLOCKS=1 KRASIS_DECODE_MIXED_SEGMENT_CLOCKS=1 KRASIS_DECODE_GQA_PATH_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; benchmark summary `5614.9` prefill, `64.23` internal decode, `115.66` HTTP, HCS `3840/3840`, min free `11474 MB`. HCS stayed clean with `240/0` hit/miss, `0` cold DMA, `0` DMA calls, and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. HD512 k4v4 GQA path split across 49/99/249/511 rows, ms/tok: projection `0.23/0.23/0.23/0.23`, norm+RoPE+KV `0.05/0.06/0.06/0.06`, attention/reduce `0.22/0.27/0.37/0.41`, O projection `0.17/0.17/0.17/0.17`, marked total `0.66/0.73/0.83/0.87`, full GQA path `2.98/3.34/3.92/4.66`, residual `2.32/2.60/3.09/3.78`. Mixed-segment rows stayed clean: MoE expert `3.25/3.34/3.30/3.30`, route/top-k `2.72/2.86/2.80/2.80`, final sync about `1.04` | attribution only. The named projection, norm/KV, attention/reduce, and O-projection pieces are not the decode limiter; the dominant GQA-path cost is the post-attention residual gap between attention/reduce and O projection. Chosen next candidate: a tightly gated decode-only HD512 k4v4 graph path that fuses the post-attention gated BF16 conversion into the k4v4 reduce/output stage while preserving output math and BF16 O-projection input. Do not retry rejected GQA/thread/score/single-tile/all-HCS candidates | [full log](20260613_165454_gemma4_hqq4_k4v4_gqa_path_attr_timing.log) |

Notes:
- The residual is now the largest row-growing part of the measured GQA path:
  `2.32 -> 3.78 ms/tok`, while attention/reduce is only
  `0.22 -> 0.41 ms/tok`.
- The next candidate must stay gated to timing-proven Gemma-style graph decode:
  k4v4, effective HD512, tiled graph buffers active, gated attention output,
  and no changes to prefill, HCS policy, final segment behavior, k6,
  non-HD512, or QCN paths.

## Diagnostics - 2026-06-13 (Gemma4 HQQ4/k4v4 mixed graph segment attribution)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma. Timing
instrumentation and split clocks were enabled for attribution; no timing-off
speed run or QCN guard was run.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Diagnostic build | `./dev build` | n/a | n/a | Build passed with diagnostic-only mixed-segment markers; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=124` | proceed to attribution run | [full log](20260613_162655_gemma4_hqq4_k4v4_mixed_segment_attr_build.log) |
| Gemma4 HQQ4/k4v4 mixed-segment attribution | `KRASIS_DECODE_GQA_SPLIT_CLOCKS=1 KRASIS_DECODE_MIXED_SEGMENT_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; benchmark summary `5592.5` prefill, `64.24` internal decode, `113.71` HTTP, HCS `3840/3840`, min free `11474 MB`. HCS stayed clean with `240/0` hit/miss, `0` cold DMA, `0` DMA calls, and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. Mixed-segment split across 49/99/249/511 rows: MoE expert `3.34/3.26/3.34/3.25 ms/tok`, full GQA path `3.07/3.24/3.92/4.58`, route/top-k `2.86/2.72/2.86/2.72`, marked total `9.27/9.21/10.13/10.56`, segment residual `0.20/0.18/0.19/0.18`, final sync about `1.04`. Existing tiled-attention+Polar4-reduce split remained tiny at about `0.22/0.26/0.36/0.40 ms/tok` | attribution only. The coarse `GQA route` bucket is mixed MoE, full GQA path, and route/top-k work, not pure tiled attention/reduce or per-layer sync overhead. Chosen next candidate: target the row-growing full HD512 k4v4 graph GQA path by splitting the decode GQA path into projection, Q/K norm + RoPE + KV write, attention/reduce, and O projection before any new optimization. Do not retry rejected GQA/thread/score/single-tile/all-HCS candidates | [full log](20260613_163014_gemma4_hqq4_k4v4_mixed_segment_attr_timing.log) |

Notes:
- Segment residual is only about `0.18-0.20 ms/tok`, so the new markers
  account for almost all of the mixed graph segment.
- Classify/upload/cold-DMA were not material; the all-HCS replay candidate had
  already shown removing the per-layer sync loop does not improve total decode.
- The row-growing component is the full GQA path, not the already-isolated
  tiled attention/reduce launches.

## Rejected - 2026-06-13 (Gemma4 HQQ4/k4v4 all-HCS graph replay fast path)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma. Timing
instrumentation and split clocks were enabled for the gate; no timing-off speed
run or QCN guard was run because the candidate failed the timing gate.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Candidate build attempt 1 | `./dev build` | n/a | n/a | Build passed for the first all-HCS graph replay fast-path candidate; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=123` | proceed to timing gate | [full log](20260613_155747_gemma4_hqq4_k4v4_all_hcs_fast_build.log) |
| Inactive-gate timing attempt | `KRASIS_DECODE_GQA_SPLIT_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Run was stopped before completion with `./dev kill`. The candidate gate did not activate because Gemma had dynamic-HCS bookkeeping enabled even though runtime state showed complete residency and zero cold DMA | no correctness/speed decision. Gate adjusted to allow dynamic-HCS bookkeeping only under complete all-resident coverage; heatmap collection remained blocked | [partial log](20260613_160027_gemma4_hqq4_k4v4_all_hcs_fast_timing.log) |
| Candidate build attempt 2 | `./dev build` | n/a | n/a | Build passed after the gate adjustment; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=123` | proceed to timing gate | [full log](20260613_160451_gemma4_hqq4_k4v4_all_hcs_fast_build2.log) |
| Gemma4 HQQ4/k4v4 all-HCS fast-path timing gate | `KRASIS_DECODE_GQA_SPLIT_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; benchmark summary `5622.5` prefill, `65.38` internal decode, `116.42` HTTP, HCS `3840/3840`. The candidate eliminated per-segment route sync/upload attribution (`inter: 0.00 ms`, `Classify/Upload/Launch: 0.00 ms`), but the wait moved to the single final sync. Representative residual GQA-route rows versus the split attribution marker were not a consistent improvement: 49-token `9.16` vs `9.08`, 99-token `9.36` vs `9.03`, 249-token `9.71` vs `9.82`, 511-token `10.41` vs `10.18`. HCS stayed clean with `240/0` hit/miss, `0` cold DMA, `0` DMA calls, and dynamic-HCS `promotions=0 evictions=0 copy_failures=0` | rejected at timing gate. Correctness passed, but the required residual GQA-route bucket did not improve consistently and internal decode stayed below the accepted speed marker `65.89 tok/s`. Per gate, no timing-off Gemma speed or QCN guard was run | [full log](20260613_160737_gemma4_hqq4_k4v4_all_hcs_fast_timing2.log) |
| Restore accepted source after rejection | `./dev build` | n/a | n/a | Build passed after reverting only the all-HCS graph replay fast-path candidate; no `all_hcs` symbols remain in `src/gpu_decode.rs`; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=123` | restore/build hygiene | [full log](20260613_161338_restore_after_all_hcs_fast_reject_build.log) |

Notes:
- This candidate reused graph-side `expert_classify_prepare` and launched
  captured graph segments back-to-back when runtime HCS coverage was complete.
  It proved the per-layer CPU sync/upload loop was removable in isolation, but
  total graph work still had to be waited on at final sync, so the user-facing
  decode speed did not improve.

## Diagnostics - 2026-06-13 (Gemma4 HQQ4/k4v4 graph GQA split attribution)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma. Timing
instrumentation was enabled for attribution; no timing-off speed run or QCN
guard was run.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Diagnostic build after marker load-list fix | `./dev build` | n/a | n/a | Build passed with the env-gated `record_globaltimer_u64_g` marker kernel registered in `KERNEL_NAMES`; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=123` | proceed to attribution run | [full log](20260613_153808_gemma4_hqq4_k4v4_gqa_split_clock_build3.log) |
| Gemma4 HQQ4/k4v4 split-clock attribution | `KRASIS_DECODE_GQA_SPLIT_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; benchmark summary `5429.2` prefill, `64.67` internal decode, `117.31` HTTP, HCS `3840/3840`, min free `11474 MB`. HCS stayed clean in graph rows: `100.00%` hit rate, `240/0` hit/miss, `0` cold DMA, `0` copy failures. Split rows showed the accepted HD512 k4v4 tiled attention and Polar4 reduce are small versus the coarse GQA-route segment: 49-token row `0.20 + 0.02 = 0.21 ms/tok` versus `9.30 ms/tok` GQA route; 99-token `0.24 + 0.02 = 0.25` versus `9.29`; 249-token `0.35 + 0.02 = 0.37` versus `10.18`; 511-token `0.39 + 0.02 = 0.40` versus `10.58` | attribution only. The coarse `GQA route` bucket is mostly residual `experts L{n-1} + route L{n}` graph segment work, not the two k4v4 attention/reduce kernels. Next candidate selected from data: an all-HCS-resident graph replay path that removes the per-layer CPU route sync/upload loop when all experts are already resident; do not spend the next pass on Polar4 reduce or tiled-attention micro-kernel changes | [full log](20260613_154108_gemma4_hqq4_k4v4_gqa_split_clock_timing2.log) |

Notes:
- Earlier CUDA-event instrumentation inside the captured graph was rejected
  because `cuEventElapsedTime` failed under graph replay after `4/14` tests:
  [timing failure](20260613_151221_gemma4_hqq4_k4v4_gqa_split_timing.log).
- Nsight Systems extraction was attempted but stalled before a usable graph
  benchmark state and produced no kernel summary rows:
  [nsys log](20260613_152000_gemma4_hqq4_k4v4_gqa_split_nsys.log).
- The final split-clock path uses env-gated marker kernels and does not run
  unless `KRASIS_DECODE_GQA_SPLIT_CLOCKS=1` is set.

## Rejected - 2026-06-13 (Gemma4 HQQ4/k4v4 single-tile reducer bypass decode candidate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma. Timing
instrumentation was enabled for attribution; the candidate failed the timing
gate, so no timing-off speed run or QCN guard was run.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Candidate build attempt 1 | `./dev build` | n/a | n/a | Build failed before runtime testing after adding an output pointer to `gqa_attention_k4v4_tiled_g`; the launch became a 13-argument `cudarc` tuple, which is unsupported by the tuple launch trait | fixed launch plumbing only, then rebuilt | [full log](20260613_144558_gemma4_hqq4_k4v4_single_tile_reduce_build.log) |
| Candidate build attempt 2 | `./dev build` | n/a | n/a | Build passed after converting only the k4v4 tiled launch to the existing raw parameter-vector style; `KRASIS_BUILD_TIMING phase="dev build total" status=0 duration_s=123` | proceed to timing gate | [full log](20260613_144811_gemma4_hqq4_k4v4_single_tile_reduce_build2.log) |
| Gemma4 HQQ4/k4v4 single-tile reducer bypass attribution | `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; benchmark summary `4904.0` prefill, `64.09` internal decode, `115.89` HTTP, min free `11474 MB`. Representative GQA route rows were not a consistent improvement versus the accepted tiled timing marker: 49-token rows around `9.30/9.41 ms/tok`, 99-token around `9.31/9.30`, 249-token around `10.26/10.16`, and final code-gen row around `10.69/10.62`; final remained about `1.04-1.06 ms/tok`. HCS stayed clean with `240/0` hit/miss and `0` copy failures | rejected at timing gate: correctness passed, but the relevant GQA bucket did not improve consistently and internal decode regressed versus the accepted tiled timing run (`64.94 tok/s`). Per gate, no timing-off Gemma speed or QCN guard was run | [full log](20260613_145127_gemma4_hqq4_k4v4_single_tile_reduce_timing.log) |
| Restore accepted source after rejection | `./dev build` | n/a | n/a | Build passed after reverting only the single-tile reducer-bypass candidate; `src/cuda/decode_kernels.cu` returned to no diff and rejected candidate symbols were absent | restore/build hygiene | [full log](20260613_145916_restore_after_single_tile_reduce_reject_build.log) |

Notes:
- The exact candidate gate was graph-mode k4v4 decode with effective
  `hd == 512`, `tile_size == 256`, `max_tiles > 1`, tiled buffers allocated,
  and runtime device-side `num_tiles == 1`.
- The rejected path tried to write final Polar4 output directly from
  `gqa_attention_k4v4_tiled_g` and no-op a k4v4-specific reducer for the
  single-tile case. Output correctness passed, but timing did not justify
  keeping the change.

## Rejected - 2026-06-13 (Gemma4 HQQ4/k4v4 HD512 score-unroll decode candidate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma and QCN
guard runs. Timing instrumentation was enabled only for attribution and
disabled for speed numbers.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Candidate build | `./dev build` | n/a | n/a | Build passed for the HD512/tile256 fixed K-score loop branch inside `gqa_attention_k4v4_tiled_g` | proceed to timing gate | [full log](20260613_141728_gemma4_hqq4_k4v4_hd512_score_unroll_build.log) |
| Gemma4 HQQ4/k4v4 HD512 score-unroll attribution | `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; benchmark summary `4876.5` prefill, `65.18` internal decode, `114.41` HTTP, min free `11474 MB`. Visible graph rows showed GQA route improvement versus the accepted tiled timing in representative rows: 49-token row around `8.77/8.69 ms/tok`, 99-token around `8.96/8.99`, 249-token around `9.52/9.53`, and final code-gen row around `10.55/10.48`; final stayed flat at about `1.04-1.07 ms/tok`. HCS stayed clean with `240/0` hit/miss and `0` copy failures | timing bucket improved enough to run the timing-off speed gate, but this was not an accepted speed result | [full log](20260613_141728_gemma4_hqq4_k4v4_hd512_score_unroll_timing.log) |
| Gemma4 HQQ4/k4v4 HD512 score-unroll speed gate | `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | Full run passed network `14/14`; benchmark summary `5604.3` prefill, `65.32` internal decode, `116.10` HTTP, HCS `3840/3840`, min free `11474 MB`, `0` copy failures | rejected: timing-off speed regressed versus the accepted tiled-GQA marker (`5613.8` prefill, `65.89` decode, `118.74` HTTP). The score-unroll branch was reverted and accepted source was rebuilt | [full log](20260613_142631_gemma4_hqq4_k4v4_hd512_score_unroll_speed.log) |
| QCN speed guard on rejected score-unroll candidate | `./dev speed-test` | HQQ4 | k4v4 | Benchmark complete: `6354.9` prefill, `90.90` internal decode, `153.41` HTTP, HCS `15957/24576`, min free `896 MB`, `0` copy failures | guard completed. QCN `head_dim=256`, so it is gated away from the HD512/tile256 candidate branch; min free remained above the `600 MB` safety margin. Candidate remained rejected based on Gemma timing-off regression | [full log](20260613_143040_qcn_speed_guard_hd512_score_unroll.log) |
| Restore accepted source after rejection | `./dev build` | n/a | n/a | Build passed after reverting only the HD512/tile256 score-unroll candidate branch; `src/cuda/decode_kernels.cu` returned to no diff | restore/build hygiene | [full log](20260613_143620_restore_after_hd512_score_unroll_reject_build.log) |

Notes:
- The candidate confirmed the remaining graph GQA route score loop is a real
  timing-sensitive area, but the speed benchmark did not accept the change.
- Source and installed extension are restored to the prior accepted tiled-GQA
  decode state.

## Rejected - 2026-06-13 (Gemma4 HQQ4/k4v4 HD512 thread-count decode candidate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma and QCN
guard runs. Timing instrumentation was enabled only for attribution and
disabled for speed numbers.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Gemma4 HQQ4/k4v4 HD512 thread-count attribution | `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; summary `5559.5` prefill, `64.47` internal decode, `117.56` HTTP, min free `11474 MB`. Versus the accepted tiled timing log, GQA route averages were mixed: 49-token `9.26` vs `9.20 ms/tok` worse, 99-token `9.50` vs `9.52` slightly better, 249-token `10.01` vs `10.11` better, 511-token `10.49` vs `10.65` better. Final stayed flat around `1.04-1.05 ms/tok`; HCS stayed clean with `240/0` hit/miss and `0` copy failures | long-row GQA timing improved enough to run the timing-off speed gate, but this did not establish an accepted speed win | [full log](20260613_135300_gemma4_hqq4_k4v4_hd512_threads_timing.log) |
| Gemma4 HQQ4/k4v4 HD512 thread-count speed gate | `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | Full run passed network `14/14`; benchmark summary `5471.1` prefill, `65.25` internal decode, `118.26` HTTP, HCS `3840/3840`, min free `11474 MB`, `0` copy failures | rejected: timing-off speed regressed versus the accepted tiled-GQA marker (`5613.8` prefill, `65.89` decode, `118.74` HTTP). The `512` attention-thread launch override was reverted and accepted source was rebuilt | [full log](20260613_140200_gemma4_hqq4_k4v4_hd512_threads_speed.log) |
| QCN speed guard on rejected HD512 thread-count candidate | `./dev speed-test` | HQQ4 | k4v4 | Benchmark complete: `6348.3` prefill, `88.73` internal decode, `148.18` HTTP, HCS `15957/24576`, min free `896 MB`, `0` copy failures | guard completed. QCN `head_dim=256`, so it is gated away from the HD512 candidate branch; the low HTTP result is recorded as a watch item. Candidate remained rejected based on Gemma timing-off regression | [full log](20260613_140700_qcn_speed_guard_hd512_threads.log) |
| Restore accepted source after rejection | `./dev build` | n/a | n/a | Build passed after reverting only the `attn_threads = 512` candidate override; the accepted tiled-GQA decode dispatch remains | restore/build hygiene | [full log](20260613_140846_restore_after_hd512_threads_reject_build.log) |

Notes:
- The measured next decode target remains GQA route work: final is still
  about `1.04-1.05 ms/tok`, while GQA route is about `9-11 ms/tok`.
- Increasing the HD512 tiled attention launch from `256` to `512` threads is
  not accepted because it failed the timing-off speed gate.

## Accepted - 2026-06-13 (Gemma4 HQQ4/k4v4 graph decode tiled-GQA candidate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma and QCN
guard runs. Timing instrumentation was enabled only for attribution and disabled
for speed numbers.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Gemma4 HQQ4/k4v4 decode tiled-GQA attribution | `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; summary `4691.3` prefill, `64.94` internal decode, `119.80` HTTP, min free `11474 MB`. Internal graph rows were mixed: 49-token `15.23 ms/tok`, 99-token `15.47 ms/tok`, 249-token `16.12 ms/tok`; long code-gen row improved to `16.61 ms/tok` with GQA route `10.65 ms/tok`. HCS stayed clean: `100.00%` hit rate, `240/0` hit/miss, `0.00 MB/tok` DMA, `0` copy failures | correctness passed and enough decode buckets improved to run the timing-off gate; the 249-token row was mixed, so the candidate remains a marginal decode win, not a large step toward `200 tok/s` | [full log](20260613_132838_gemma4_hqq4_k4v4_decode_tiled_timing.log) |
| Gemma4 HQQ4/k4v4 decode tiled-GQA speed gate | `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | Full run passed network `14/14`; benchmark summary `5613.8` prefill, `65.89` internal decode, `118.74` HTTP, HCS `3840/3840`, min free `11474 MB`, `0` copy failures. Decode rows: `65.89/64.25/63.23 tok/s` for 50/100/250 | accepted as a small decode improvement over the prior timing-off q2 marker (`65.37` decode, `117.61` HTTP) while preserving correctness and HCS residency. Prefill is effectively unchanged | [full log](20260613_133400_gemma4_hqq4_k4v4_decode_tiled_speed.log) |
| QCN speed guard after decode tiled-GQA candidate | `./dev speed-test` | HQQ4 | k4v4 | Benchmark complete: `6351.4` prefill, `87.72` internal decode, `199.77` HTTP, HCS `15957/24576`, min free `896 MB`, `0` copy failures | guard completed. The new tiled-GQA dispatch is gated on effective `hd == 512`; QCN config head_dim is `256`, so this candidate is gated away from QCN. QCN min free remains near the safety target; internal decode was lower than the previous `90.21 tok/s` guard and is recorded as run variance to watch, not an activation of the new branch | [full log](20260613_133900_qcn_speed_guard_decode_tiled.log) |

Notes:
- The candidate only affects graph-mode k4v4 decode when runtime tiled buffers
  are present and the effective head dimension is `512`; all other k4v4 graph
  decode keeps the existing single-block route kernel.
- This does not solve the decode target: Gemma moves from about `65.37` to
  `65.89 tok/s`, still far from `200 tok/s`.
- The fixed final segment remains about `1.04-1.05 ms/tok`; further decode work
  needs another measured target rather than more blind GQA dispatch changes.

## Diagnostics - 2026-06-13 (Gemma4 HQQ4/k4v4 q2 decode attribution)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the run. Timing
instrumentation was enabled for attribution; speeds below are not timing-off
regression numbers.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Gemma4 HQQ4/k4v4 q2 decode attribution | `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed `14/14` and `ALL TESTS PASSED`; summary `5561.1` prefill, `64.76` internal decode, `116.67` HTTP, min free `11474 MB`. Internal decode rows: `64.76/63.28/61.96 tok/s` for 50/100/250. HCS was clean in graph mode: `100.00%` hit rate, `240/0` hit/miss, `0.00 MB/tok` DMA, `0` copy failures | attribution only: decode bottleneck is graph replay/sync wait, not cold DMA or HCS misses. Representative 249-token graph decode was `15.97 ms/tok`: GPU compute `4.44 ms`, sync wait `11.23 ms`, GQA route `10.04 ms/tok`, final segment about `1.04 ms`. Long code-gen row was `17.19 ms/tok` with sync wait `12.48 ms`. Long prefill still has a separate residual in the five HD512 custom-tiled launches: about `2504.6 ms` over 5 calls on a 14780-token inner row | [full log](20260613_130700_gemma4_hqq4_k4v4_q2_decode_attribution.log) |

Findings:
- HCS/cold-transfer is not the measured decode limiter: graph-mode decode
  reports `3840` cached experts, `240/0` hit/miss, no cold DMA, and no copy
  failures.
- The next decode target is graph-mode replay/sync structure around the GQA
  route segments plus the fixed final segment. GPU compute is already about
  `4.4 ms/tok`, so `200 tok/s` needs both sync reduction and likely additional
  compute/final-segment work.
- Long prefill remains a separate target: the q2 HD512 custom-tiled path costs
  about `500 ms` per fallback layer at 14780 tokens; KV append is not material.

## Accepted - 2026-06-13 (Gemma4 HQQ4 HD512 q2/wide specialization)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for Gemma and QCN
guard runs. Timing instrumentation was disabled for accepted speed numbers.
The timing-enabled Gemma k4 attribution run passed network `14/14` first and
showed `5563.7 tok/s` best prefill.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Gemma4 HQQ4/k4v4 HD512 q2 candidate | `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | Full run passed network `14/14`; benchmark summary `5613.0` prefill, `65.37` decode, `117.61` HTTP, HCS `3840/3840`, min free `11474 MB`, `0` copy failures. Prefill rows: `2932.6/5613.0/4679.2/3794.3/3649.8/3572.9 tok/s` | accepted: improves the current k4 timing-off baseline `4176.7 tok/s` while preserving correctness. Decode remains unchanged and Gemma min-free VRAM remains far above the `600 MB` safety target because this config already keeps all experts hot | [full log](20260613_130500_gemma4_hqq4_k4v4_hd512_q2_speed.log) |
| Gemma4 HQQ4/k6v6 HD512 q2 guard | `./dev test tests/gemma-4-4-hqq4-k6v6-a16.conf` | HQQ4 | k6v6 | Full run passed network `14/14`; benchmark summary `4942.5` prefill, `65.50` decode, `117.91` HTTP, HCS `3840/3840`, min free `11748 MB`, `0` copy failures. Prefill rows: `1865.2/4942.5/4495.2/4557.0/4416.3/4174.8 tok/s` | guard passed: shared HD512 selection also improves the accepted k6 path versus the prior final-validation `3864.2 tok/s`; correctness remains `14/14` | [full log](20260613_142000_gemma4_hqq4_k6v6_hd512_q2_guard.log) |
| QCN speed guard after Gemma HD512 q2 specialization | `./dev speed-test` | HQQ4 | k4v4 | Benchmark complete: `6349.0` prefill, `90.21` decode, `199.75` HTTP, HCS `15957/24576`, min free `928 MB`, `0` copy failures | guard passed; result is consistent with prior QCN speed-test guards and min free remains close to the safety target | [full log](20260613_141000_qcn_speed_test_after_gemma_hd512_q2.log) |

Notes:
- The q2/wide specialization is a partial win against the new Gemma 5090 goal:
  k4 prefill improved from `4176.7` to `5613.0 tok/s`, and k6 prefill improved
  from `3864.2` to `4942.5 tok/s`.
- Remaining target gaps are Gemma prefill long rows and decode/HCS utilization:
  Gemma decode remains about `65 tok/s`, and min-free VRAM remains
  `11.4-11.7 GB` because the tested Gemma configs already pin `3840/3840`
  experts.

## Blocked - 2026-06-13 (Zephyrus connectivity diagnosis)

Connectivity-only diagnosis for the pre-release Zephyrus target. No model
validation, podman run, benchmark, or optimization candidate was started.
Existing notes still identify Zephyrus as `main@192.168.1.228` in
`/home/main/Documents/BOX_3070.txt`.

| Probe | Command | Result | Decision | Logs |
|-------|---------|--------|----------|------|
| Local route | `ip route get 192.168.1.228` | Route exists: `192.168.1.228 dev enp65s0 src 192.168.1.181 uid 1000` | host is expected to be on the local `192.168.1.0/24` link | [full log](20260613_093522_zephyrus_connectivity_diagnosis.log) |
| ICMP | `ping -c 3 -W 2 192.168.1.228` | `Destination Host Unreachable`, `0 received`, `100% packet loss` | target is not reachable on the local link | [full log](20260613_093522_zephyrus_connectivity_diagnosis.log) |
| Neighbour/ARP | `ip neigh show 192.168.1.228`; `arp -n 192.168.1.228` | neighbour entry `FAILED`; ARP entry `(incomplete)` on `enp65s0` | layer-2 resolution is failing | [full log](20260613_093522_zephyrus_connectivity_diagnosis.log) |
| SSH | `ssh -o BatchMode=yes -o ConnectTimeout=8 main@192.168.1.228 'hostname'` | `ssh: connect to host 192.168.1.228 port 22: No route to host` | pre-release matrix remains blocked; no podman substitution | [full log](20260613_093522_zephyrus_connectivity_diagnosis.log) |

## Blocked - 2026-06-13 (pre-release environment validation matrix)

Timing instrumentation was disabled; no model timing rows were produced because
the first remote environment failed preflight. `./dev build` passed first and
the matrix stopped at the Zephyrus reachability failure instead of substituting
another target.

Identified variants:
- Zephyrus RTX 3070 Laptop 8 GB: installed-command Qwen3.5-35B-A3B HQQ4/k4v4
  500 MB KV and HQQ4+10% (`hqq46_auto`) k4v4 500 MB KV.
- Podman `krasis-run`: Lore Qwen3.6-35B-A3B config on GPU 1/A4500, HQQ6+10%
  (`hqq68_auto`, 10%), k6v6, source-mode dynamic HCS, Krasis-owned SSH tunnel.

| Run | Command | Result | Decision | Logs |
|-----|---------|--------|----------|------|
| Pre-release build | `./dev build` | Passed; repo-local extension rebuilt from current source | proceed to first environment variation | [full log](20260613_093124_prerelease_build.log) |
| Zephyrus preflight | `ssh -o BatchMode=yes -o ConnectTimeout=8 main@192.168.1.228 ...` | Failed before remote command execution: `ssh: connect to host 192.168.1.228 port 22: No route to host` | stop for diagnosis; no Zephyrus timing-off variants and no podman run | [full log](20260613_093124_prerelease_zephyrus_preflight_failed.log) |

## Validation - 2026-06-13 (Gemma4 HQQ4 final timing-off matrix)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for both runs.
Timing instrumentation was disabled. `./dev build` passed before the matrix.
Both runs rebuilt the GPU INT4 Marlin expert cache during startup; benchmark
summary rows below are the timing-off speed comparison points.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Gemma4 HQQ4/k4v4 final validation | `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | Full run passed network `14/14`; benchmark summary `4176.7` prefill, `65.47` decode, `117.99` HTTP, HCS `3840/3840`, min free `11474 MB`, `0` copy failures observed in HCS rows. Prefill rows: `2824.5/4176.7/3013.0/2291.5/2283.2/2214.2 tok/s` | validation passed; remains above the accepted k4 comparison point `4056.2 tok/s`; no new candidate or QCN guard | [full log](20260613_091559_gemma4_hqq4_k4v4_final_validation.log) |
| Gemma4 HQQ4/k6v6 final validation | `./dev test tests/gemma-4-4-hqq4-k6v6-a16.conf` | HQQ4 | k6v6 | Full run passed network `14/14`; benchmark summary `3864.2` prefill, `65.69` decode, `118.91` HTTP, HCS `3840/3840`, min free `11748 MB`, `0` copy failures observed in HCS rows. Prefill rows: `1565.0/3864.2/2991.2/2865.7/2738.2/2829.3 tok/s` | validation passed; remains above the accepted k6 comparison point `3750.8 tok/s`; no new candidate or QCN guard | [full log](20260613_092009_gemma4_hqq4_k6v6_final_validation.log) |

## Accepted - 2026-06-13 (Gemma4 HQQ4/k6v6 stage-exact head_dim=512 specialization)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for both runs.
Timing instrumentation was disabled for speed numbers.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Gemma4 HQQ4/k6v6 stage-exact hd512 candidate | `./dev test tests/gemma-4-4-hqq4-k6v6-a16.conf` | HQQ4 | k6v6 | Full run passed network `14/14`; benchmark summary `3750.8` prefill, `65.68` decode, `118.34` HTTP, HCS `3840/3840`, min free `11748 MB`, `0` copy failures. Prefill rows: `1579.0/3750.8/3013.3/2826.2/2735.9/2864.0 tok/s` | accepted: a separate k6 stage-exact gate reuses the hd512 full-attention specialization only for Gemma4 HQQ4 active FP8 prefill KV (`kv_format=1`, `decode_kv_format=7`, `prefill_kv_active=true`), full attention/window `0`, `head_dim=512`, `16` Q heads, `2` KV heads, and the existing shared-memory capability check. This beats the timing-off k6 baseline `2268.3 tok/s`; the k4 direct-cache gate remains unchanged | [full log](20260613_085858_gemma4_hqq4_k6v6_stage_exact_hd512_candidate_test.log) |
| QCN speed guard after Gemma k6 stage-exact hd512 specialization | `./dev speed-test` | HQQ4 | k4v4 | Benchmark complete: `6357.4` prefill, `88.91` decode, `145.97` HTTP, HCS `15957/24576`, min free `864 MB`, `0` copy failures | guard passed; the Gemma stage-exact specialization is gated away from QCN and did not break the standard QCN speed-test path | [full log](20260613_090338_qcn_speed_guard_after_gemma_k6_stage_hd512.log) |

Tracked k6 speed context:
- Timing-off baseline: `2268.3 tok/s` best from
  [20260613_081729_gemma4_hqq4_k6v6_baseline_test.log](20260613_081729_gemma4_hqq4_k6v6_baseline_test.log).
- Rejected direct-cache-style k6 gate candidate: `2264.1 tok/s` best from
  [20260613_082645_gemma4_hqq4_k6v6_hd512_full_attention_candidate_test.log](20260613_082645_gemma4_hqq4_k6v6_hd512_full_attention_candidate_test.log).

## Diagnostics - 2026-06-13 (Gemma4 HQQ4/k6v6 timing attribution)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the run.
Timing instrumentation was enabled for attribution; speeds below are not
timing-off regression numbers.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Gemma4 HQQ4/k6v6 attribution | `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k6v6-a16.conf --timing` | HQQ4 | k6v6 | Full run passed `ALL TESTS PASSED`; benchmark summary `1647.9` prefill, `64.31` decode, `116.77` HTTP, HCS `3840/3840`, min free `11748 MB`, `0` copy failures. Timing rows: `1555.0/1647.9/1012.4/972.0/907.7/892.2 tok/s`. The k6 path is stage-exact during prefill (`kv_format=1`, `decode_kv_format=7`, `prefill_kv_active=true`). The five `custom_no_fa2` layers are still `5/11/17/23/29`; on the 8419-token calibration row, `flash_attn_tiled_launch` took `6313.9 ms` wall / `6314.0 ms` event over 5 calls, while `kv_append_kernel` took `2.1 ms` over 30 calls | attribution only: the prior k6 HD512 gate extension did not help because it kept direct-cache conditions and did not match k6 stage-exact prefill. Next optimization must target the k6 stage-exact `custom_tiled` fallback path or change the k6 cache/attention path deliberately; no QCN guard was run | [full log](20260613_084355_gemma4_hqq4_k6v6_timing_attribution.log) |

Tracked k6 speed context:
- Timing-off baseline: `2268.3 tok/s` best from
  [20260613_081729_gemma4_hqq4_k6v6_baseline_test.log](20260613_081729_gemma4_hqq4_k6v6_baseline_test.log).
- Rejected k6 HD512 gate candidate: `2264.1 tok/s` best from
  [20260613_082645_gemma4_hqq4_k6v6_hd512_full_attention_candidate_test.log](20260613_082645_gemma4_hqq4_k6v6_hd512_full_attention_candidate_test.log).
- Current timing-enabled summary: `1647.9 tok/s` prefill; use this run for
  attribution, not speed regression claims.

## Rejected - 2026-06-13 (Gemma4 HQQ4/k6v6 head_dim=512 gate extension)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for both runs.
Timing instrumentation was disabled for speed numbers.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Gemma4 HQQ4/k6v6 baseline | `./dev test tests/gemma-4-4-hqq4-k6v6-a16.conf` | HQQ4 | k6v6 | Full run passed network `14/14`; benchmark summary `2268.3` prefill, `65.82` decode, `115.72` HTTP, HCS `3840/3840`, min free `11748 MB`, `0` copy failures. Prefill rows: `2268.3/1708.6/950.7/891.2/893.5/905.0 tok/s` | baseline for k6v6 before extending the accepted hd512 specialization gate | [full log](20260613_081729_gemma4_hqq4_k6v6_baseline_test.log) |
| Gemma4 HQQ4/k6v6 hd512 gate candidate | `./dev test tests/gemma-4-4-hqq4-k6v6-a16.conf` | HQQ4 | k6v6 | Full run passed network `14/14`; benchmark summary `2264.1` prefill, `65.89` decode, `118.24` HTTP, HCS `3840/3840`, min free `11748 MB`, `0` copy failures. Prefill rows: `2264.1/1679.0/950.2/892.9/895.3/903.5 tok/s` | rejected: correctness passed, but prefill did not improve over the k6v6 baseline; the k6 gate was reverted and `./dev build` passed afterward. No QCN guard was run because no speed win was accepted | [full log](20260613_082645_gemma4_hqq4_k6v6_hd512_full_attention_candidate_test.log) |

Notes:
- The current accepted specialization remains k4v4-only. k6v6 has the same
  Gemma `head_dim=512` full-attention layer geometry, but this candidate shows
  the existing HD512 kernel does not improve the k6v6 timing-off benchmark.
- The candidate run started by killing a leftover baseline server process; the
  actual benchmark launched cleanly afterward.

## Accepted - 2026-06-13 (Gemma4 HQQ4/k4v4 head_dim=512 full-attention specialization)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for both runs.
Timing instrumentation was disabled for speed numbers.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Gemma4 HQQ4/k4v4 hd512 full-attention candidate | `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | Full run passed network `14/14`; benchmark summary `4056.2` prefill, `65.31` decode, `118.03` HTTP, HCS `3840/3840`, min free `11474 MB`, `0` copy failures. Prefill rows: `1409.2/4056.2/3004.7/2219.5/2222.5/2211.8 tok/s` | accepted: a strictly gated `head_dim=512`, full-attention, Gemma4 HQQ4/k4v4 direct-cache kernel with `BC=32` improves timing-off prefill versus the tracked `1653.2` latest timing-off marker and `2303.8` old best | [full log](20260613_080259_gemma4_hqq4_k4v4_hd512_full_attention_candidate_test.log) |
| QCN speed guard after Gemma hd512 specialization | `./dev speed-test` | HQQ4 | k4v4 | Benchmark complete: `6365.4` prefill, `87.28` decode, `146.51` HTTP, HCS `15957/24576`, min free `928 MB`, `0` copy failures | guard passed; QCN path remains functional and the Gemma specialization gate did not affect QCN branch selection | [full log](20260613_080758_qcn_speed_guard_after_gemma_hd512_attention.log) |

Tracked Gemma HQQ4/k4v4 speed context:
- Old accepted timing-off marker: `2303.8 tok/s` best from
  [20260610_181558_gemma4_hqq4_k4v4_prefill_skip_final_test.log](20260610_181558_gemma4_hqq4_k4v4_prefill_skip_final_test.log).
- Latest timing-off marker before this candidate: `1653.2 tok/s` best from
  [20260613_062700_gemma4_hqq4_k4v4_after_kvappend_timing_instrumentation.log](20260613_062700_gemma4_hqq4_k4v4_after_kvappend_timing_instrumentation.log).

## Diagnostics - 2026-06-13 (Gemma4 HQQ4/k4v4 custom_tiled sub-step attribution)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the run.
Timing instrumentation was enabled for attribution; speeds below are not
timing-off regression numbers.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Gemma4 HQQ4/k4v4 `custom_tiled` sub-step attribution | `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed network `14/14`; benchmark summary `2220.7` prefill, `64.68` decode, `116.62` HTTP, HCS `3840/3840`, min free `11474 MB`, `0` copy failures. Timing rows: `4764/1806/990/729/689/728 tok/s`. `custom_tiled` host sub-steps were `0.0 ms`; `flash_attn_tiled_launch` event owned the wait: `116.1 ms` (1K), `2424.7 ms` (5K), `9380.5 ms` (10K), `13091.7 ms` (11,824 calibration), and `19171.5/20346.8/19175.0 ms` on capped 14,780/14,780/14,779 rows | attribution only: the producer is the `flash_attn_tiled` fallback kernel for the five `fa2_head_dim=false` layers, not wrapper layout math, argument packing, sync placement, append, fixed FA2, or HQQ projection | [full log](20260613_074551_gemma4_hqq4_k4v4_custom_tiled_substep_timing.log) |

Tracked speed context:
- Old accepted timing-off marker: `2303.8 tok/s` best from
  [20260610_181558_gemma4_hqq4_k4v4_prefill_skip_final_test.log](20260610_181558_gemma4_hqq4_k4v4_prefill_skip_final_test.log).
- Latest timing-off marker: `1653.2 tok/s` best from
  [20260613_062700_gemma4_hqq4_k4v4_after_kvappend_timing_instrumentation.log](20260613_062700_gemma4_hqq4_k4v4_after_kvappend_timing_instrumentation.log).
- Previous timing-enabled summary: `2097.3 tok/s` prefill from
  [20260613_072716_gemma4_hqq4_k4v4_attention_branch_attribution.log](20260613_072716_gemma4_hqq4_k4v4_attention_branch_attribution.log).
- Current run is timing-enabled and should be used for attribution only.

Findings:
- The `custom_no_fa2` fallback is selected for layers `5`, `11`, `17`, `23`,
  and `29` because `fa2_head_dim=false`.
- The wrapper performs head-dim validation, fixed `16x16` tile/shared-memory
  math, cache-pointer selection, argument packing, and one `flash_attn_tiled`
  launch on the main stream.
- Entry, validation, layout math, pointer selection, argument packing, and
  pre-launch checkpoints were all `0.0 ms`.
- The `flash_attn_tiled_launch` CUDA event and sync time matched the full
  branch debt, so the next target is the tiled fallback attention kernel/path
  for `head_dim=512`, not host wrapper work.
- No optimization candidate was built from this run, and no QCN guard was run.

## Diagnostics - 2026-06-13 (Gemma4 HQQ4/k4v4 attention branch attribution)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the run.
Timing instrumentation was enabled for attribution; speeds below are not
timing-off regression numbers.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Gemma4 HQQ4/k4v4 attention branch attribution | `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed network `14/14`; benchmark summary `2097.3` prefill, `64.77` decode, `116.76` HTTP, HCS `3840/3840`, min free `11474 MB`, `0` copy failures. Timing rows: `4909/1863/1047/728/689/690 tok/s`. 11,824-token row: non-fixed calls are layers `5/11/17/23/29`, all `custom_no_fa2` / `custom_tiled` because `fa2_head_dim=false`; `custom_no_fa2` after-branch sync `13075.6 ms`. Measured 10K row: `8830.2 ms`; measured 14,780/14,780/14,779 rows: `19179.9/20318.5/20313.0 ms` | attribution only: the long wait is owned by five custom tiled fallback GQA layers, not fixed FA2, append, trace/reference hooks, or HQQ projection. Next target is the custom tiled fallback/wrapper for unsupported head-dim layers | [full log](20260613_072716_gemma4_hqq4_k4v4_attention_branch_attribution.log) |

Tracked speed context:
- Old accepted timing-off marker: `2303.8 tok/s` best from
  [20260610_181558_gemma4_hqq4_k4v4_prefill_skip_final_test.log](20260610_181558_gemma4_hqq4_k4v4_prefill_skip_final_test.log).
- Latest timing-off marker: `1653.2 tok/s` best from
  [20260613_062700_gemma4_hqq4_k4v4_after_kvappend_timing_instrumentation.log](20260613_062700_gemma4_hqq4_k4v4_after_kvappend_timing_instrumentation.log).
- Last timing-enabled summary before this pass: `1630.4 tok/s` prefill from
  [20260613_071017_gemma4_hqq4_k4v4_append_gap_bisection.log](20260613_071017_gemma4_hqq4_k4v4_append_gap_bisection.log).
- Current run is timing-enabled and should be used for attribution only.

Findings:
- Every measured GQA call had `start_pos=0`, `kv_format=9`, and
  `prefill_kv_active=false`.
- Fixed FA2 covered 25 layers. The five non-fixed layers were exactly
  `5`, `11`, `17`, `23`, and `29`.
- Those five layers took the `custom_no_fa2` fallback because
  `fa2_head_dim=false`; no sliding-window/ring branch was involved.
- Branch-entry sync was `0.0 ms`; branch-exit sync carried the long debt.
- No optimization candidate was built from this run, and no QCN guard was run.

## Diagnostics - 2026-06-13 (Gemma4 HQQ4/k4v4 append-gap bisection)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the run.
Timing instrumentation was enabled for attribution; speeds below are not
timing-off regression numbers.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Gemma4 HQQ4/k4v4 post-FA2 append-gap bisection | `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | Full run passed network `14/14`; benchmark summary `1630.4` prefill, `66.10` decode, `115.73` HTTP, HCS `3840/3840`, min free `11474 MB`, `0` copy failures. Timing rows: `4803/1862/1048/729/687/728 tok/s`. 11,824-token row: `after_fa2_bookkeeping=0.0 ms`, `after_attention_branch=13080.6 ms`, append `2.5/2.2 ms`; 14,780-token rows: `after_attention_branch=19162.0-20399.1 ms` | attribution only: debt is first observed immediately after the attention branch exits. Because fixed FA2 has 25 calls and `after_attention_branch` has 30, next target is the unbracketed non-fixed-FA2 attention branch/layer work, not trace/reference hooks, append setup/events, or append kernel | [full log](20260613_071017_gemma4_hqq4_k4v4_append_gap_bisection.log) |

Tracked speed context:
- Old accepted timing-off marker: `2303.8 tok/s` best from
  [20260610_181558_gemma4_hqq4_k4v4_prefill_skip_final_test.log](20260610_181558_gemma4_hqq4_k4v4_prefill_skip_final_test.log).
- Latest timing-off marker: `1653.2 tok/s` best from
  [20260613_062700_gemma4_hqq4_k4v4_after_kvappend_timing_instrumentation.log](20260613_062700_gemma4_hqq4_k4v4_after_kvappend_timing_instrumentation.log).
- Current run is timing-enabled and should be used for attribution only.

Findings:
- The prior `pre_append` debt was moved earlier: after fixed FA2 bookkeeping is
  clean, but the first checkpoint after the attention branch exits carries the
  long wait.
- All later checkpoints are clean: reference hook, trace hook, trace summary,
  append setup, direct k4 shape/argument setup, append start event, and
  before-launch.
- Append remains tiny (`2-4 ms` wall, `2-3 ms` event on long rows).
- No optimization candidate was built from this run, and no QCN guard was run.

## Diagnostics - 2026-06-13 (Gemma4 HQQ4/k4v4 sync-debt bisection)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the run.
Timing instrumentation was enabled for attribution. The run was stopped after
the long calibration and one 14,780-token benchmark warmup row because the data
was conclusive; the log therefore contains expected `Killed` / server-start
failure lines from intentional cleanup.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Gemma4 HQQ4/k4v4 sync-debt bisection | `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | 11,824-token row: GQA entry/Q/K/V debt `0.0 ms`, pre-GQA `6.7 ms`, after-FA2 `137.4 ms`, pre-append `13063.3 ms`, append `2.2/2.0 ms`; 14,780-token row: pre-append `19119.8 ms`, append `3.2/3.0 ms` | attribution only: producer is after the post-FA2 checkpoint and before append launch; do not optimize Q/K/V projection, prior Gemma4 handoff, or append kernel from this data | [full log](20260613_065808_gemma4_hqq4_k4v4_sync_debt_bisection.log) |

Findings:
- Layer handoff, pre-GQA, GQA entry, and individual Q/K/V projection
  checkpoints do not contain the 13-19s debt.
- The append kernel remains tiny: `2.0 ms` event on the 11,824-token row and
  `3.0 ms` event on the 14,780-token row.
- The missing wait appears at the explicit pre-append checkpoint, before the
  append event can be recorded. This narrows the next attribution target to the
  small host/code interval between the post-FA2 sync and direct-cache append
  launch.
- No optimization candidate was built from this run, and no QCN guard was run.

## Diagnostics - 2026-06-13 (Gemma4 HQQ4/k4v4 GQA queue event timing)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the run.
Timing instrumentation was enabled for attribution. The run was stopped after
the long calibration and one 14,780-token benchmark warmup row because the data
was conclusive; the log therefore contains expected `Killed` / server-start
failure lines from intentional cleanup.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Gemma4 HQQ4/k4v4 GQA queue event timing | `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | 11,824-token row: projection `181.3 ms` wall / `174.6 ms` event, QKV norm `11.7/11.5`, RoPE `6.8/6.6`, FA2 `137.6/137.4`, O projection `92.6/92.4`, direct append `13066.9/1.9`; 14,780-token row: append `20260.6/2.9` while projection/FA2/O stayed close to event time | attribution only: do not build a projection/norm/RoPE/FA2/O-projection optimization from this data; debt appears as explicit stream-sync wait at append boundary, not as measured kernel time in the bracketed GQA phases | [full log](20260613_064303_gemma4_hqq4_k4v4_gqa_queue_timing.log) |

Findings:
- The accepted `2303.8 tok/s` prior result was a 1K-row best from
  [20260610_181558_gemma4_hqq4_k4v4_prefill_skip_final_test.log](20260610_181558_gemma4_hqq4_k4v4_prefill_skip_final_test.log).
  A later timing-off run with the same config/GPU/HCS/min-free/prompt sizes did
  not reproduce that 1K row, but rows 2-6 were comparable or better.
- On the 11,824-token row, projection, QKV norm, RoPE, FA2, and O projection
  CUDA-event times closely matched their wall buckets. They are not hiding the
  `13.1 s` queue debt.
- Direct k4/v4 append still shows the attribution gap: `13066.9 ms` wall and
  `1.9 ms` kernel event on the 11,824-token row; `20260.6 ms` wall and
  `2.9 ms` event on the 14,780-token row.
- No optimization candidate was built from this run, and no QCN guard was run.

## Diagnostics - 2026-06-13 (Gemma4 HQQ4/k4v4 KV append event timing)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the run.
Timing instrumentation was enabled for attribution. The run was stopped after
the long calibration and one benchmark warmup row because the data was
conclusive; the log therefore contains expected `Killed` / server-start failure
lines from intentional cleanup.

| Run | Command | Attention | KV | Result | Decision | Logs |
|-----|---------|-----------|----|--------|----------|------|
| Gemma4 HQQ4/k4v4 direct-cache append event timing | `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | 11,824-token row: wall `kv_append=13070.1 ms`, `kv_append_kernel=2.0 ms` over 30 calls; 14,780-token row: wall `kv_append=19114.7 ms`, `kv_append_kernel=3.0 ms` | attribution only: do not build replacement k4/v4 append kernel; old wall bucket is synchronization debt, not append kernel body time | [full log](20260613_061700_gemma4_hqq4_k4v4_kvappend_event_timing.log) |
| Gemma4 HQQ4/k4v4 timing-off validation after append instrumentation | `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | `1653.2` prefill, `65.41` decode, `117.73` HTTP, network `14/14`, HCS `3840/3840`, min free `11474 MB` | passed validation but no speed win; best row did not reproduce the earlier `2303.8 tok/s` 1K result, while longer rows were comparable | [full log](20260613_062700_gemma4_hqq4_k4v4_after_kvappend_timing_instrumentation.log) |

Findings:
- The existing `kv_append` timing syncs at the append boundary and can charge
  earlier queued stream work to the append bucket.
- CUDA-event timing around only the direct-cache k4/v4 append kernel measured
  `2.0-3.0 ms` total over 30 layer calls on long rows, while the old wall
  bucket was `13.1-19.1 s`.
- The k4/v4 append kernel uses one block per token and threads over 16-value
  KV blocks, writing packed k4 bytes plus BF16 scale/radius data. It is not the
  right rewrite target in this profile because its measured kernel body is only
  a few milliseconds.
- A replacement append kernel would at most target a few milliseconds in this
  run, so the next speed work should instrument the preceding GQA/attention
  queue rather than rewriting the append stores.

## Diagnostics - 2026-06-13 (Gemma4 HQQ projection backend)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the runs.
Timing instrumentation was enabled only for attribution and disabled for the
candidate validation row. No new runtime default was accepted.

| Run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Decision | Logs |
|-----|---------|-----------|----|----------------:|---------------:|-------------------:|-----|----------|------|
| Gemma4 HQQ4/k4v4 current backend timing | `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` | HQQ4 | k4v4 | n/a | n/a | n/a | 3840/3840 (100.0%) | attribution only: long row still dominated by KV append, not HQQ correction | [full log](20260613_055800_gemma4_hqq4_k4v4_backend_timing.log) |
| Gemma4 HQQ4/k4v4 base-only Marlin projection | `KRASIS_GEMMA_HQQ4_PREFILL_BASE_ONLY=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 base-only experiment | k4v4 | 1742.1 | 66.08 | 115.27 | 3840/3840 (100.0%) | rejected: passed network `14/14` but slower than accepted HQQ4/k4v4 stage-exact skip (`2303.8 tok/s`) and visibly degraded code generation | [full log](20260613_070000_gemma4_hqq4_k4v4_base_only_projection_test.log) |

Findings:
- Timing attribution on the current backend showed `11824` tokens spent
  `13082.4 ms` in `kv_append`, while HQQ projection internals were
  `237.0 ms` Marlin float-zp, `5.2 ms` group sums, `10.4 ms` correction GEMM,
  and `18.4 ms` correction add.
- Skipping HQQ4 two-scale/intercept repair did not help. It removed a small
  backend repair cost, but the timing-off row was slower and output quality was
  weaker on the code-gen sample.
- The candidate was reverted and `./dev build` passed afterward; the active
  runtime diff remains the previously validated HQQ4/k4v4 single-chunk
  stage-exact skip.

## Diagnostics - 2026-06-13 (Gemma4 fused cache-prep candidate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the run.
Timing instrumentation was disabled. No new runtime default was accepted.

| Run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Decision | Logs |
|-----|---------|-----------|----|----------------:|---------------:|-------------------:|-----|----------|------|
| Gemma4 HQQ4/k4v4 fused Q/K/V cache prep | `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | 1710.7 | 65.13 | 117.36 | 3840/3840 (100.0%) | rejected: passed network `14/14` but prefill regressed below the accepted HQQ4/k4v4 stage-exact skip (`2303.8 tok/s`) | [full log](20260613_053500_gemma4_hqq4_k4v4_fused_cache_prepare_test.log) |

Findings:
- Fusing Q norm+half-split RoPE, K norm+half-split RoPE+final k4 cache write,
  and V no-scale RMSNorm+final v4 cache write was correct enough for the short
  network suite, but slower than the existing split kernels plus direct-cache
  append path.
- Prefill row speeds were `1552.5`, `1710.7`, `969.3`, `713.5`, `667.0`, and
  `668.8 tok/s`, so the larger post-projection/cache-prep fusion repeated the
  long-row collapse seen in the smaller Q/K fusion.
- The candidate was reverted and `./dev build` passed afterward; the active
  runtime diff remains the previously validated HQQ4/k4v4 single-chunk
  stage-exact skip.

## Diagnostics - 2026-06-12 (Gemma4 fused HQQ/GQA kernel candidates)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the runs.
Timing instrumentation was disabled for these candidate validation runs. No new
runtime default was accepted.

| Run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | Decision | Logs |
|-----|---------|-----------|----|----------------:|---------------:|-------------------:|----------|------|
| Gemma4 HQQ4/k4v4 fused Q/K norm + RoPE | `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | 1659.5 best early row | n/a | n/a | rejected: fused Q/K RMSNorm plus half-split RoPE regressed badly; run stopped after prefill rows | [full log](20260612_221900_gemma4_hqq4_k4v4_qk_norm_rope_fused_test.log) |
| Gemma4 HQQ8/k4v4 generalized stage-exact skip | `./dev test tests/gemma-4-4-hqq8-k4v4-a16.conf` | HQQ8 | k4v4 | n/a | n/a | n/a | rejected: timed prefill stalled with GPU0 pegged and no row output; restored HQQ4-only gate | [full log](20260612_232300_gemma4_hqq8_k4v4_stage_exact_skip_test.log) |

Findings:
- Small post-projection fusion is not enough for Gemma HQQ speed and can
  regress badly. The attempted Q/K RMSNorm plus RoPE fusion added scheduling
  overhead instead of removing the dominant GQA/KV cost.
- The existing stage-exact skip is not safely generalizable across HQQ modes
  by config shape alone. HQQ8/k4v4 stalled in the timed prefill row, so the
  runtime gate was restored to the validated HQQ4/k4v4-only condition.
- The next serious speed path should be a larger Gemma-specific fused
  projection/GQA/KV staging path, not another isolated post-projection kernel.

## Diagnostics - 2026-06-12 (Gemma4 HQQ prefill modes)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the runs.
Timing instrumentation was disabled for the validation rows. These runs compare
existing HQQ prefill execution modes; no new default was accepted.

| Run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Decision | Logs |
|-----|---------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|----------|------|
| Gemma4 HQQ4/k4v4 persistent BF16 materialization | `KRASIS_HQQ_PREFILL_PERSISTENT_BF16=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | 1707.7 | 65.50 | 117.70 | 3840/3840 (100.0%) | 9330 MB | rejected: correct but slower than HQQ4/k4v4 baseline and accepted skip | [full log](20260612_205300_gemma4_hqq4_k4v4_persistent_bf16_test.log) |
| Gemma4 HQQ8/k4v4 native fused Marlin | `KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin ./dev test tests/gemma-4-4-hqq8-k4v4-a16.conf` | HQQ8 | k4v4 | 1734.0 | 65.26 | 119.35 | 3840/3840 (100.0%) | 10954 MB | candidate: faster than default HQQ8 but below HQQ4/k4v4 and BF16 | [full log](20260612_211250_gemma4_hqq8_k4v4_native_fused_marlin_test.log) |
| Gemma4 HQQ8/k4v4 native fused Marlin v2 | `KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin-v2 ./dev test tests/gemma-4-4-hqq8-k4v4-a16.conf` | HQQ8 | k4v4 | 1855.3 | 66.67 | 119.10 | 3840/3840 (100.0%) | 10954 MB | best HQQ8 mode tested; not accepted as default because it remains far below BF16 | [full log](20260612_212540_gemma4_hqq8_k4v4_native_fused_marlin_v2_test.log) |
| Gemma4 HQQ8/k4v4 symmetric Marlin | `KRASIS_HQQ8_PREFILL_MODE=symmetric-marlin ./dev test tests/gemma-4-4-hqq8-k4v4-a16.conf` | HQQ8 | k4v4 | 1652.3 | 66.00 | 119.29 | 3840/3840 (100.0%) | 10954 MB | rejected: slower than v2 and produced a visibly weaker multi-turn sample | [full log](20260612_214020_gemma4_hqq8_k4v4_symmetric_marlin_test.log) |

Findings:
- HQQ8 v2 is the fastest existing HQQ8 runtime format tested on Gemma4, but
  the improvement is modest (`1590.0` default HQQ8/k4v4 to `1855.3 tok/s`).
- HQQ-specific projection-mode changes do not close the gap to Gemma4 BF16
  attention (`5196.5 tok/s` on BF16/k4v4), and they do not beat the narrower
  HQQ4/k4v4 stage-exact skip (`2303.8 tok/s`).
- Persistent BF16 materialization from HQQ artifacts is not a speed solution:
  it preserves correctness but regresses speed and spends more VRAM.

## Standard Benchmarks - 2026-06-05 (Gemma4 26B A4B text INT4 baseline)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the run.
Timing instrumentation was disabled. The run used
`./dev benchmark tests/gemma-4-4-a16.conf` on branch `gemma-dev`. Gemma4 uses
compact BF16 KV cache because its mixed per-layer KV geometry is not supported
by the integer KV cache formats yet. Decode ran in the Gemma4 ungraphed path;
per-layer CUDA graph capture is disabled for Gemma4 until the graph segment
path implements Gemma's dense-MLP-plus-routed-MoE layer composition.

| Model / run | Command | Experts | Attention | KV | Context | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Health | Logs |
|-------------|---------|---------|-----------|----|--------:|----------------:|---------------:|-------------------:|-----|--------------:|--------|------|
| Gemma4 26B A4B IT text INT4 baseline | `./dev benchmark tests/gemma-4-4-a16.conf` | INT4 | BF16 | BF16 | 4,640 | 5051.6 | 39.07 | 71.33 | 3840/3840 (100.0%) | 11084 MB | clean | [stdout](20260605_165100_gemma4_int4_nograph_benchmark_stdout.log), [report](20260605_165100_gemma4_int4_nograph_benchmark_report.log), [server log](20260605_165100_gemma4_int4_nograph_krasis.log) |

Notes:
- Benchmark prompt targets above the available context were truncated to
  `4,540` prompt tokens, so the 5K/10K/20K/35K/50K rows all measure the same
  capped short-context length. Row speeds were `5051.6 tok/s` at 1K, then
  `2097.7`, `2097.3`, `1998.7`, `1994.6`, and `1990.2 tok/s` at the capped
  `4540`-token length.
- Short reference validation without `KRASIS_NO_GRAPH` matched the legacy HF
  BF16 artifact for turn 1 exactly and turn 2 first token exactly. The graph
  gate test logs are [server stdout](20260605_165746_gemma4_int4_graphgate_server.log),
  [server log](20260605_165746_gemma4_int4_graphgate_krasis.log), and
  [reference outputs](20260605_gemma4_graphgate_reference/turn1_actual.json).
- Longer exact-match validation against the legacy HF BF16 artifact passed
  turns 1-6 and diverged on longer generations. Outputs remained coherent and
  top-token differences were often close; Gemma4 INT4 should not be claimed as
  witness-validated until a llama-witness/GGUF reference exists.
- INT8 Gemma4 is not supported by this baseline. Diagnostic INT8 probes
  produced invalid token/logprob output, so INT8 remains blocked.

## Diagnostics - 2026-06-05 (Gemma4 compressed KV and ring-window)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the runs.
These are validation diagnostics on branch `gemma-dev`, not clean speed
benchmarks.

| Run | Config / command | KV | Ring-window | Context | Result | Health | Logs |
|-----|------------------|----|-------------|--------:|--------|--------|------|
| Gemma4 k6v6 default gate | `./dev run tests/gemma-4-4-k6v6-a16.conf --test-endpoints`; `./dev network 18013` | k6v6 | off | 10,624 | short network `14/14` passed | clean | [server](20260605_183523_gemma4_k6v6_nonring_gate_startcheck.log) |
| Gemma4 k4v4 default gate | `./dev run tests/gemma-4-4-k4v4-a16.conf --test-endpoints`; `./dev network 18014` | k4v4 | off | 14,880 | short network `14/14` passed | clean | [server](20260605_184024_gemma4_k4v4_nonring_gate_startcheck.log) |
| Gemma4 k6v6 ring initial diagnostic | `./dev run tests/gemma-4-4-k6v6-ring-a16.conf --test-endpoints`; equivalent large-network validation on port 18013 | k6v6 | on | 106,784 | large prompt output was garbled at 25K/100K despite script pass | clean | [startup](20260605_180611_gemma4_k6v6_ring_startcheck.log), [large](20260605_182246_gemma4_k6v6_ring_largecheck.log) |
| Gemma4 k6v6 ring fixed 25K validation | `./dev run tests/gemma-4-4-k6v6-ring-a16.conf --test-endpoints`; `./dev network 18015 --large` stopped after 25K | k6v6 | on | 106,784 | `large_25k` passed; 25K prefill `421.6 tok/s`, decode `10.2 tok/s` | clean | [server](20260605_214730_gemma4_k6v6_ring_default_fixed_25k_validation.log), [network](20260605_214730_gemma4_k6v6_ring_default_fixed_network_large.log) |
| Gemma4 k6v6 ring FA2 full-layer follow-up | `KRASIS_STARTUP_CAL_LONG_TOKENS=12000 ./dev run tests/gemma-4-4-k6v6-ring-a16.conf --test-endpoints`; `./dev network 18015 --large` | k6v6 | on | 106,784 | `large_25k` passed; 25K prefill `480.1 tok/s`, decode `10.2 tok/s`; 100K client timed out while server stayed GPU-bound | clean through 25K | [timing baseline](20260606_075511_gemma4_k6v6_ring_timing_baseline_server.log), [graph failure](20260606_080037_gemma4_k6v6_ring_fastpath_server.log), [server](20260606_080721_gemma4_k6v6_ring_fa2_guarded_server.log), [network](20260606_080721_gemma4_k6v6_ring_fa2_guarded_network_large.log) |
| Gemma4 k6v6 ring tiled decode follow-up | `KRASIS_STARTUP_CALIBRATION_LONG_TOKENS_CAP=12000 ./dev run tests/gemma-4-4-k6v6-ring-a16.conf --port 18015`; `./dev network 18015 --large` stopped after 25K | k6v6 | on | 106,784 | `large_25k` passed; 25K prefill `498.2 tok/s`, decode `28.4 tok/s`; 87K remained impractical and was stopped | clean through 25K | [server](20260606_083226_gemma4_k6v6_ring_decode_tiled_server.log), [network](20260606_083226_gemma4_k6v6_ring_decode_tiled_network_large.log), [rejected export-grid server](20260606_084320_gemma4_k6v6_ring_prefill_exportfix_server.log), [rejected export-grid network](20260606_084320_gemma4_k6v6_ring_prefill_exportfix_network_large.log) |
| Gemma4 k6v6 ring temp-KV cap rejected | `KRASIS_STARTUP_CALIBRATION_LONG_TOKENS_CAP=12000 ./dev run tests/gemma-4-4-k6v6-ring-a16.conf --port 18015`; `./dev network 18015 --large` stopped after 25K | k6v6 | on | 106,784 | `large_25k` passed but prefill regressed to `470.4 tok/s`; calibration prefill growth dropped to `359.0 KB/tok` | clean through 25K; rejected | [server](20260606_141847_gemma4_k6v6_ring_tempcap_server.log), [network](20260606_141847_gemma4_k6v6_ring_tempcap_network_large.log) |
| Gemma4 k6v6 ring temp-KV cap + chunk4096 rejected | `KRASIS_STARTUP_CALIBRATION_LONG_TOKENS_CAP=12000 KRASIS_PREFILL_DIAG_MAX_CHUNK_TOKENS=4096 ./dev run tests/gemma-4-4-k6v6-ring-a16.conf --port 18015`; `./dev network 18015 --large` stopped after 25K | k6v6 | on | 106,784 | `large_25k` passed but prefill regressed to `296.6 tok/s`; calibration prefill growth dropped to `155.3 KB/tok` | clean through 25K; rejected | [server](20260606_142800_gemma4_k6v6_ring_tempcap_chunk4096_server.log), [network](20260606_142800_gemma4_k6v6_ring_tempcap_chunk4096_network_large.log) |
| Gemma4 k6v6 ring direct compressed prefill rejected | `CFG_RING_WINDOW_KV=1 ./dev run tests/gemma-4-4-k6v6-ring-a16.conf --benchmark` | k6v6 | on | 106,784 | startup calibration completed safely with prefill growth `345.1 KB/tok`, but benchmark warmup took `213.0s` and the first 1K timed prefill row did not finish within the bounded wait | clean before stop; rejected and reverted | [server](20260606_184741_gemma4_k6v6_ring_direct_server.log) |
| Gemma4 k4v4 ring rejected | `./dev run tests/gemma-4-4-k4v4-a16.conf --ring-window-kv --test-endpoints` | k4v4 | on | n/a | rejected before model load after 25K validation produced `<unused6226>` | n/a | [guard](20260605_215941_gemma4_k4v4_ring_guard_negative.log), [failed validation](20260605_215349_gemma4_k4v4_ring_default_fixed_25k_validation.log) |

Findings:
- Gemma4 now supports variable per-layer compressed KV allocation and pointer
  registration for k6v6/k4v4. The default configs keep full physical KV per
  layer and do not enable ring-window.
- k6v6 ring-window now passes the formerly failing 25K large-prompt row after
  two fixes: prefill chunked GQA now receives real chunk starts, and
  ring-capped sliding layers use the custom local-window prefill path. The path
  remains explicit/diagnostic until 100K output and witness validation are done.
- The 2026-06-06 timing pass found the old k6 ring custom attention path
  dominated long prefill (`39920` tokens in `151956 ms`, `263 tok/s`) and long
  decode GQA attention (`123.42 ms/tok` of `174.41 ms/tok` during the timing-on
  calibration). The accepted speed patch uses bounded FA2 staging for ring
  sliding-layer prefill where available and keeps Gemma full-attention prefill
  layers out of the custom ring branch.
  Clean timing-off 25K prefill improved to `480.1 tok/s`; decode stayed
  `10.2 tok/s` because Gemma's five full-attention layers still scale with
  prompt length. A CUDA graph decode experiment hit `CUDA_ERROR_ILLEGAL_ADDRESS`
  during long calibration and was removed from the shared graph path.
- The tiled decode follow-up reused the existing graph-compatible compressed
  GQA tiled kernels from the ungraphed decode path rather than enabling graph
  capture. This raised 25K decode from `10.2 tok/s` to `28.4 tok/s` with clean
  health. Prefill remains slow: the accepted run was `498.2 tok/s` at 25K,
  while a low-risk-looking export-grid change regressed to `470.2 tok/s` and
  was reverted. A broader FP8 stage-cache skip produced invalid 25K output and
  was also rejected. A later ring-aware temp-KV cap improved the memory model
  but not speed (`470.4 tok/s`), and forcing 4096-token chunks improved memory
  headroom while reducing 25K prefill to `296.6 tok/s`; both were reverted.
- A direct compressed k6 ring prefill prototype also failed the speed bar. It
  avoided enough BF16 staging to reduce calibration prefill growth to
  `345.1 KB/tok`, but benchmark warmup took `213.0s` and the first 1K timed
  prefill row did not complete within the bounded wait. The active
  kernel/routing code was removed; the accepted k6 ring baseline remains the
  tiled-decode row above.
- k4v4 ring-window remains unsafe: after the k6 fix it still produced
  `<unused6226>` on the 25K large-prompt row and later hit request-time VRAM
  pressure when the suite advanced into the 100K case. The mode is rejected
  before model load; use k6v6 ring-window or k4v4 without ring-window.

## Diagnostics - 2026-06-06 (Qwen35 vs Gemma4 timing attribution)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the runs.
Timing instrumentation was enabled with `KRASIS_PREFILL_TIMING=1`,
`KRASIS_BENCHMARK_PREFILL_BREAKDOWN=1`, and `--timing`, so these rows are
diagnostic attribution runs rather than clean speed benchmarks.

| Run | Command | Result | Key attribution | Logs |
|-----|---------|--------|-----------------|------|
| Qwen35 HQQ6 k6v6 diagnostic | `KRASIS_PREFILL_TIMING=1 KRASIS_BENCHMARK_PREFILL_BREAKDOWN=1 ./dev benchmark tests/q35b-4-4-hqq6-k6v6-diagnostic.conf --timing` | `9593.2 tok/s` internal prefill, `113.92 tok/s` internal decode, HCS `10240/10240`, min free `9556 MB` | 10K prefill total `758.2 ms`; KV append `0.3 ms` over 10 GQA layers. Decode used CUDA graph replay at about `8.6-8.8 ms/tok`, with `100%` HCS hits and no cold DMA. | [timing log](20260606_225805_q35_hqq6_k6v6_timing_compare.log) |
| Gemma4 k6v6 non-ring diagnostic | `KRASIS_PREFILL_TIMING=1 KRASIS_BENCHMARK_PREFILL_BREAKDOWN=1 ./dev benchmark tests/gemma-4-4-k6v6-a16.conf --timing` | `4945.1 tok/s` best internal prefill at 1K, about `1004.6 tok/s` at 10K, `36.60 tok/s` internal decode, HCS `3840/3840`, min free `11170 MB` | 10K prefill total `9920.2 ms`; stage-exact KV append `9404.8 ms` over 30 GQA layers. Decode was ungraphed: about `23.0 ms/tok`, with ~`15.6 ms/tok` MoE and ~`5.5 ms/tok` GQA, despite `100%` HCS hits. | [timing log](20260606_230127_gemma4_k6v6_nonring_timing_compare.log) |
| Gemma4 timing-bucket smoke | `KRASIS_PREFILL_TIMING=1 ./dev run tests/gemma-4-4-k6v6-a16.conf --test-endpoints` | accounting diagnostic only; stopped after server-ready smoke | After the attribution fix, an 8419-token calibration row reported `6434.9 ms` GQA/attention, including `6304.7 ms` KV append, and `63.2 ms` MoE. This confirms the Gemma prefill bottleneck is KV/GQA staging, not MoE. | [smoke log](20260606_231300_gemma4_timing_bucket_smoke.log) |

Findings:
- Gemma4 prefill is not slow because HCS is missing. In the final benchmark
  rows HCS hit rate was `100%` and cold DMA was zero.
- The dominant Gemma4 prefill gap is the current stage-exact KV append path:
  Gemma writes all 30 GQA layers into temporary KV during prefill, and its
  per-token KV geometry is much larger than Qwen35's hybrid GQA stack.
- The top-level Gemma prefill timing attribution was fixed after the comparison
  run. The diagnostic smoke now charges nested Gemma GQA/MoE timers into the
  top-level buckets and confirms the long calibration row is dominated by
  GQA/KV append (`6304.7 ms` of `6434.9 ms` GQA/attention).
- The Gemma4 decode gap is separate: Qwen35 uses CUDA graph replay, while
  Gemma4 remains on the ungraphed path because the attempted Gemma graph path
  previously produced a CUDA illegal address.
- No local Gemma GGUF exists under `~/.krasis/models`, so llama-witness output
  comparison cannot be run until a Gemma GGUF/witness artifact is downloaded or
  produced.

## Diagnostics - 2026-06-07 (Gemma4 prefill and CUDA graph attempts)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the runs.
The prefill rows with timing enabled are diagnostic attribution runs, not clean
speed benchmarks. The final guard-restored row is the accepted timing-off
validation.

| Run | Command | Result | Decision | Logs |
|-----|---------|--------|----------|------|
| BF16 temp-KV staging | `KRASIS_PREFILL_TIMING=1 ./dev run tests/gemma-4-4-k6v6-a16.conf --test-endpoints` | 8419-token calibration `kv_append=6679.2 ms` over 30 calls, worse than the prior `6304.7 ms` | rejected and reverted | [log](20260607_210627_gemma4_k6v6_bf16_stage_timing_smoke.log) |
| FP8-window FA2 route | `KRASIS_PREFILL_TIMING=1 ./dev run tests/gemma-4-4-k6v6-ring-a16.conf --test-endpoints` | 39,920-token timing row `142313.8 ms` total (`281 tok/s`), `kv_append=141115.7 ms`; route did not help the single-chunk long calibration | rejected and removed | [log](20260607_211701_gemma4_k6v6_ring_fp8_window_timing_smoke.log) |
| Vectorized FP8 append | `KRASIS_PREFILL_TIMING=1 ./dev run tests/gemma-4-4-k6v6-ring-a16.conf --test-endpoints` | 39,920-token timing row `142180.8 ms` total (`281 tok/s`), `kv_append=140982.8 ms` | rejected and reverted | [log](20260607_212521_gemma4_k6v6_ring_vec_append_timing_smoke.log) |
| Direct decode-KV single-chunk bypass | `KRASIS_PREFILL_TIMING=1 ./dev run tests/gemma-4-4-k6v6-ring-a16.conf --test-endpoints` | 39,920-token timing row `142272.6 ms` total (`281 tok/s`), `kv_append=141074.8 ms` | rejected and reverted | [log](20260607_213548_gemma4_k6v6_ring_direct_decodekv_timing_smoke.log) |
| Gemma4 CUDA graph decode attempt | `./dev test tests/gemma-4-4-k6v6-a16.conf` with Gemma graph guard temporarily lifted | decode improved to roughly `64-73 tok/s`, but correctness failed: benchmark hit EOS at 4 tokens and network passed only `2/10` prompts with garbled/control-character output | rejected; guard restored | [log](20260607_215044_gemma4_k6v6_graph_test.log) |
| Gemma4 guard-restored validation | `./dev test tests/gemma-4-4-k6v6-a16.conf` | benchmark best `5035.9 tok/s` prefill, `38.72 tok/s` internal decode, `67.57 tok/s` HTTP; network `14/14`; HCS `3840/3840`; min free `11170 MB` | accepted restored state | [log](20260607_222707_gemma4_k6v6_guard_restored_test.log) |

Findings:
- The attempted low-risk prefill fixes did not improve the measured bottleneck.
  BF16 temp staging regressed the short diagnostic, and the ring-window FP8
  append/direct-write variants still spent about `141 s` in KV append on the
  39,920-token timing row.
- Gemma4 CUDA graph replay can improve raw decode speed, but the current graph
  segmenter is not semantically correct for Gemma4's dense-MLP-plus-routed-MoE
  layer. The guard is restored until Gemma has split graph segmentation for
  dense branch, router, expert branch, merge, and `layer_scalar`.
- The accepted code removes the unused FP8-window sidecar symbol/loader and
  keeps existing graph-capable models off the failed Gemma graph experiment.

---

## Standard Benchmarks - 2026-06-08 (Gemma4 CUDA graph decode)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the runs.
Timing instrumentation was disabled for benchmark rows. The QCN row is the
fixed `./dev speed-test` guard to check shared graph-path behavior.

| Run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Health | Logs |
|-----|---------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|--------|------|
| Gemma4 k6v6 graph decode endpoint gate | `./dev run tests/gemma-4-4-k6v6-a16.conf --test-endpoints`; `./dev network 18013` | BF16 | k6v6 | n/a | server rows mostly `57-63` | n/a | 3840/3840 (100.0%) | 11124 MB endpoint low-water | network `14/14`, `0` copy failures | [server](20260608_084624_gemma4_k6v6_graph_perlayer_endpoint.log) |
| Gemma4 k6v6 graph decode benchmark | `./dev benchmark tests/gemma-4-4-k6v6-a16.conf` | BF16 | k6v6 | 5230.3 | 63.92 | 116.20 | 3840/3840 (100.0%) | 11156 MB | clean, `0` copy failures | [full log](20260608_084936_gemma4_k6v6_graph_benchmark.log) |
| Qwen3-Coder-Next HQQ4 k4v4 speed-test guard | `./dev speed-test` | HQQ4 | k4v4 | 6664.9 | 88.42 | 138.06 | 15957/24576 (64.9%) | 896 MB | clean, `0` copy failures; VRAM pressure stayed above 600 MB safety | [full log](20260608_085446_qcn_speed_guard_after_gemma_graph.log) |

Findings:
- Gemma4 non-ring k6v6 decode improved from the guard-restored ungraphed
  baseline (`38.72 tok/s`) to `63.92 tok/s` internal decode after adding
  Gemma-specific graph segmentation and per-layer graph sequence lengths.
- Gemma graph decode is limited in code to the validated non-ring `k6v6` path.
  k4v4 and ring-window Gemma still require separate validation before graph
  replay is enabled for them.
- The short endpoint suite passed `14/14` after the per-layer sequence-length
  fix. The earlier unsafe graph attempt failed because it replayed generic MoE
  semantics; the accepted path explicitly handles Gemma dense branch,
  routed-expert input, merge norms, router scaling, per-expert scale, and
  `layer_scalar`.
- QCN HQQ4/k4v4 speed-test did not regress relative to the latest indexed
  June 4 HQQ4 guards (`85.34` and `83.27` internal decode). This run reached
  `88.42 tok/s` decode with no copy failures. Low-water VRAM was `896 MB`,
  close to but above the default `600 MB` safety margin, so HCS is still
  operating near the intended boundary.

---

## Standard Benchmarks - 2026-06-09 (Gemma4 k4v4 CUDA graph decode)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the runs.
Timing instrumentation was disabled for accepted benchmark rows. The QCN row
uses the fixed `./dev speed-test` regression guard.

| Run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Health | Logs |
|-----|---------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|--------|------|
| Gemma4 k4v4 ungraphed baseline | `./dev test tests/gemma-4-4-k4v4-a16.conf` | BF16 | k4v4 | 5046.9 | 38.79 | 66.92 | 3840/3840 (100.0%) | 11044 MB | network `14/14`, `0` copy failures | [full log](20260608_235412_gemma4_k4v4_baseline_test.log) |
| Gemma4 k4v4 broad graph gate rejected | `./dev test tests/gemma-4-4-k4v4-a16.conf` | BF16 | k4v4 | 5021.2 | 62.95 | 114.82 | 3840/3840 (100.0%) | 11030 MB | rejected: pre-HCS calibration logged graph replay errors | [full log](20260609_000232_gemma4_k4v4_graph_test.log) |
| Gemma4 k4v4 HCS-gated graph decode | `./dev test tests/gemma-4-4-k4v4-a16.conf` | BF16 | k4v4 | 5192.4 | 63.69 | 115.47 | 3840/3840 (100.0%) | 11030 MB | network `14/14`, `0` copy failures, clean error scan | [full log](20260609_000935_gemma4_k4v4_graph_hcsguard_test.log) |
| Qwen3-Coder-Next HQQ4 k4v4 speed-test guard | `./dev speed-test` | HQQ4 | k4v4 | 6856.8 | 88.67 | 149.12 | 15957/24576 (64.9%) | 896 MB | clean, `0` copy failures; VRAM pressure stayed above 600 MB safety | [full log](20260609_001711_qcn_speed_guard_after_gemma_k4_graph.log) |

Findings:
- Gemma4 non-ring k4v4 decode now uses the same Gemma-specific graph segment
  as k6v6, but only after HCS has populated expert residency. This avoids the
  rejected pre-HCS graph replay failure while preserving clean startup
  calibration.
- The accepted k4v4 graph row improves internal decode from `38.79` to
  `63.69 tok/s` and HTTP from `66.92` to `115.47 tok/s`, while retaining the
  compact k4v4 `14880`-token context with a 1000 MB KV cache.
- Ring-window Gemma remains ungraphed. The ring path still needs separate
  correctness and speed validation before graph replay is allowed there.
- QCN speed-test remained in the expected range with `0` copy failures and
  a decode low-water of `896 MB`, above the default `600 MB` safety margin.

---

## Standard Benchmarks - 2026-06-10 (Gemma4 fixed HQQ attention)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the runs.
Timing instrumentation was disabled for accepted benchmark rows. The QCN row
uses the fixed `./dev speed-test` regression guard.

| Run | Command | Attention | KV | Context | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Health | Logs |
|-----|---------|-----------|----|--------:|----------------:|---------------:|-------------------:|-----|--------------:|--------|------|
| Gemma4 HQQ8 k6v6 fused-descriptor reject | `./dev test tests/gemma-4-4-hqq8-k6v6-a16.conf` | HQQ8 | k6v6 | n/a | n/a | n/a | n/a | n/a | n/a | rejected: `D2D HQQ fused K split[5]: CUDA_ERROR_INVALID_VALUE` during calibration | [full log](20260610_075334_gemma4_hqq8_k6v6_test.log) |
| Gemma4 HQQ8 k6v6 split descriptors | `./dev test tests/gemma-4-4-hqq8-k6v6-a16.conf` | HQQ8 | k6v6 | 10,624 | 1838.4 | 66.56 | 119.62 | 3840/3840 (100.0%) | 11228 MB | network `14/14`, `0` copy failures | [full log](20260610_075630_gemma4_hqq8_k6v6_split_test.log) |
| Gemma4 HQQ6 k6v6 | `./dev test tests/gemma-4-4-hqq6-k6v6-a16.conf` | HQQ6 | k6v6 | 10,624 | 1632.5 | 63.85 | 116.36 | 3840/3840 (100.0%) | 11368 MB | network `14/14`, `0` copy failures | [full log](20260610_080221_gemma4_hqq6_k6v6_test.log) |
| Gemma4 HQQ4 k6v6 | `./dev test tests/gemma-4-4-hqq4-k6v6-a16.conf` | HQQ4 | k6v6 | 10,624 | 1706.5 | 65.39 | 115.78 | 3840/3840 (100.0%) | 11748 MB | network `14/14`, `0` copy failures | [full log](20260610_080755_gemma4_hqq4_k6v6_test.log) |
| Gemma4 HQQ8 k4v4 | `./dev test tests/gemma-4-4-hqq8-k4v4-a16.conf` | HQQ8 | k4v4 | 14,880 | 1590.0 | 65.42 | 119.29 | 3840/3840 (100.0%) | 10954 MB | network `14/14`, `0` copy failures | [full log](20260610_081953_gemma4_hqq8_k4v4_test.log) |
| Gemma4 HQQ6 k4v4 | `./dev test tests/gemma-4-4-hqq6-k4v4-a16.conf` | HQQ6 | k4v4 | 14,880 | 1628.2 | 64.43 | 118.62 | 3840/3840 (100.0%) | 11094 MB | network `14/14`, `0` copy failures | [full log](20260610_082537_gemma4_hqq6_k4v4_test.log) |
| Gemma4 HQQ4 k4v4 | `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | 14,880 | 2150.7 | 65.25 | 120.10 | 3840/3840 (100.0%) | 11474 MB | network `14/14`, `0` copy failures; code-gen sample visibly weaker | [full log](20260610_083136_gemma4_hqq4_k4v4_test.log) |
| Qwen3-Coder-Next HQQ4 k4v4 speed-test guard | `./dev speed-test` | HQQ4 | k4v4 | 136,528 | 6902.6 | 89.37 | 149.29 | 15957/24576 (64.9%) | 928 MB | clean, `0` copy failures; VRAM pressure stayed above 600 MB safety | [full log](20260610_083712_qcn_speed_guard_after_gemma_hqq.log) |

Findings:
- Gemma4 now supports fixed HQQ4/HQQ6/HQQ8 attention with both non-ring k6v6
  and non-ring k4v4 KV. Mixed/auto HQQ remains rejected for Gemma4, and
  ring-window Gemma remains outside this support matrix.
- The accepted fix is descriptor-level, not a broad runtime fallback: Gemma4
  GQA HQQ registration skips the generic fused-QKV descriptor and registers
  split Q/K/V/O descriptors. Existing Qwen/QCN fused HQQ paths are unchanged.
- HQQ attention reduces Gemma4 attention cache size: HQQ8 validated at about
  `1870 MB` cached, HQQ6 at about `1430 MB`, and HQQ4 at about `990 MB`.
  Lower HQQ bits increased measured free VRAM, but fixed-HQQ attention is much
  slower in prefill than the BF16 Gemma attention path.
- The short network suite passed for all six rows, but HQQ4/k4v4 generated a
  visibly weak recursive code sample despite passing the suite. Do not treat
  Gemma4 HQQ as llama-witness validated.
- QCN `./dev speed-test` stayed in the expected range with `0` copy failures
  and low-water VRAM above the default `600 MB` safety margin, so the Gemma
  split-descriptor change did not regress the standard QCN HQQ path.

---

## Diagnostics - 2026-06-10 (Gemma4 HQQ4/k4v4 prefill speed follow-up)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the runs.
Timing instrumentation was enabled only for the attribution row. The accepted
speed rows used timing-off `./dev test` / `./dev speed-test`.

| Run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Decision | Logs |
|-----|---------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|----------|------|
| Gemma4 HQQ4/k4v4 materialized BF16 prefill | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 materialized BF16 | k4v4 | 1702.7 | 65.49 | 120.33 | 3840/3840 (100.0%) | 11474 MB | rejected: slower than HQQ4/k4v4 baseline | [full log](20260610_174308_gemma4_hqq4_k4v4_materialized_prefill_test.log) |
| Gemma4 HQQ4/k4v4 prefill timing | `KRASIS_PREFILL_TIMING=1 KRASIS_BENCHMARK_PREFILL_BREAKDOWN=1 ./dev run tests/gemma-4-4-hqq4-k4v4-a16.conf --test-endpoints --timing` | HQQ4 | k4v4 | diagnostic only | diagnostic only | n/a | n/a | n/a | attribution: `11824` tokens spent `13091.5 ms` in KV append inside `13521.6 ms` GQA | [timing log](20260610_174921_gemma4_hqq4_k4v4_prefill_timing.log) |
| Gemma4 HQQ4/k4v4 no stage-exact diagnostic | `KRASIS_KV_STAGE_EXACT=0 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | 2272.8 | 64.63 | 116.75 | 3840/3840 (100.0%) | 11474 MB | diagnostic: faster single-chunk path, not safe as a global switch | [full log](20260610_175214_gemma4_hqq4_k4v4_no_stage_exact_test.log) |
| Gemma4 HQQ4/k4v4 guarded single-chunk skip | `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` | HQQ4 | k4v4 | 2303.8 | 65.10 | 117.49 | 3840/3840 (100.0%) | 11474 MB | accepted: skips stage-exact only for validated Gemma4 HQQ4/k4v4 single-chunk fit | [full log](20260610_181558_gemma4_hqq4_k4v4_prefill_skip_final_test.log) |
| Qwen3-Coder-Next HQQ4 k4v4 speed-test guard | `./dev speed-test` | HQQ4 | k4v4 | 6356.9 | 90.14 | 148.29 | 15957/24576 (64.9%) | 928 MB | clean, `0` copy failures; VRAM pressure stayed above 600 MB safety | [full log](20260610_180754_qcn_speed_guard_after_gemma_hqq_prefill_skip.log) |

Findings:
- Gemma4 HQQ4/k4v4 prefill remains GQA/KV-append bound. Materializing HQQ
  weights to BF16 before prefill was slower, so HQQ projection cost is not the
  main bottleneck.
- The accepted code does not globally disable stage-exact KV. It skips the
  temporary stage only when runtime scratch budgeting shows the whole prompt
  fits as one chunk and only for the validated non-ring Gemma4 HQQ4/k4v4
  surface. Multi-chunk prompts, k6v6, HQQ6/HQQ8, ring-window, and non-Gemma
  models keep the existing stage-exact path.
- The improvement is modest (`2150.7` to `2303.8 tok/s` best prefill) because
  long rows are still dominated by Gemma's 30 GQA-layer KV work.
- QCN `./dev speed-test` passed after the change with `0` copy failures and
  low-water VRAM close to, but above, the default `600 MB` safety margin.

---

## Diagnostics - 2026-06-11 (Gemma4 BF16/k4v4 fast-path probes)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the runs.
Timing instrumentation was enabled only for the graph attribution rows. Speed
comparison rows used timing-off `./dev test`.

| Run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Decision | Logs |
|-----|---------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|----------|------|
| Gemma4 BF16/k4v4 baseline | `./dev test tests/gemma-4-4-k4v4-a16.conf` | BF16 | k4v4 | 5196.5 | 64.36 | 115.43 | 3840/3840 (100.0%) | 11030 MB | baseline, network `14/14` | [full log](20260611_063100_gemma4_bf16_k4v4_stage_exact_baseline_test.log) |
| Gemma4 BF16/k4v4 no stage-exact | `KRASIS_KV_STAGE_EXACT=0 ./dev test tests/gemma-4-4-k4v4-a16.conf` | BF16 | k4v4 | 5059.5 | 63.68 | 114.45 | 3840/3840 (100.0%) | n/a | rejected: lower transient VRAM but slower than baseline | [full log](20260611_063628_gemma4_bf16_k4v4_no_stage_exact_diag_test.log) |
| Gemma4 BF16/k4v4 graph timing | `KRASIS_DECODE_TIMING=1 ./dev run tests/gemma-4-4-k4v4-a16.conf --test-endpoints --timing` + `./dev network 18014` | BF16 | k4v4 | diagnostic only | diagnostic only | n/a | 3840/3840 (100.0%) | n/a | attribution: post-HCS graph mode already active; 79-token block `15.59 ms/tok`, GQA route `9.31 ms/tok`, final `1.05 ms/tok`, cold DMA `0` | [timing log](20260611_064201_gemma4_bf16_k4v4_decode_graph_timing.log), [network log](20260611_064623_gemma4_bf16_k4v4_decode_graph_network_timing.log) |
| Gemma4 BF16/k4v4 GPU route-sync | `KRASIS_GPU_ROUTE_SYNC=1 ./dev test tests/gemma-4-4-k4v4-a16.conf` | BF16 | k4v4 | 5051.3 | 62.64 | 113.71 | 3840/3840 (100.0%) | 11030 MB | rejected: network `14/14` but slower decode | [full log](20260611_064741_gemma4_bf16_k4v4_gpu_route_sync_test.log) |

Findings:
- The accepted HQQ4/k4v4 stage-exact skip should not be broadened to the BF16
  Gemma fast path. On BF16/k4v4, disabling stage-exact reduced long-prefill
  transient memory (`5586 MB` to `4306 MB`) but did not improve speed.
- Gemma4 BF16/k4v4 decode is already in CUDA graph mode post-HCS. The remaining
  measured graph-mode cost is mostly GQA route graph segments and the final
  segment, not an obvious ungraphed dense/MLP section.
- The existing GPU route-sync path is not a Gemma decode speed win on this
  surface. It passed the short network suite but regressed internal decode from
  `64.36` to `62.64 tok/s`.
- No new BF16 fast-path runtime change was accepted from these probes.

---

## Diagnostics - 2026-06-12 (Gemma4 BF16/k4v4 KV-staging kernels)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the runs.
Timing instrumentation was enabled for attribution rows only. The acceptance
row used timing-off `./dev test`.

| Run | Command | Attention | KV | Long-row KV append | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Decision | Logs |
|-----|---------|-----------|----|-------------------:|----------------:|---------------:|-------------------:|-----|--------------:|----------|------|
| Gemma4 BF16/k4v4 KV-stage timing baseline | `KRASIS_PREFILL_TIMING=1 KRASIS_BENCHMARK_PREFILL_BREAKDOWN=1 KRASIS_KV_STAGE_DIAG=1 ./dev run tests/gemma-4-4-k4v4-a16.conf --test-endpoints --timing` | BF16 | k4v4 | `12829.3 ms` | diagnostic only | diagnostic only | n/a | 3840/3840 (100.0%) | n/a | attribution: 11,824-token row spent `12829.3 ms` in temp FP8 KV append inside `13007.6 ms` GQA; temp FP8 K/V `1270.2 MB` | [full log](20260612_171741_gemma4_bf16_k4v4_kv_stage_timing_baseline.log) |
| Gemma4 BF16/k4v4 2D FP8 append launch | same timing command after 2D token x KV-segment launch candidate | BF16 | k4v4 | `12836.7 ms` | diagnostic only | diagnostic only | n/a | 3840/3840 (100.0%) | n/a | rejected: unchanged/slightly slower long-row append and total row (`13219.5 ms`) | [full log](20260612_172318_gemma4_bf16_k4v4_kv_stage_2d_append_timing.log) |
| Gemma4 BF16/k4v4 FP8x2 append timing | same timing command after `__nv_fp8x2_e4m3` append candidate | BF16 | k4v4 | `12505.5 ms` | diagnostic only | diagnostic only | n/a | 3840/3840 (100.0%) | n/a | candidate only: instrumentation improved long-row append, required timing-off validation | [full log](20260612_172923_gemma4_bf16_k4v4_kv_stage_fp8x2_timing.log) |
| Gemma4 BF16/k4v4 FP8x2 append validation | `./dev test tests/gemma-4-4-k4v4-a16.conf` | BF16 | k4v4 | n/a | 5033.4 | 63.23 | 115.50 | 3840/3840 (100.0%) | 11030 MB | rejected: network `14/14` but prefill regressed vs `5196.5` baseline; candidate reverted | [full log](20260612_173149_gemma4_bf16_k4v4_fp8x2_append_test.log) |

Findings:
- The measured bottleneck is real: Gemma4 BF16/k4v4 stage-exact prefill writes
  temporary FP8 K/V for every active GQA layer, and the long row spends almost
  all GQA time in that append.
- Micro-optimizing the generic FP8 append kernel did not produce an acceptable
  timing-off speed win. The vectorized FP8x2 candidate looked better under
  instrumentation but regressed the normal `./dev test` benchmark, so it was
  reverted.
- The direct compressed append path is still not accepted for BF16 Gemma:
  the earlier `KRASIS_KV_STAGE_EXACT=0` test saved transient memory but was
  slower. A larger redesign, such as a runtime-budgeted BF16 temp stage or a
  fused attention/export path, would need separate implementation and
  validation rather than a small kernel launch tweak.
- No new KV-staging runtime change was accepted from this pass.

---

## Diagnostic Benchmarks - 2026-06-04 (Q122B HQQ6+k4v4 prefill recovery)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the run.
These were instrumented diagnostics on `main` at `4a40cca`, using
`./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf` with
`KRASIS_VRAM_LEDGER=1`, `KRASIS_PREFILL_DEBUG=1`, and
`KRASIS_PREFILL_TIMING=1`. They are not clean speed benchmarks.

| Run | Pinning | Prefill | Decode | HTTP | HCS | Min free | Health | Logs |
|-----|---------|--------:|-------:|-----:|-----|---------:|--------|------|
| Q122B current-main component timing | on | 2272.0 | 27.76 | 50.89 | 4077/12288 (33.2%) | 682 MB | clean | [full log](20260604_q122b_hqq6_k4v4_current_main_prefill_component_timing.log), [report](20260604_q122b_hqq6_k4v4_current_main_prefill_component_timing_report.log), [server log](20260604_q122b_hqq6_k4v4_current_main_prefill_component_timing_krasis.log) |
| Q122B current-main component timing, `KRASIS_NO_PINNING=1` | off | 3032.4 | 27.91 | 50.34 | 4077/12288 (33.2%) | 646 MB | one warning: 598 MB free during warmup | [full log](20260604_q122b_hqq6_k4v4_current_main_no_pinning_component_timing.log), [report](20260604_q122b_hqq6_k4v4_current_main_no_pinning_component_timing_report.log), [server log](20260604_q122b_hqq6_k4v4_current_main_no_pinning_component_timing_krasis.log) |
| Q122B tail smoothing + dense-active pinning skip + post-scratch floor | adaptive | 2992.1 | 28.27 | 49.45 | 4077/12288 (33.2%) | 678 MB | clean | [full log](20260604_2015_q122b_hqq6_k4v4_tail_dense_floor_component_timing.log), [report](20260604_q122b_hqq6_k4v4_tail_dense_floor_component_timing_report.log), [server log](20260604_q122b_hqq6_k4v4_tail_dense_floor_component_timing_krasis.log) |
| Q122B measured post-scratch reserve | adaptive | 3401.7 | 28.31 | 48.47 | 4050/12288 (33.0%) | 776 MB | clean | [full log](20260604_q122b_hqq6_k4v4_post_scratch_reserve_diag.log) |
| Q122B MoE accumulator scratch shrink | adaptive | 3304.1 | 27.60 | 49.84 | 4050/12288 (33.0%) | 772 MB | clean | [full log](20260604_q122b_hqq6_k4v4_moe_accum_shrink_diag.log) |
| Q122B HQQ runtime-stage breakdown | adaptive | 3653.6 | 27.75 | 49.34 | 4050/12288 (33.0%) | 806 MB | clean | [full log](20260604_q122b_hqq6_k4v4_runtime_stage_breakdown_diag.log), [report](20260604_q122b_hqq6_k4v4_runtime_stage_breakdown_diag_report.log), [server log](20260604_q122b_hqq6_k4v4_runtime_stage_breakdown_diag_krasis.log) |
| Q122B HQQ dual-stage residency experiment | adaptive | 3727.8 | 22.31 | 40.26 | 2862/12288 (23.3%) | 830 MB | clean; rejected | [full log](20260605_q122b_hqq6_k4v4_hqq_dual_stage_diag.log), [report](20260605_q122b_hqq6_k4v4_hqq_dual_stage_diag_report.log), [server log](20260605_q122b_hqq6_k4v4_hqq_dual_stage_diag_krasis.log) |

Findings:
- The no-HCS long calibration chunk still reaches the old speed class:
  `17138` tokens at `4693-4701 tok/s`.
- With pinning on, timed `20K` was `2461 tok/s` and `50K` was `2662 tok/s`;
  MoE DMA dominated (`4026.8 ms` at 20K, `7396.8 ms` at 50K).
- With pinning off, first large chunks returned to `~4.75K-4.79K tok/s`, but
  tail chunks stayed slow (`2947` tail at `1198 tok/s`, `80` tail at
  `34.8 tok/s`), and the run briefly crossed the 600 MB floor (`598 MB`).
- Cross-chunk KV staging was effectively zero-time in these timing rows; the
  remaining regression surface is the dense pointer-table/cold-staging MoE DMA
  policy plus the chunk/tail shape, not basic attention kernel throughput.
- Tail smoothing changed the long prompt plans from pathological tiny tails to
  balanced same-pass-count plans where measured VRAM required it: `35K` became
  `11667 + 11667 + 11666`, and `50K` became `12500 x 4`.
- Dense-active optional pinning skip restored large-chunk prefill to the
  old-speed class for dense Q122B chunks, while the post-scratch floor guard
  kept the instrumented diagnostic above the 600 MB safety floor.
- The measured post-scratch reserve row removed a double count in scratch
  planning: calibration measures `post_alloc_free - min_free`, which already
  includes runtime cold-staging low-water effects, so the allocator now uses
  that measured reserve instead of adding a second full cold-staging reserve
  after calibration. In the diagnostic, `35K` improved to `18245 + 16755`
  chunks and `50K` improved to `18054 + 18054 + 13892`, with no safety warning.
- The MoE accumulator scratch shrink row sized the routed/shared output
  accumulator as `[tokens, hidden]` FP32 instead of routing-width scratch.
  The fused Marlin reduction scratch is already covered by `d_fp32_scratch`.
  This raised the measured-safe calibration probe to `19268` tokens and made
  the `20K` diagnostic row one chunk at `4737 tok/s`, but 35K/50K still paid
  multi-chunk MoE/DMA overhead.
- The HQQ runtime-stage breakdown showed the remaining request-boundary cost:
  `20K` spent `561 ms` preparing prefill and `663 ms` restoring decode, almost
  entirely HQQ stage copies. The core `20K` prefill body was `4156 ms`, close
  to the old May 3 fast path, but the broader request window was `5474 ms`.
- The HQQ dual-stage residency experiment removed those HQQ boundary copies
  after initial installation, but required about `4.9 GB` persistent extra HQQ
  residency, reduced HCS from `4050` to `2862` experts, and regressed decode
  and HTTP. It is rejected as a default optimization because it spends too much
  VRAM and weakens the HCS/safety tradeoff.

---

## Standard Benchmarks - 2026-06-04 (Q122B tail smoothing and dense-active pinning policy)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the run.
Timing instrumentation was disabled. Both rows used
`./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf`.

| Run | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Health | Logs |
|-----|----------------:|---------------:|-------------------:|-----|--------------:|--------|------|
| Q122B tail smoothing + dense-active pinning skip, before reload guard | 3106.2 | 28.62 | 48.78 | 4077/12288 (33.2%) | 648 MB | one warning: 598 MB free during warmup | [full log](20260604_2055_q122b_hqq6_k4v4_tail_dense_floor_clean_benchmark.log), [report](20260604_q122b_hqq6_k4v4_tail_dense_floor_clean_benchmark_report.log), [server log](20260604_q122b_hqq6_k4v4_tail_dense_floor_clean_benchmark_krasis.log) |
| Q122B tail smoothing + dense-active pinning skip + reload guard | 3238.3 | 27.21 | 47.04 | 4050/12288 (33.0%) | 804 MB | clean | [full log](20260604_2004_q122b_hqq6_k4v4_tail_dense_reload_guard_clean_benchmark.log), [report](20260604_q122b_hqq6_k4v4_tail_dense_reload_guard_clean_benchmark_report.log), [server log](20260604_q122b_hqq6_k4v4_tail_dense_reload_guard_clean_benchmark_krasis.log) |
| Q122B measured post-scratch runtime reserve | 3608.7 | 27.76 | 47.99 | 4050/12288 (33.0%) | 806 MB | clean | [full log](20260604_q122b_hqq6_k4v4_post_scratch_reserve_clean.log), [report](20260604_q122b_hqq6_k4v4_post_scratch_reserve_clean_report.log), [server log](20260604_q122b_hqq6_k4v4_post_scratch_reserve_clean_krasis.log) |
| Q122B MoE accumulator scratch shrink | 3900.5 | 27.78 | 45.98 | 4050/12288 (33.0%) | 806 MB | clean | [full log](20260604_q122b_hqq6_k4v4_moe_accum_shrink_clean.log), [stdout](20260604_q122b_hqq6_k4v4_moe_accum_shrink_clean_stdout.log), [report](20260604_q122b_hqq6_k4v4_moe_accum_shrink_clean_report.log), [server log](20260604_q122b_hqq6_k4v4_moe_accum_shrink_clean_krasis.log) |
| Q122B post rejected HQQ-stage experiments | 3796.8 | 28.98 | 46.71 | 4050/12288 (33.0%) | 772 MB | clean | [full log](20260605_q122b_hqq6_k4v4_post_reject_clean.log), [stdout](20260605_q122b_hqq6_k4v4_post_reject_clean_stdout.log), [report](20260605_q122b_hqq6_k4v4_post_reject_clean_report.log), [server log](20260605_q122b_hqq6_k4v4_post_reject_clean_krasis.log) |

Notes:
- The first timing-off run improved prefill but still dipped to `598 MB`, below
  the `600 MB` safety margin. The final row adds a measured HCS reload guard:
  reload stops with one soft-HCS chunk of headroom instead of loading exactly
  to the idle floor.
- The reload guard reduced HCS by one measured soft chunk (`27` experts) versus
  the prior effective post-pressure coverage, and raised minimum free VRAM to
  `804 MB`. Health scan found no CUDA errors, no VRAM monitor warnings, no
  hard-floor exits, and no HCS copy failures.
- The measured post-scratch runtime-reserve row keeps the same HCS coverage and
  safety behavior while raising best clean prefill from `3238.3` to
  `3608.7 tok/s` (`+11.4%`). It remains below the old May 3 Q122B anchor
  (`4880.4 tok/s`) because `20K` still cannot run as one measured-safe chunk
  and dense multi-pass prefill still pays repeated all-expert cold-staging work.
- The MoE accumulator scratch shrink keeps the same HCS coverage and minimum
  free VRAM while raising best clean prefill from `3608.7` to `3900.5 tok/s`
  (`+8.1%`). The `20K` row improved from `2601.9` to `3801.7 tok/s`, showing
  the single-chunk path is restored safely for that size. The remaining gap to
  the May 3 anchor is mostly long-prompt multi-chunk MoE/DMA overhead and
  per-request fixed overhead on 1K/5K/10K rows.
- The post-rejection clean row confirms the tree after removing the HQQ
  dual-residency and async-copy experiments still preserves the accepted
  scratch improvements and safety behavior. It is slightly below the first
  MoE accumulator clean row on prefill, but remains above the measured
  post-scratch-reserve row and kept `772 MB` minimum free VRAM with a clean
  health scan.

---

## Standard Benchmarks - 2026-06-04 (QCN after Typhon pressure-drain and prefill planner fixes)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the run.
Timing instrumentation was disabled. The run used the repeatable
`./dev speed-test` entry point on `main`. The Typhon pressure-drain baseline
was run at `0ceb770`; the prefill planner row was run from current `main`
after reapplying the shelved planner fixes.

| Model / run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Logs |
|-------------|---------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|------|
| Qwen3-Coder-Next HQQ4 k4v4 INT4 mirror speed-test after prefill planner fixes | `./dev speed-test` | HQQ4 | k4v4 | 5054.2 | 85.34 | 117.52 | 16443/24576 (66.9%) | 672 MB | [full log](20260604_qcn_hqq4_k4v4_int4_mirror_speedtest_prefill_planner_fix_stdout.log), [report](20260604_qcn_hqq4_k4v4_int4_mirror_speedtest_prefill_planner_fix_report.log), [server log](20260604_qcn_hqq4_k4v4_int4_mirror_speedtest_prefill_planner_fix_krasis.log) |
| Qwen3-Coder-Next HQQ4 k4v4 INT4 mirror speed-test after Typhon pressure fix | `./dev speed-test` | HQQ4 | k4v4 | 4789.0 | 83.27 | 128.29 | 16362/24576 (66.6%) | 734 MB | [full log](20260604_qcn_hqq4_k4v4_int4_mirror_speedtest_typhon_pressure_fix.log), [report](20260604_qcn_hqq4_k4v4_int4_mirror_speedtest_typhon_pressure_fix_report.log), [server log](20260604_qcn_hqq4_k4v4_int4_mirror_speedtest_typhon_pressure_fix_krasis.log) |

Notes:
- Config confirmation: `attention_quant='hqq4'`, `kv_dtype='k4v4'`,
  `gpu_expert_bits=4`, `cpu_expert_bits=4`, `enable_thinking=False`, and
  `hcs_host_cache_mode='mirror'`.
- The prefill planner row restored extra soft-HCS eviction for one-pass prefill,
  post-scratch chunk-guard eviction, max-safe-plus-tail chunking, and the
  optional-pinning budget fix.
- Startup built/validated HQQ4 attention artifacts (`947 MB` cache). The
  planner row loaded soft HCS in mirror mode at `16443/24576` startup soft
  experts (`25435.3 MB`, `host_mode=mirror`).
- VRAM calibration used the full `39,920`-token long probe. Long prefill
  low-water was `3244 MB`; the planner row timed benchmark decode minimum free
  VRAM was `672 MB`, above the `600 MB` safety margin.
- Health scan found no CUDA errors, no VRAM monitor warnings, no HCS copy
  failures, and no hard-floor exit in the benchmark logs.
- Compared with the immediately prior Typhon pressure-fix speed test
  (`4789.0` prefill, `83.27` decode, `128.29` HTTP), the planner row changed:
  internal prefill `+5.5%`, internal decode `+2.5%`, HTTP `-8.4%`.
- Compared with the latest indexed QCN speed-test gate
  (`20260529_qcn_hqq4_k4v4_int4_mirror_speedtest_v1011_release_gate`:
  `5925.2` prefill, `91.09` decode, `150.62` HTTP), the planner row remained
  lower: internal prefill `-14.7%`, internal decode `-6.3%`, HTTP `-22.0%`.

---

## Standard Benchmarks - 2026-05-27 (QCN HQQ4 speed-test policy)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the run.
Timing instrumentation was disabled. The run used the repeatable
`./dev speed-test` entry point after updating the fixed speed-test surface to
`tests/qcn-k4v4-hqq4-int4-benchmark.conf`: Qwen3-Coder-Next, INT4 experts,
HQQ4 attention, `k4v4` KV, thinking off, and HCS RAM saver off
(`host_mode=mirror`).

| Model / run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Logs |
|-------------|---------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|------|
| Qwen3-Coder-Next HQQ4 k4v4 INT4 mirror speed-test v1.0.11 release gate | `./dev speed-test` | HQQ4 | k4v4 | 5925.2 | 91.09 | 150.62 | 16362/24576 (66.6%) | 734 MB | [full log](20260529_qcn_hqq4_k4v4_int4_mirror_speedtest_v1011_release_gate.log), [report](20260529_qcn_hqq4_k4v4_int4_mirror_speedtest_v1011_release_gate_report.log), [server log](20260529_qcn_hqq4_k4v4_int4_mirror_speedtest_v1011_release_gate_krasis.log) |
| Qwen3-Coder-Next HQQ4 k4v4 INT4 mirror speed-test vision final gate | `./dev speed-test` | HQQ4 | k4v4 | 5783.7 | 89.57 | 144.14 | 16362/24576 (66.6%) | 734 MB | [full log](20260529_qcn_hqq4_k4v4_int4_mirror_speedtest_vision_final_gate.log), [report](20260529_qcn_hqq4_k4v4_int4_mirror_speedtest_vision_final_gate_report.log), [server log](20260529_qcn_hqq4_k4v4_int4_mirror_speedtest_vision_final_gate_krasis.log) |
| Qwen3-Coder-Next HQQ4 k4v4 INT4 mirror speed-test vision branch gate | `./dev speed-test` | HQQ4 | k4v4 | 5754.8 | 90.10 | 151.29 | 16362/24576 (66.6%) | 734 MB | [full log](20260529_qcn_hqq4_k4v4_int4_mirror_speedtest_vision_gate.log), [report](20260529_qcn_hqq4_k4v4_int4_mirror_speedtest_vision_gate_report.log), [server log](20260529_qcn_hqq4_k4v4_int4_mirror_speedtest_vision_gate_krasis.log) |
| Qwen3-Coder-Next HQQ4 k4v4 INT4 mirror speed-test release-fix gate | `./dev speed-test` | HQQ4 | k4v4 | 5748.5 | 91.92 | 150.09 | 16362/24576 (66.6%) | 734 MB | [full log](20260528_qcn_hqq4_k4v4_int4_mirror_speedtest_fixgate.log), [report](20260528_qcn_hqq4_k4v4_int4_mirror_speedtest_fixgate_report.log), [server log](20260528_qcn_hqq4_k4v4_int4_mirror_speedtest_fixgate_krasis.log) |
| Qwen3-Coder-Next HQQ4 k4v4 INT4 mirror speed-test | `./dev speed-test` | HQQ4 | k4v4 | 5752.6 | 90.65 | 162.16 | 16362/24576 (66.6%) | 734 MB | [full log](20260527_qcn_hqq4_k4v4_int4_mirror_speedtest.log), [report](20260527_qcn_hqq4_k4v4_int4_mirror_speedtest_report.log), [server log](20260527_qcn_hqq4_k4v4_int4_mirror_speedtest_krasis.log) |

Notes:
- v1.0.11 release gate comparison against the final vision gate:
  internal prefill `+2.4%`, internal decode `+1.7%`, HTTP `+4.5%`.
  Text-only QCN speed remains within normal variance; timing instrumentation
  was disabled, and no vision request path was loaded during this run. Health
  scan found no runtime errors, VRAM monitor warnings, or HCS copy failures.
- Final vision gate comparison against the earlier same-branch vision gate:
  internal prefill `+0.5%`, internal decode `-0.6%`, HTTP `-4.7%`.
  Core text-only engine speed remains within normal variance; timing
  instrumentation was disabled and no vision request path was loaded during
  this run.
- Vision branch gate comparison against the `v1.0.10` release-fix speed-test
  row: internal prefill `+0.1%`, internal decode `-2.0%`, HTTP `+0.8%`. The
  text-only QCN path remains within normal variance; timing instrumentation was
  disabled and no vision request path was loaded during this run.
- Release-fix gate comparison against the previous speed-test row: internal
  prefill `-0.1%`, internal decode `+1.4%`, HTTP `-7.4%`. The core engine
  speed and decode VRAM floor are within normal run noise; timing
  instrumentation was disabled.
- Config confirmation: `attention_quant='hqq4'`, `kv_dtype='k4v4'`,
  `gpu_expert_bits=4`, `cpu_expert_bits=4`, `enable_thinking=False`, and
  `hcs_host_cache_mode='mirror'`.
- Startup built/validated HQQ4 attention artifacts (`947 MB` cache) and loaded
  soft HCS in mirror mode: `16443/24576` startup soft experts
  (`25435.3 MB`, `host_mode=mirror`).
- VRAM calibration used a 39,920-token long probe. Long prefill low-water was
  `3244 MB`; timed benchmark decode minimum free VRAM was `734 MB`, above the
  `600 MB` safety margin.
- Health scan found no hard exit, no VRAM monitor warning, no copy failures, and
  no warning/error lines in the benchmark logs.

---

## Standard Benchmarks - 2026-05-28 (Zephyrus RTX 3070 Laptop, v1.0.10 RAM saver)

Hardware: Zephyrus, AMD Ryzen 7 5800HS, 31 GB RAM, 1x NVIDIA GeForce RTX
3070 Laptop GPU 8 GB. Timing instrumentation was disabled. The run used the
installed `krasis 1.0.10` command path after `krasis update`, a temporary
thinking-off copy of `/home/main/krasis-ledger/tests/q35b-4-4-hqq4.conf`, and
forced HCS RAM saver/source mode with `--hcs-host-cache-mode source`.

| Model / run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Logs |
|-------------|---------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|------|
| Qwen3.5-35B-A3B Zephyrus RTX 3070 Laptop HQQ4 source | installed `krasis --config /tmp/krasis-zephyrus-q35b-hqq4-thinkingoff-benchmark.conf --benchmark --hcs-host-cache-mode source` | HQQ4 | fp8 | 76.4 | 10.71 | 17.66 | 369/10240 (3.6%) | 676 MB | [full log](20260528_zephyrus_qwen35_hqq4_kvfp8_int4_source_benchmark_stdout.log), [report](20260528_zephyrus_qwen35_hqq4_kvfp8_int4_source_benchmark_report.log), [server log](20260528_zephyrus_qwen35_hqq4_kvfp8_int4_source_benchmark_krasis.log) |

Notes:
- RAM saver/source mode was confirmed in the log: `HCS host cache: source` and
  startup soft HCS `533` experts (`824.5 MB` logical, `host_mode=source`).
- The 8 GB card is extremely constrained: startup calibration stopped long
  calibration at `500` tokens because the short prefill low-water was already
  `696 MB` against the configured `600 MB` safety margin.
- Full benchmark completed without CUDA errors, illegal-address failures, VRAM
  monitor warnings, or HCS source copy failures. Decode low-water was `676 MB`.
- The timed prefill section took about 24.5 minutes for the 1K/5K/10K/20K/35K/50K
  runs; throughput was stable in the `73.0-76.4 tok/s` range.
- Config loaded with `enable_thinking=False`, but the server-ready banner still
  printed `Think: on`; this appears to be a display/reporting issue, consistent
  with the earlier Zephyrus prompt validation.

---

## Standard Benchmarks - 2026-05-27 (Typhon RTX 5080)

Hardware: Typhon, AMD Ryzen 9 5900X, 117 GB RAM, 1x NVIDIA GeForce RTX 5080
16 GB. Timing instrumentation was disabled. The run used the published
`v1.0.7` installed command path and the Typhon Qwen3.6-35B-A3B HQQ4/`k4v4`
benchmark config `/tmp/krasis-j0rq33l2.conf`.

| Model / run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Logs |
|-------------|---------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|------|
| Qwen3.6-35B-A3B Typhon RTX 5080 HQQ4 k4v4 | installed `krasis` benchmark run from `server-run-benchmark_20260527_160458` | HQQ4 | k4v4 | 3743.5 | 60.04 | 128.55 | 5289/10240 (51.7%) | 963 MB | [report](20260527_typhon_qwen36_5080_hqq4_k4v4_benchmark_report.log), [server log](20260527_typhon_qwen36_5080_hqq4_k4v4_krasis.log) |

Notes:
- This is the post-`v1.0.7` installed-release rerun after the prefill pinning
  VRAM-floor fix. It completed with no hard-floor exit.
- Timed prefill increased with prompt length and peaked at the 50K row:
  `1K 252.7`, `5K 1558.6`, `10K 2810.5`, `20K 3471.8`, `35K 3668.9`,
  `50K 3743.5 tok/s`.
- Decode retained `5289/10240` HCS experts with `963 MB` minimum free VRAM
  against the configured `600 MB` safety margin.

---

## Standard Benchmarks - 2026-05-27 (Zephyrus RTX 3070 Laptop)

Hardware: Zephyrus, AMD Ryzen 7 5800HS, 31 GB RAM, 1x NVIDIA GeForce RTX
3070 Laptop GPU 8 GB. Timing instrumentation was disabled. The run used the
current local fixed-residency cleanup build from `/home/main/krasis-ledger` and
the Zephyrus Qwen3.5-35B-A3B HQQ4/`k4v4` launch config
`/tmp/krasis-tbaflolx.conf`.

| Model / run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Logs |
|-------------|---------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|------|
| Qwen3.5-35B-A3B Zephyrus RTX 3070 Laptop HQQ4 k4v4 | `./dev benchmark /tmp/krasis-tbaflolx.conf` | HQQ4 | k4v4 | 222.1 | 12.48 | 22.00 | 779/10240 (7.6%) | 536 MB | [full log](20260527_zephyrus_q35_hqq4_k4v4_benchmark.log), [report](20260527_zephyrus_q35_hqq4_k4v4_benchmark_report.log), [server log](20260527_zephyrus_q35_hqq4_k4v4_krasis.log) |

Notes:
- Startup released `565 MB` of redundant GPU execution source tensors before
  calibration (`485 MB` lm-head source + `80 MB` router FP32 mirrors).
- HCS loaded no hard experts and `984/10240` soft experts at startup; the
  benchmark ended with `779/10240` experts loaded and a `536 MB` decode
  low-water against the configured `500 MB` safety margin.
- Prefill peaked at the 5K row (`222.1 tok/s`). Longer prompts still fell back
  sharply: `20K 152.3 tok/s`, `35K 74.3 tok/s`, `50K 74.5 tok/s`.

---

## Standard Benchmarks - 2026-05-18 (Qwen3.6 RTX 5090 HQQ/KV sweep)

Hardware: EPYC 7742, 1007 GB RAM, selected physical GPU0 RTX 5090 32 GB.
Timing instrumentation was disabled. All rows used
`./dev benchmark <config>`, Qwen3.6-35B-A3B, INT4 GPU/CPU experts, INT8
shared/dense/lm-head, layer group size 2, 1000 MB KV cache, and the default
600 MB VRAM safety margin.

| Model / run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Logs |
|-------------|---------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|------|
| Qwen3.6-35B-A3B local RTX 5090 HQQ4 k4v4 | `./dev benchmark qwen36-35b-5090-hqq4-k4v4-benchmark.conf` | HQQ4 | k4v4 | 10030.3 | 124.88 | 267.00 | 10240/10240 (100.0%) | 9576 MB | [full log](20260518_qwen36_5090_hqq4_k4v4_benchmark.log), [report](20260518_qwen36_5090_hqq4_k4v4_benchmark_report.log), [server log](20260518_qwen36_5090_hqq4_k4v4_krasis.log) |
| Qwen3.6-35B-A3B local RTX 5090 HQQ4 k6v6 | `./dev benchmark qwen36-35b-5090-hqq4-k6v6-benchmark.conf` | HQQ4 | k6v6 | 10004.2 | 120.71 | 254.31 | 10240/10240 (100.0%) | 9580 MB | [full log](20260518_qwen36_5090_hqq4_k6v6_benchmark.log), [report](20260518_qwen36_5090_hqq4_k6v6_benchmark_report.log), [server log](20260518_qwen36_5090_hqq4_k6v6_krasis.log) |
| Qwen3.6-35B-A3B local RTX 5090 HQQ6 k4v4 | `./dev benchmark qwen36-35b-5090-hqq6-k4v4-benchmark.conf` | HQQ6 | k4v4 | 9532.5 | 116.82 | 237.82 | 10240/10240 (100.0%) | 9172 MB | [full log](20260518_qwen36_5090_hqq6_k4v4_benchmark.log), [report](20260518_qwen36_5090_hqq6_k4v4_benchmark_report.log), [server log](20260518_qwen36_5090_hqq6_k4v4_krasis.log) |
| Qwen3.6-35B-A3B local RTX 5090 HQQ6 k6v6 | `./dev benchmark qwen36-35b-5090-hqq6-k6v6-benchmark.conf` | HQQ6 | k6v6 | 9588.8 | 116.11 | 237.34 | 10240/10240 (100.0%) | 9208 MB | [full log](20260518_qwen36_5090_hqq6_k6v6_benchmark.log), [report](20260518_qwen36_5090_hqq6_k6v6_benchmark_report.log), [server log](20260518_qwen36_5090_hqq6_k6v6_krasis.log) |
| Qwen3.6-35B-A3B local RTX 5090 HQQ8 k4v4 | `./dev benchmark qwen36-35b-5090-hqq8-k4v4-benchmark.conf` | HQQ8 | k4v4 | 8682.7 | 125.70 | 268.55 | 10240/10240 (100.0%) | 8832 MB | [full log](20260518_qwen36_5090_hqq8_k4v4_benchmark.log), [report](20260518_qwen36_5090_hqq8_k4v4_benchmark_report.log), [server log](20260518_qwen36_5090_hqq8_k4v4_krasis.log) |
| Qwen3.6-35B-A3B local RTX 5090 HQQ8 k6v6 | `./dev benchmark qwen36-35b-5090-hqq8-k6v6-benchmark.conf` | HQQ8 | k6v6 | 9300.3 | 125.61 | 268.16 | 10240/10240 (100.0%) | 8868 MB | [full log](20260518_qwen36_5090_hqq8_k6v6_benchmark.log), [report](20260518_qwen36_5090_hqq8_k6v6_benchmark_report.log), [server log](20260518_qwen36_5090_hqq8_k6v6_krasis.log) |

Notes:
- HQQ4 + `k4v4` was the fastest prefill row (`10030.3 tok/s`) and remained
  close to the end-to-end winner.
- HQQ8 + `k4v4` was the fastest internal decode (`125.70 tok/s`) and fastest
  HTTP round trip (`268.55 tok/s`), but by a small margin over HQQ4 + `k4v4`.
  The public README table uses HQQ4 + `k4v4` because it is the only five-figure
  prefill row while remaining effectively tied on decode and HTTP.
- All rows retained the full 10240/10240 HCS pool. Decode minimum free VRAM was
  far above the 600 MB safety margin; calibration long-prefill lows were around
  4.9-5.6 GB free depending on the HQQ cache size.

---

## Standard Benchmarks - 2026-05-18 (Qwen3.6 Ampere validation)

Hardware: EPYC 7742, 1007 GB RAM, selected physical GPU1 RTX A4500 20 GB.
Timing instrumentation was disabled. The benchmark process reported
`GPU 0 (physical 1): NVIDIA RTX A4500`, so process `cuda:0` maps to the
physical A4500.

| Model / run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Logs |
|-------------|---------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|------|
| Qwen3.6-35B-A3B local RTX A4500 HQQ6 k6v6 | server benchmark run from `logs/server-run-benchmark_20260518_001918` | HQQ6 | k6v6 | 2235.2 | 50.98 | 103.98 | 8150/10240 (79.6%) | 720 MB | [report](20260518_qwen36_a4500_hqq6_k6v6_benchmark_report.log), [server log](20260518_qwen36_a4500_hqq6_k6v6_krasis.log) |

Notes:
- The run validates the current Qwen3.6-35B-A3B HQQ6 + `k6v6` path on Ampere
  hardware without requiring FP8 KV cache support.
- Timed prefill peaked at the 35K row (`2235.2 tok/s`); the 50K row completed
  at `2064.8 tok/s`.
- Internal decode was stable across the 50/100/250 token rows
  (`50.98`, `50.88`, `50.58 tok/s`).

---

## Standard Benchmarks - 2026-05-16 (rc30 VRAM pressure guardrails)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the run.
Timing instrumentation was disabled. At the time, the run used the repeatable
`./dev speed-test` entry point and the then-fixed
`tests/qcn-k4v4-hqq8-int4-benchmark.conf` config. The current fixed
speed-test surface is QCN INT4 with HQQ4 attention, `k4v4` KV, thinking off,
and HCS RAM saver off via `tests/qcn-k4v4-hqq4-int4-benchmark.conf`.

| Model / run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Logs |
|-------------|---------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|------|
| Qwen3-Coder-Next rc30 VRAM pressure guardrails | `./dev speed-test` | HQQ8 | k4v4 | 6111.2 | 88.59 | 157.00 | 15633/24576 (63.6%) | 656 MB | [full log](20260516_qcn_k4v4_hqq8_int4_rc30_pressure_speedtest.log), [report](20260516_qcn_k4v4_hqq8_int4_rc30_pressure_speedtest_report.log), [server log](20260516_qcn_k4v4_hqq8_int4_rc30_pressure_speedtest_krasis.log) |

Calibration:
- Short probe: `500` prompt tokens, baseline `25174 MB`, prefill post-alloc
  `24324 MB`, prefill min `23716 MB`, decode min `25124 MB`.
- Long probe: `39920` prompt tokens, baseline `25124 MB`, prefill post-alloc
  `3502 MB`, prefill min `2062 MB`, decode min `25122 MB`.
- Decode HCS budget: `24472 MB`; prefill HCS budgets: short `23064 MB`, long
  `1460 MB`.

Notes:
- The run validated the rc30 runtime pressure-eviction changes on a clean
  no-pressure speed path. No `VRAM MONITOR` warning appeared, so pressure
  eviction did not trigger during the benchmark.
- Decode low-water was `656 MB`, close to but above the configured `600 MB`
  safety margin.
- Startup HCS loaded `15633/24576` soft experts (`63.6%`) for `24182.3 MB`.
- The 50K timed prefill row completed at `5616.3 tok/s`.

---

## Standard Benchmarks - 2026-05-13 (Local RTX A4500)

Hardware: EPYC 7742, 1007 GB RAM, selected physical GPU1 RTX A4500 20 GB
(Ampere, compute capability 8.6, PCIe Gen4 x16 max). Timing instrumentation was
disabled. The benchmark process ran with `CFG_SELECTED_GPUS="1"`, so process
`cuda:0` maps to the physical A4500. The 600 MB row includes the benchmark
report GPU-label fix and reports `GPU 0 (physical 1): NVIDIA RTX A4500`.

| Model / run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Logs |
|-------------|---------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|------|
| Qwen3-Coder-Next local RTX A4500 HQQ6 stage-split HCS safety 600 | `./dev benchmark tests/qcn-a4500-hqq6-benchmark.conf` after decode/prefill HCS budget split | HQQ6 | k4v4 | 1569.5 | 34.69 | 60.47 | 8100/24576 (33.0%) | 664 MB | [full log](20260513_qcn_a4500_hqq6_stage_split_benchmark.log), [report](20260513_qcn_a4500_hqq6_stage_split_benchmark_report.log), [server log](20260513_qcn_a4500_hqq6_stage_split_krasis.log) |
| Qwen3.5-35B-A3B local RTX A4500 HQQ6 stage-split HCS safety 600 | `./dev benchmark tests/q35b-a4500-hqq6-benchmark.conf` after decode/prefill HCS budget split | HQQ6 | fp8 | 2252.7 | 49.98 | 101.84 | 8100/10240 (79.1%) | 702 MB | [full log](20260513_q35_a4500_hqq6_stage_split_benchmark.log), [report](20260513_q35_a4500_hqq6_stage_split_benchmark_report.log), [server log](20260513_q35_a4500_hqq6_stage_split_krasis.log) |
| Qwen3.5-35B-A3B local RTX A4500 HQQ6 safety 600 pre-stage-split | `./dev benchmark tests/q35b-a4500-hqq6-benchmark.conf` | HQQ6 | fp8 | 2264.5 | 49.33 | 95.36 | 7150/10240 (69.8%) | 1996 MB | [full log](20260513_q35_a4500_hqq6_safety600_benchmark.log), [report](20260513_q35_a4500_hqq6_safety600_benchmark_report.log), [server log](20260513_q35_a4500_hqq6_safety600_krasis.log) |
| Qwen3.5-35B-A3B local RTX A4500 HQQ6 safety 3000 | `./dev benchmark tests/q35b-a4500-hqq6-benchmark.conf` at safety `3000` | HQQ6 | fp8 | 2417.7 | 47.07 | 84.25 | 5600/10240 (54.7%) | 4418 MB | [full log](20260513_q35_a4500_hqq6_benchmark.log), [report](20260513_q35_a4500_hqq6_benchmark_report.log), [server log](20260513_q35_a4500_hqq6_krasis.log) |

Qwen3.5 calibration:
- Short probe: `500` prompt tokens, baseline `13510 MB`, prefill post-alloc
  `12604 MB`, prefill min `11900 MB`, decode min `13438 MB`.
- Long probe: `39920` prompt tokens, baseline `13438 MB`, prefill post-alloc
  `3286 MB`, prefill min `2902 MB`, decode min `13428 MB`.

Notes:
- The QCN run used an active HQQ6/k4v4 benchmark profile bound to physical GPU1
  with `CFG_VRAM_SAFETY_MARGIN="600"`. Startup loaded `8250/24576` soft
  experts (`33.6%`) and the benchmark ended at `8100/24576` (`33.0%`) with a
  `664 MB` decode low-water. No `below-vram-safety-limit.log` was produced and
  no `VRAM MONITOR` warning was found in the archived logs.
- QCN calibration: short probe baseline `13670 MB`, prefill post/min
  `12402/11794 MB`, decode min `13620 MB`; long probe baseline `13620 MB`,
  prefill post/min `5274/3834 MB`, decode min `13618 MB`. Stage-split budgets:
  decode HCS `12968 MB`, prefill short/long `11142/3232 MB`.
- Reducing the safety margin from `3000 MB` to the default `600 MB` increased
  resident HCS from `5600/10240` (`54.7%`) to `7150/10240` (`69.8%`) and
  improved internal decode from `47.07` to `49.33 tok/s`. No
  `below-vram-safety-limit.log` was produced and no `VRAM MONITOR` warning was
  found in the archived 600 MB logs.
- Splitting decode-resident HCS from prefill HCS budgets raised resident HCS to
  `8100/10240` (`79.1%`) and lowered decode low-water to `702 MB`, close to the
  configured `600 MB` safety margin. The run produced no
  `below-vram-safety-limit.log` and no `VRAM MONITOR` warning.
- The run completed startup calibration, heatmap, warmup, 1K/5K/10K/20K/35K/50K
  prefill, internal decode, and HTTP round-trip rows with no VRAM hard-floor
  exit.
- The safety 3000 benchmark used the conservative margin from the initial test.
  Long calibration low-water was just under that target (`2902 MB`), but the
  decode benchmark low-water stayed higher at `4418 MB`.
- Timed prefill peaked at the 20K row (`2417.7 tok/s`); the 50K row completed
  at `2208.5 tok/s`.
- With safety 600, timed prefill peaked at the 35K row (`2264.5 tok/s`); the
  50K row completed at `2191.8 tok/s`.

---

## Standard Benchmarks - 2026-05-11 (Typhon resident-HCS and scratch budget rc17)

Hardware: Ryzen 9 5900X, 117 GB RAM, RTX 5080 16 GB under WSL.

These runs validate the resident-HCS, optional prefill pinning, and prefill
scratch budget fixes after public rc16 failed this Typhon profile during
benchmark warmup/timed prefill. Timing instrumentation was disabled for the
clean benchmark runs.

| Model / run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Logs |
|-------------|---------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|------|
| Qwen3.5-35B-A3B Typhon HQQ46 public rc17 | installed `krasis --config /tmp/krasis-aql0_caf.conf --benchmark` from public `v0.1.66-rc17` cp312 wheel | HQQ46_AUTO | k4v4 | 4666.1 | 59.54 | 131.22 | 4510/10240 (44.0%) | 1307 MB | [full log](20260511_typhon_q35_hqq46_rc17_public_benchmark.log), [report](20260511_typhon_q35_hqq46_rc17_public_benchmark_report.log), [server log](20260511_typhon_q35_hqq46_rc17_public_krasis.log) |
| Qwen3.5-35B-A3B Typhon HQQ46 local rc17 candidate | installed local cp312 wheel, `krasis --config /tmp/krasis-aql0_caf.conf --benchmark` | HQQ46_AUTO | k4v4 | 4537.7 | 60.14 | 119.01 | 4510/10240 (44.0%) | 1307 MB | [report](20260511_typhon_q35_hqq46_resident_hcs_budget_report.log), [server log](20260511_typhon_q35_hqq46_resident_hcs_budget_krasis.log) |

Calibration:
- Short probe: `500` prompt tokens, baseline `9531 MB`, prefill post-alloc
  `8943 MB`, prefill min `8591 MB`, decode min `9497 MB`.
- Long probe: `39920` prompt tokens, baseline `9497 MB`, prefill post-alloc
  `1557 MB`, prefill min `1173 MB`, decode min `9495 MB`.
- Resident HCS budget: `7955 MB`, derived from the stricter measured
  request-start floor instead of short decode alone.

Notes:
- The public rc17 benchmark completed warmup, 1K/5K/10K/20K/35K/50K prefill,
  internal decode, and HTTP round-trip rows with no hard-floor exit.
- The local candidate benchmark completed warmup, 1K/5K/10K/20K/35K/50K prefill, internal
  decode, and HTTP round-trip rows with no hard-floor exit.
- The public rc17 50K timed prefill row completed at `4666.1 tok/s`.

---

## Standard Benchmarks - 2026-05-11 (Measured prefill pinning budget rc15)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the run.

This run validates the rc15 prefill-pinning budget change. Startup short/long
VRAM calibration ran with optional prefill pinning disabled, recorded
post-scratch and low-water VRAM, and runtime prefill pinning was capped from
the measured post-scratch transient requirement. Timing instrumentation was
disabled.

| Model / run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Log |
|-------------|---------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|-----|
| Qwen3-Coder-Next measured pinning budget | `./dev speed-test` | HQQ8 | k4v4 | 6007.7 | 89.53 | 220.18 | 15633/24576 (63.6%) | 670 MB | [log](20260511_qcn_rc15_measured_prefill_pinning_speedtest.log) |

Calibration:
- Short probe: `500` prompt tokens, baseline `25188 MB`, prefill post-alloc
  `24338 MB`, prefill min `23730 MB`, decode min `25138 MB`.
- Long probe: `39920` prompt tokens, baseline `25138 MB`, prefill post-alloc
  `3516 MB`, prefill min `2044 MB`, decode min `25136 MB`.

Notes:
- The run completed with `SPEED_RC15_2_EXIT:0`.
- No hard-floor VRAM monitor warning occurred in the archived speed log.
- The 50K timed prefill row completed at `5374.3 tok/s`.

---

## Standard Benchmarks - 2026-05-09 (Peak VRAM safe reductions QCN speed gate)

Hardware: EPYC 7742, 1007 GB RAM, RTX 5090 32 GB selected for the run.

This run validates the safe peak-VRAM reductions after the Qwen3.5 and QCN
llama-witness accuracy gates. The standard repeatable QCN benchmark was used
with timing instrumentation disabled.

| Model / run | Command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Log |
|-------------|---------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|-----|
| Qwen3-Coder-Next peak-VRAM reductions | `./dev speed-test` | HQQ8 | k4v4 | 6324.1 | 88.79 | 146.00 | 15552/24576 (63.3%) | 732 MB | [log](20260509_qcn_peak_vram_safe_reductions_speedtest.log) |

Validation:
- Qwen3.5-35B HQQ4/k4v4 witness comparison passed against
  `llama_witness_qwen35_expanded_thinking_off`: `10 PASS, 0 WARN, 0 FAIL`,
  prefill argmax `10/10`, first-token `10/10`, average decode top-k `100.0%`.
- Qwen3-Coder-Next HQQ8/k4v4 witness comparison passed against
  `llama_witness_stage3_qcn_expanded`: `8 PASS, 0 WARN, 0 FAIL`, prefill
  argmax `8/8`, first-token `8/8`, average decode top-k `100.0%`.

Notes:
- The first standard speed-test attempt exposed an unsafe prefill pinning
  budget: long startup calibration hit the 125 MB hard VRAM floor. Pinning now
  reserves both configured safety and the hard floor before using spare VRAM.
- The debug speed-test also showed 50K timed prefill reserved scratch for
  `40218` tokens while runtime chunking used two `25000` token chunks. Scratch
  allocation is now capped to the clean runtime chunk size; this preserves
  the chunking policy while avoiding unused scratch residency.
- The final standard speed-test completed successfully and kept timed decode
  minimum free VRAM at `732 MB`.

---

## Experimental Runs - 2026-05-04 (Phase 2HA Q122B heatmap prefix + recency tail)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the runs.

These extend the Phase 2GZ heatmap-prefix/recency-tail sweep to Qwen3.5-122B.
Runs used normal global heatmap HCS plus `KRASIS_DYNAMIC_HCS=1
KRASIS_DYNAMIC_HCS_TAIL_BLOCKS=N KRASIS_HCS_COLD_SWAP=0`. Timing
instrumentation was disabled. Decode remained exact.

For Q122B, one activated-expert block is computed at runtime as `48 * 8 = 384`
routed expert invocations per generated token.

| Model / run | Command | Tail slots | Protected slots | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Dynamic result | Log |
|-------------|---------|-----------:|----------------:|----------------:|---------------:|-------------------:|-----|----------------|-----|
| Qwen3.5-122B-A10B tail 1 block | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_DYNAMIC_HCS=1 KRASIS_DYNAMIC_HCS_TAIL_BLOCKS=1 KRASIS_HCS_COLD_SWAP=0 ./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf` | 384 | 3396 | 4071.3 | 23.88 | 43.01 | 3780/12288 (30.8%) | Main final hit `66.44%`, final cumulative hit `66.12%`, `copy_failures=0` | [log](20260504_phase2ha_q122b_tail1.log) |
| Qwen3.5-122B-A10B tail 2 blocks | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_DYNAMIC_HCS=1 KRASIS_DYNAMIC_HCS_TAIL_BLOCKS=2 KRASIS_HCS_COLD_SWAP=0 ./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf` | 768 | 3012 | 4964.7 | 24.26 | 46.13 | 3780/12288 (30.8%) | Main final hit `68.33%`, final cumulative hit `68.06%`, `copy_failures=0` | [log](20260504_phase2ha_q122b_tail2.log) |

Notes:
- No CUDA/runtime errors were found in the two Phase 2HA logs.
- Two blocks improved Q122B modestly over one block (`23.88 -> 24.26 tok/s`)
  and increased main final hit (`66.44% -> 68.33%`).
- Q122B still does not clearly beat the stronger prior static prompt-HCS result
  (`25.29 tok/s` from Phase 2GO), so unlike Q235 this is not enough evidence to
  make recency-tail mode the Q122B default.

---

## Experimental Runs - 2026-05-04 (Phase 2GZ heatmap prefix + recency tail)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the runs.

These are opt-in runtime-policy experiments, not new defaults. Runs used normal
global heatmap HCS plus `KRASIS_DYNAMIC_HCS=1 KRASIS_DYNAMIC_HCS_TAIL_BLOCKS=N
KRASIS_HCS_COLD_SWAP=0`. Timing instrumentation was disabled. Decode remained
exact: cold experts were still computed exactly, and dynamic HCS only changed
soft-tier residency.

The recency tail is measured in runtime activated-expert blocks. One block is
computed from the model/layers as routed expert invocations per generated token:
QCN `48 * 10 = 480`; Q235 `94 * 8 = 752`. The high-ranked heatmap prefix is
protected; only the low-ranked tail can be replaced by recency promotions.

| Model / run | Command | Tail slots | Protected slots | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Dynamic result | Log |
|-------------|---------|-----------:|----------------:|----------------:|---------------:|-------------------:|-----|----------------|-----|
| Qwen3-Coder-Next tail 1 block | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_DYNAMIC_HCS=1 KRASIS_DYNAMIC_HCS_TAIL_BLOCKS=1 KRASIS_HCS_COLD_SWAP=0 ./dev speed-test` | 480 | 14667 | 7856.8 | 87.95 | 140.43 | 15147/24576 (61.6%) | Final cumulative hit `95.20%`, `copy_failures=0`; small tail was safe but not best | [log](20260504_phase2gz_qcn_tail1.log) |
| Qwen3-Coder-Next tail 2 blocks | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_DYNAMIC_HCS=1 KRASIS_DYNAMIC_HCS_TAIL_BLOCKS=2 KRASIS_HCS_COLD_SWAP=0 ./dev speed-test` | 960 | 14187 | 7287.4 | 90.53 | 151.01 | 15147/24576 (61.6%) | Best QCN row in this sweep; final cumulative hit `96.58%`, `copy_failures=0` | [log](20260504_phase2gz_qcn_tail2.log) |
| Qwen3-235B-A22B tail 1 block | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_DYNAMIC_HCS=1 KRASIS_DYNAMIC_HCS_TAIL_BLOCKS=1 KRASIS_HCS_COLD_SWAP=0 ./dev benchmark tests/q235-k4v4-hqq6-int4-benchmark.conf` | 752 | 834 | 1943.2 | 4.69 | 8.33 | 1586/12032 (13.2%) | Stable improvement over static HCS, but below two-block tail; final cumulative hit `36.70%`, `copy_failures=0` | [log](20260504_phase2gz_q235_tail1.log) |
| Qwen3-235B-A22B tail 2 blocks | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_DYNAMIC_HCS=1 KRASIS_DYNAMIC_HCS_TAIL_BLOCKS=2 KRASIS_HCS_COLD_SWAP=0 ./dev benchmark tests/q235-k4v4-hqq6-int4-benchmark.conf` | 1504 | 82 | 1758.1 | 4.86 | 7.94 | 1586/12032 (13.2%) | Best Q235 row in this sweep; final cumulative hit `40.87%`, `copy_failures=0` | [log](20260504_phase2gz_q235_tail2.log) |

Notes:
- No CUDA/runtime errors were found in the four Phase 2GZ logs.
- QCN benefits from keeping most of the global heatmap cache; two recency
  blocks (`960` slots, only `6.3%` of loaded HCS) performed best in this sweep.
- Q235 still wants recency dominance: two blocks leave only `82` protected
  heatmap slots and recover the previous budgeted dynamic best without using
  heuristic startup. This suggests low-coverage models should prioritize
  roughly 1-2 activated-expert blocks of recency tail over static heatmap tail.
- These runs still pay global heatmap startup cost. The earlier heuristic
  dynamic mode avoids that startup cost, but for QCN/Q122B it discarded too much
  useful static heatmap state.

---

## Experimental Runs - 2026-05-04 (Phase 2GY dynamic recency HCS)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the runs.

These are opt-in runtime-policy experiments, not new defaults. Runs used
`KRASIS_DYNAMIC_HCS=1 KRASIS_HCS_HEURISTIC_INIT=1 KRASIS_HCS_COLD_SWAP=0`.
Timing instrumentation was disabled. Decode remained exact: cold experts were
still computed exactly, and dynamic HCS only changed soft-tier residency.

The first unbounded Q235 attempt reached higher hit rate but regressed to
`2.7-2.8 tok/s` because it promoted almost every cold expert. The completed
rows below use the budgeted policy: at most one loaded soft-cache worth of
promotions per decode request by default, controlled by
`KRASIS_DYNAMIC_HCS_PROMOTION_BUDGET_MULT`.

| Model / run | Config / command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Dynamic result | Log |
|-------------|------------------|-----------|----|----------------:|---------------:|-------------------:|-----|----------------|-----|
| Qwen3-Coder-Next budgeted dynamic HCS | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_DYNAMIC_HCS=1 KRASIS_HCS_HEURISTIC_INIT=1 KRASIS_HCS_COLD_SWAP=0 ./dev speed-test` | HQQ8 | k4v4 | 7554.9 | 78.04 | 148.30 | 15147/24576 (61.6%) | Budget did not bind; final cumulative hit `97.14%`, `copy_failures=0` | [log](20260504_phase2gy_qcn_dynamic_hcs_budgeted.log) |
| Qwen3.5-122B-A10B budgeted dynamic HCS | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_DYNAMIC_HCS=1 KRASIS_HCS_HEURISTIC_INIT=1 KRASIS_HCS_COLD_SWAP=0 ./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf` | HQQ6 | k4v4 | 4159.0 | 23.30 | 32.45 | 3780/12288 (30.8%) | Final internal hit `72.65%`, but slower than static HCS; not useful as a blanket default | [log](20260504_phase2gy_q122b_dynamic_hcs_budgeted.log) |
| Qwen3-235B-A22B budgeted dynamic HCS | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_DYNAMIC_HCS=1 KRASIS_HCS_HEURISTIC_INIT=1 KRASIS_HCS_COLD_SWAP=0 ./dev benchmark tests/q235-k4v4-hqq6-int4-benchmark.conf` | HQQ6 | k4v4 | 2018.9 | 4.87 | 8.01 | 1586/12032 (13.2%) | Improved Q235 exact decode `3.54 -> 4.87 tok/s`; final internal hit `40.07%`, `copy_failures=0` | [log](20260504_phase2gy_q235_dynamic_hcs_budgeted.log) |

Notes:
- Dynamic recency HCS helps the low-coverage Q235 case, but the budgeted
  policy is still conservative. The `budget_skips` counters show many skipped
  promotions on Q235, so there is room to tune admission without returning to
  unbounded churn.
- Q122B shows the opposite tradeoff: dynamic hit rate is high, but promotion
  overhead and heuristic startup make it slower than the previous exact static
  HCS row. Dynamic HCS should be gated by model/coverage/performance data.
- Heuristic startup avoids global heatmap build cost and is explicitly rejected
  unless `KRASIS_DYNAMIC_HCS=1` is also set.

---

## Diagnostic Runs - 2026-05-04 (Phase 2GX cache strategy shadow simulator)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the runs.

These are instrumentation runs, not speed baselines. `KRASIS_ROUTE_LOCALITY=1`
was enabled to shadow-simulate cache policies during exact decode. Runtime HCS
behavior and expert selection were unchanged; the policy columns are simulated
hit rates over the main internal 49/99/249 decode rows.

| Model | Command | Runtime HCS | Actual HCS hit | Heatmap-only 15% | LRU 15% | Heatmap+LRU 15% | Decayed LFU 15% | Adaptive 15% | Log |
|-------|---------|------------:|---------------:|-----------------:|--------:|----------------:|----------------:|-------------:|-----|
| Qwen3-Coder-Next | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_ROUTE_LOCALITY=1 KRASIS_HCS_COLD_SWAP=0 ./dev speed-test` | 15147/24576 (61.6%) | 95.87% | 46.39% | 79.10% | 74.89% | 79.21% | 79.51% | [log](20260504_phase2gx_qcn_strategy_sim.log) |
| Qwen3.5-122B-A10B | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_ROUTE_LOCALITY=1 KRASIS_HCS_COLD_SWAP=0 ./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf` | 3780/12288 (30.8%) | 67.13% | 42.36% | 61.01% | 60.03% | 62.53% | 62.49% | [log](20260504_phase2gx_q122b_strategy_sim.log) |
| Qwen3-235B-A22B | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_ROUTE_LOCALITY=1 KRASIS_HCS_COLD_SWAP=0 ./dev benchmark tests/q235-k4v4-hqq6-int4-benchmark.conf` | 1586/12032 (13.2%) | 37.62% | 40.21% | 64.95% | 58.81% | 64.61% | 64.72% | [log](20260504_phase2gx_q235_strategy_sim.log) |

Notes:
- The first Phase 2GW simulator archive was removed because a pre-report code
  review found that fallback heatmap ranks were not stored in the per-layer
  rank cache. Phase 2GX is the corrected rerun set.
- Q235 shows the largest opportunity: at 15% simulated capacity, simple
  recency/frequency policies hit about `64-65%` versus current runtime HCS
  `37.62%`. With Q235's measured `~9.28 MB` INT4 expert payload, that would
  reduce cold expert traffic from roughly `4.35 GB/tok` to `2.45-2.46 GB/tok`
  if the simulated hits can be made resident or prefetched in the real runtime.
- QCN's actual HCS hit rate is much higher because the real runtime cache has
  `61.6%` coverage. The simulated 15% policies are useful for shape comparison,
  not as a claim that QCN should reduce residency.

---

## Standard Benchmarks - 2026-05-03 (Phase 2GT Q235B llama-witness gate and benchmark)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Q235B was benchmarked only after building a BF16 llama-witness GGUF and
capturing a non-HF witness artifact. The exact/default prompt-HCS path was
used; HCS cold swaps were off and timing instrumentation was disabled.

| Model / run | Config / command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Log |
|-------------|------------------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|-----|
| Qwen3-235B-A22B prompt-HCS default | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 ./dev benchmark tests/q235-k4v4-hqq6-int4-benchmark.conf` | HQQ6 | k4v4 | 1459.1 | 3.54 | 6.17 | 1586/12032 (13.2%) | 1062 MB | [log](20260503_phase2gt_q235_k4v4_hqq6_prompt_hcs_benchmark.log) |

Validation:
- Deleted the stale Q235 Q4_K_M GGUF shards from
  `~/.krasis/models/Qwen3-235B-A22B-GGUF/Q4_K_M/Q4_K_M/`.
- Added Q235 support to the built `./dev witness-*` wrappers.
- `./dev witness-gguf-preflight q235` passed, then
  `./dev witness-gguf-convert q235` produced
  `~/.krasis/witness-gguf/Qwen3-235B-A22B-bf16.gguf`
  (`470,293,436,256` bytes).
- `./dev witness-capture q235` produced
  `krasis-internal/reference-outputs/output/Qwen3-235B-A22B/llama_witness_q235_thinking_off.json`
  with `14` llama-witness first-token cases.
- Q235B reference gate against the llama-witness artifact passed:
  prompts `14/14`, first token `13/14`, generated containment `14/14`,
  prefill top-10 containment `14/14`.

Notes:
- The benchmark capped timed prefill at the config context limit, so the
  best prefill row is `17,324` tokens rather than the nominal 20K/35K/50K
  prompt sizes.
- HCS coverage is low (`13.2%` during timed decode), so decode remains heavily
  cold-expert limited.

---

## Standard Benchmarks - 2026-05-03 (Phase 2GR QCN HCS cold swaps)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

QCN was run with opt-in approximate HCS cold swaps enabled. Timing
instrumentation was disabled. The exact default remains unchanged.

| Model / run | Config / command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Log |
|-------------|------------------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|-----|
| Qwen3-Coder-Next HCS cold swaps | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_HCS_COLD_SWAP=1 ./dev speed-test` | HQQ8 | k4v4 | 8488.2 | 84.45 | 192.63 | 15147/24576 (61.6%) | 706 MB | [log](20260503_phase2gr_qcn_hcs_cold_swap_speedtest.log) |

Notes:
- This was worse than the recent exact prompt-HCS QCN rows:
  Phase 2GO 85% default `89.37 tok/s`, Phase 2GP 90% retain `90.78 tok/s`.
- Official internal rows applied only `1604` swaps over `397` decoded tokens
  (`4.04/tok`), much less than Q122B's `19.11/tok`.
- Weighted cold after swaps was `16.48/tok`. The QCN cold-miss surface is
  already small enough that this approximate mode does not help.

---

## Standard Benchmarks - 2026-05-03 (Phase 2GQ opt-in HCS cold swaps)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

HCS cold swaps are an approximate decode mode and are disabled by default.
This run used `KRASIS_HCS_COLD_SWAP=1` with the default policy: protect the
top `75%` selected expert ranks, and replace only lower-rank cold selected
experts with same-layer HCS-resident experts whose router score is within
`max(0.005, 10%)`. Routing weights are preserved, but selected expert identity
changes for swapped slots.

Timing instrumentation was disabled for the benchmark row.

| Model / run | Config / command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Log |
|-------------|------------------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|-----|
| Qwen3.5-122B-A10B HCS cold swaps | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_HCS_COLD_SWAP=1 ./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf` | HQQ6 | k4v4 | 4280.2 | 27.30 | 50.01 | 3780/12288 (30.8%) | 662 MB | [log](20260503_phase2gq_q122b_k4v4_hqq6_hcs_cold_swap_benchmark.log) |

Validation:
- Q122B HQQ6+k4v4 seq32 witness with swaps enabled passed:
  first token `14/14`, prefill `14/14`, exact `261/361`, containment
  `279/361`, full exact `8/14`.
- This is weaker than the exact prompt-HCS default witness row
  (`280/361` exact, `303/361` containment), so HCS cold swaps remain an
  opt-in approximate experiment, not a default.

Notes:
- Compared with Phase 2GO exact prompt-HCS default, Q122B internal decode
  improved `25.29 -> 27.30 tok/s`, but generated-token agreement dropped.
- Official internal benchmark swap summaries reported `7586` swaps over
  `397` decoded tokens (`19.11/tok`), weighted cold after swaps
  `109.03/tok`, and weighted router score delta `0.004325` absolute
  (`5.30%` relative).
- Prefill changed `4880.4 -> 4280.2 tok/s` in this run. This mode only changes
  decode routing after prefill, so treat the prefill movement as run variance
  unless reproduced.

---

## Standard Benchmarks - 2026-05-03 (Phase 2GP 35B control and QCN retain sweep)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the runs.

Prompt-conditioned HCS reload was enabled. Timing instrumentation was disabled
for all speed rows below.

35B was run once to confirm the prompt-HCS default basically does not affect a
full-HCS model. QCN was run with explicit retain percentages using the standard
fixed `./dev speed-test` surface.

| Model / run | Config / command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Log |
|-------------|------------------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|-----|
| Qwen3.5-35B-A3B prompt-HCS default | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 ./dev benchmark tests/q35b-4-4-hqq6-benchmark.conf` | HQQ6 | fp8 | 12127.0 | 113.70 | 230.83 | 10240/10240 (100.0%) | 8990 MB | [log](20260503_phase2gp_q35b_hqq6_prompt_hcs_default_benchmark.log) |
| Qwen3-Coder-Next retain 75% | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_PROMPT_HCS_RETAIN_PCT=75 ./dev speed-test` | HQQ8 | k4v4 | 7875.8 | 87.81 | 143.15 | 15147/24576 (61.6%) | 706 MB | [log](20260503_phase2gp_qcn_hqq8_k4v4_retain75_speedtest.log) |
| Qwen3-Coder-Next retain 80% | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_PROMPT_HCS_RETAIN_PCT=80 ./dev speed-test` | HQQ8 | k4v4 | 7910.7 | 88.07 | 142.08 | 15147/24576 (61.6%) | 706 MB | [log](20260503_phase2gp_qcn_hqq8_k4v4_retain80_speedtest.log) |
| Qwen3-Coder-Next retain 85% | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_PROMPT_HCS_RETAIN_PCT=85 ./dev speed-test` | HQQ8 | k4v4 | 8178.1 | 87.50 | 153.63 | 15147/24576 (61.6%) | 706 MB | [log](20260503_phase2gp_qcn_hqq8_k4v4_retain85_speedtest.log) |
| Qwen3-Coder-Next retain 90% | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_PROMPT_HCS_RETAIN_PCT=90 ./dev speed-test` | HQQ8 | k4v4 | 7993.1 | 90.78 | 136.69 | 15147/24576 (61.6%) | 706 MB | [log](20260503_phase2gp_qcn_hqq8_k4v4_retain90_speedtest.log) |

Notes:
- Qwen3.5-35B decode was effectively unchanged versus the Phase 2GI row
  (`113.32 -> 113.70 tok/s`), as expected with full HCS coverage.
- All QCN retain settings remained above the Phase 2GI pre-prompt-HCS decode
  baseline (`81.02 tok/s`).
- In this pass, `90%` was the best QCN internal decode row at
  `90.78 tok/s`. The earlier Phase 2GO 85% QCN run reported `89.37 tok/s`,
  so a default change from `85%` to `90%` should use another paired A/B if we
  want to separate policy from normal run variance.

---

## Standard Benchmarks - 2026-05-03 (Phase 2GO prompt-conditioned HCS reload default)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the runs.

Prompt-conditioned HCS reload is enabled by default. It preserves the hard tier,
keeps the top `85%` of the global heatmap soft ranking, and fills the lower
soft tail from exact prefill expert counts for the current request. Routing,
expert execution, weights, and outputs remain exact; only soft-tier residency
order changes. `KRASIS_PROMPT_HCS_RELOAD=0` disables it, and
`KRASIS_PROMPT_HCS_RETAIN_PCT` can override the retain percentage.

Timing instrumentation was disabled for all speed rows below.

| Model / run | Config / command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Log |
|-------------|------------------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|-----|
| Qwen3.5-122B-A10B prompt-HCS default | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 ./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf` | HQQ6 | k4v4 | 4880.4 | 25.29 | 44.95 | 3780/12288 (30.8%) | 662 MB | [log](20260503_phase2go_q122b_k4v4_hqq6_prompt_hcs_reload_benchmark.log) |
| Qwen3.5-122B-A10B same-build reload-off control | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_PROMPT_HCS_RELOAD=0 ./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf` | HQQ6 | k4v4 | 4060.3 | 24.30 | 43.63 | 3780/12288 (30.8%) | 662 MB | [log](20260503_phase2go_q122b_k4v4_hqq6_prompt_hcs_reload_off_control_benchmark.log) |
| Qwen3-Coder-Next prompt-HCS default | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 ./dev speed-test` | HQQ8 | k4v4 | 8231.6 | 89.37 | 145.21 | 15147/24576 (61.6%) | 706 MB | [log](20260503_phase2go_qcn_hqq8_k4v4_prompt_hcs_speedtest.log) |

Validation:
- Q122B HQQ6+k4v4 seq32 witness passed before speed testing:
  first token `14/14`, prefill `14/14`, exact `280/361`, containment
  `303/361`, full exact `7/14`.
- A marker-enabled run verified the path was active:
  `retain_pct=85`, `effective_heatmap_slots=3456`, and roughly
  `588-594` soft tail slots repacked for short benchmark requests. The final
  production benchmark gated those stderr markers behind
  `KRASIS_PROMPT_HCS_LOG=1`.

Notes:
- Compared with Phase 2GH Q122B baseline, prompt-HCS default improved prefill
  `4689.8 -> 4880.4 tok/s` and internal decode `24.80 -> 25.29 tok/s`, with
  min free VRAM unchanged at `662 MB`.
- The same-build reload-off control was slower than the production-default
  run, confirming the default behavior is not just benchmark variance in this
  pair.
- QCN speed-test decode improved versus the prior Phase 2GI QCN row
  (`81.02 -> 89.37 tok/s`); prefill remained in the same band
  (`8352.2 -> 8231.6 tok/s`).

---

## Standard Benchmarks - 2026-05-03 (Phase 2GJ metadata-only GPU route sync experiment)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3.5-122B-A10B
`tests/q122b-k4v4-hqq6-int4-benchmark.conf`, INT4 GPU/CPU experts, HQQ6
attention, `k4v4` KV cache, INT8 shared/dense/lm-head, layer group size 2,
graph replay enabled, timing instrumentation off.

This was an opt-in route-sync experiment using
`KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_GPU_ROUTE_SYNC=1`. Mapped
cold-weight reads remained disabled (`mapped_reads=false`); cold experts still
used CPU-initiated DMA into VRAM.

| Variant | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Log |
|--------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|-----|
| Q122B materialized HQQ prefill + metadata-only GPU route sync | HQQ6 | k4v4 | 4094.5 | 23.87 | 42.69 | 3780/12288 (30.8%) | 1846 MB | [log](20260503_phase2gj_q122b_k4v4_hqq6_gpu_route_sync_benchmark.log) |

Notes:
- Q122B HQQ6+k4v4 seq32 witness passed before speed testing:
  first token `14/14`, prefill `14/14`, exact `283/361`, containment
  `304/361`.
- The route-sync path initially loaded `4050/12288 (33.0%)` soft experts, but
  the benchmark settled at `3780/12288 (30.8%)` after prefill/decode
  transitions, matching the current best baseline coverage.
- Compared with Phase 2GH pointer-table prefill prefetch, decode regressed
  `24.80 -> 23.87 tok/s` and prefill regressed `4689.8 -> 4094.5 tok/s`.
- Conclusion: metadata-only GPU route sync is stable as an opt-in diagnostic
  path, but it is not a speed win and should remain disabled by default.

---

## Standard Benchmarks - 2026-05-03 (Phase 2GI QCN/35B pointer-prefetch regression)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for both runs.

These runs validate the current materialized-HQQ-prefill and pointer-table
prefetch code on QCN and Qwen3.5-35B after the Q122B Phase 2GH speed win.
Timing instrumentation was disabled for both speed runs.

| Model | Config / command | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Log |
|-------|------------------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|-----|
| Qwen3-Coder-Next | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 ./dev speed-test` | HQQ8 | k4v4 | 8352.2 | 81.02 | 165.05 | 15147/24576 (61.6%) | 706 MB | [log](20260503_phase2gi_qcn_hqq8_k4v4_speedtest.log) |
| Qwen3.5-35B-A3B | `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 ./dev benchmark tests/q35b-4-4-hqq6-benchmark.conf` | HQQ6 | fp8 | 11074.0 | 113.32 | 230.62 | 10240/10240 (100.0%) | 8990 MB | [log](20260503_phase2gi_q35b_hqq6_fp8_benchmark.log) |

Accuracy gates before speed:
- QCN `tests/qcn-k6v6-hqq8-accuracy.conf` with
  `llama_witness_stage3_qcn_expanded`: PASS, first token `8/8`, prefill
  `8/8`, exact generated prefix `8/8`, containment `8/8`.
- Qwen3.5-35B `tests/q35b-4-4-hqq6-benchmark.conf` with
  `llama_witness_qwen35_expanded_thinking_off`: PASS, first token `10/10`,
  prefill `10/10`, exact generated prefix `10/10`, containment `10/10`.

Notes:
- QCN accuracy used the compact `k6v6` HQQ8 witness config; QCN speed used the
  standard fixed `./dev speed-test` surface, which is HQQ8/k4v4.
- 35B HCS fit all experts in soft residency (`10240/10240`), so decode is a
  useful fully cached contrast against Q122B's hybrid decode path.

---

## Standard Benchmarks - 2026-05-03 (Phase 2GH pointer-table prefill prefetch)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3.5-122B-A10B
`tests/q122b-k4v4-hqq6-int4-benchmark.conf`, INT4 GPU/CPU experts, HQQ6
attention, `k4v4` KV cache, INT8 shared/dense/lm-head, layer group size 2,
graph replay enabled, timing instrumentation off.

| Variant | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Log |
|--------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|-----|
| Q122B materialized HQQ prefill + dense pointer-table prefetch | HQQ6 | k4v4 | 4689.8 | 24.80 | 42.45 | 3780/12288 (30.8%) | 662 MB | [log](20260503_phase2gh_q122b_k4v4_hqq6_ptrprefetch_benchmark.log) |

Notes:
- This run used `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1`.
- Dense pointer-table prefetch stages all non-HCS/non-pinned experts for dense
  MoE chunks before layer attention, then MoE consumes the prefetched pointer
  table. Sparse chunks keep the exact current-layer pointer-table path.
- No second cold staging buffer and no persistent expert/BF16 residency were
  added; the path reuses the existing cold staging allocation and frees raw
  pointer-table buffers after each layer.
- Q122B HQQ6+k4v4 seq32 witness passed before speed testing:
  first token `14/14`, prefill `14/14`, exact `270/361`, containment
  `292/361`.
- Timing-enabled diagnostics confirmed all dense 35K/50K chunks used
  `[PTR-PREFETCH]`, while sparse short heatmap/decode prompts used exact
  `[PTR-TABLE]`.
- Compared with Phase 2GE materialized HQQ prefill, prefill improved
  `3003.9 -> 4689.8 tok/s`; internal decode and HCS stayed effectively flat.

---

## Standard Benchmarks - 2026-05-03 (Phase 2GE materialized HQQ prefill)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3.5-122B-A10B
`tests/q122b-k4v4-hqq6-int4-benchmark.conf`, INT4 GPU/CPU experts, HQQ6
attention, `k4v4` KV cache, INT8 shared/dense/lm-head, layer group size 2,
graph replay enabled, timing instrumentation off.

| Variant | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Log |
|--------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|-----|
| Q122B materialized HQQ prefill, HQQ6/k4v4, INT4 experts | HQQ6 | k4v4 | 3003.9 | 24.28 | 42.29 | 3780/12288 (30.8%) | 662 MB | [log](20260503_phase2ge_q122b_k4v4_hqq6_materialized_prefill_benchmark.log) |

Notes:
- This run used `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1`, an opt-in
  prefill-only HQQ path. Decode remains compact HQQ/VMM.
- The HQQ prefill runtime was forced to row-major compact HQQ and each
  projection was dequantized into a reusable transient BF16 scratch buffer
  before cuBLAS BF16 GEMM. No persistent BF16 attention residency was added.
- Q122B HQQ6+k4v4 seq32 witness was run before speed testing and passed:
  first token `14/14`, prefill `14/14`, exact `280/361`, containment
  `303/361`.
- Bounded 10K component timing improved from Phase 2FS `4143.2 ms` to
  `3600.9 ms`; the old `marlin_float_zp` HQQ projection counter no longer
  appears in the materialized path.
- Compared with Phase 2GC stable HQQ VMM graphs, prefill improved
  `2029.1 -> 3003.9 tok/s`; internal decode was effectively flat
  `24.49 -> 24.28 tok/s`.
- VRAM remained tight but inside the configured safety floor: min decode free
  was `662 MB`.

---

## Standard Benchmarks - 2026-05-03 (Phase 2GC stable HQQ CUDA graph addresses)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3.5-122B-A10B
`tests/q122b-k4v4-hqq6-int4-benchmark.conf`, INT4 GPU/CPU experts, HQQ6
attention, `k4v4` KV cache, INT8 shared/dense/lm-head, layer group size 2,
graph replay enabled, timing instrumentation off.

| Variant | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Log |
|--------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|-----|
| Q122B stable HQQ VMM graphs, HQQ6/k4v4, INT4 experts | HQQ6 | k4v4 | 2029.1 | 24.49 | 44.35 | 3780/12288 (30.8%) | 664 MB | [log](20260503_phase2gc_q122b_k4v4_hqq6_stable_graph_benchmark.log) |

Notes:
- This run follows Phase 2GC stable HQQ graph-address work. HQQ runtime slots
  use stable CUDA VMM virtual addresses and active-stage physical remapping;
  no BF16 attention fallback or duplicate HQQ residency was added.
- Accuracy/regression gates were run before speed testing:
  QCN HQQ8/BF16-KV, QCN HQQ8+k6v6, Q122B HQQ6+k4v4, and a non-HQQ
  QCN BF16-attention/BF16-KV control all passed.
- Graph trace on QCN HQQ8/BF16-KV showed one capture, 15 pointer-check reuses,
  and zero graph invalidations / HQQ decode pointer changes.
- Compared with Phase 2GB on the same config, internal decode improved
  `22.54 -> 24.49 tok/s`, round trip improved `41.47 -> 44.35 tok/s`, and HCS
  coverage improved `3483 -> 3780` experts. Prefill was essentially flat,
  `2071.5 -> 2029.1 tok/s`.

---

## Standard Benchmarks - 2026-05-02 (Phase 2GB HQQ graph GQA fix 122B)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3.5-122B-A10B
`tests/q122b-k4v4-hqq6-int4-benchmark.conf`, INT4 GPU/CPU experts, HQQ6
attention, `k4v4` KV cache, INT8 shared/dense/lm-head, layer group size 2,
graph replay enabled, timing instrumentation off.

| Variant | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Log |
|--------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|-----|
| Q122B fixed HQQ graph GQA, HQQ6/k4v4, INT4 experts | HQQ6 | k4v4 | 2071.5 | 22.54 | 41.47 | 3483/12288 (28.3%) | 640 MB | [log](20260502_phase2gb_q122b_k4v4_hqq6_graph_gqa_fix_benchmark.log) |

Notes:
- This run follows the Phase 2GB HQQ graph GQA decode fix. HQQ remains the
  only resident attention weight path; BF16 attention projection tensors were
  not restored.
- Startup rebuilt the benchmark heatmap with exact benchmark decode params:
  `temperature=0.0`, `top_k=50`, `top_p=0.95`, `enable_thinking=false`.
- Compared with Phase 2FU exact-heatmap on the same config, prefill is
  effectively unchanged (`2030.4 -> 2071.5 tok/s`), while internal decode
  recovered from `8.85` to `22.54 tok/s`.
- Min free decode VRAM remained at the configured safety floor class
  (`640 MB`), so the run completed without OOM but the 2 GB k4v4 config is
  still tight.

---

## Standard Benchmarks - 2026-05-02 (Phase 2FU exact heatmap 122B)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3.5-122B-A10B
`tests/q122b-k4v4-hqq6-int4-benchmark.conf`, INT4 GPU/CPU experts, HQQ6
attention, `k4v4` KV cache, INT8 shared/dense/lm-head, layer group size 2,
graph replay enabled, timing instrumentation off.

| Variant | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Log |
|--------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|-----|
| Q122B exact-heatmap HQQ6/k4v4, INT4 experts | HQQ6 | k4v4 | 2030.4 | 8.85 | 14.36 | 3483/12288 (28.3%) | 640 MB | [log](20260502_phase2fu_q122b_k4v4_hqq6_exact_heatmap_benchmark.log) |

Notes:
- Normal startup rebuilt `auto_heatmap.json`; no explicit `--heatmap-path`
  was used.
- Heatmap generation used exact benchmark decode params:
  `temperature=0.0`, `top_k=50`, `top_p=0.95`, `enable_thinking=false`,
  `mode=benchmark`.
- Heatmap prompts were the held-out `heatmap_prompts.txt` set, not benchmark
  `decode_prompt_*`.
- HCS capacity remained essentially unchanged from Phase 2FR, and decode did
  not improve. This falsifies the narrow hypothesis that the 122B decode
  regression was mainly caused by sampled-vs-greedy heatmap parameter mismatch.
- Timed prefill emitted a VRAM monitor low of `218 MB`, below the configured
  `600 MB` safety margin, but the benchmark completed.

---

## Standard Benchmarks - 2026-05-02 (Phase 2FR HQQ exclusive residency 122B)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3.5-122B-A10B
`tests/q122b-k4v4-hqq6-int4-benchmark.conf`, INT4 GPU/CPU experts, HQQ6
attention, `k4v4` KV cache, INT8 shared/dense/lm-head, layer group size 2,
graph replay enabled, timing instrumentation off.

| Variant | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Log |
|--------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|-----|
| Q122B exclusive HQQ6/k4v4, INT4 experts | HQQ6 | k4v4 | 2070.2 | 9.03 | 14.62 | 3456/12288 (28.1%) | 672 MB | [log](20260502_phase2fr_q122b_k4v4_hqq6_exclusive_residency_benchmark.log) |

Notes:
- HQQ runtime residency is now exclusive for this surface: the run has one
  `HQQ runtime staging prepared` line at `device_mb=3696.47`.
- The HQQ path released `108` replaced BF16 attention projection tensors,
  returning `6075.00 MB` of CUDA tensor residency before calibration.
- Reclaimable HCS budget improved to `19330 MB`; startup HCS reached
  `3888/12288 (31.6%)`, above the previous BF16/fp8 control startup coverage
  of `3537/12288 (28.8%)`.
- Prefill improved from Phase 2FQ `1377.3 tok/s` to `2070.2 tok/s`, and from
  the earlier broken Phase 2FM HQQ6/k4v4 row of `541.9 tok/s`.
- Decode regressed versus Phase 2FQ (`16.41 -> 9.03 tok/s`) despite restored
  HCS coverage. This points to a decode-path issue, not insufficient HCS
  capacity, and needs a separate timing-enabled diagnosis.
- VRAM remains tight on the 2 GB k4v4 config: timed prefill emitted a VRAM
  monitor low of `518 MB`, below the configured `600 MB` safety margin.

---

## Standard Benchmarks — 2026-05-02 (Phase 2FQ stage-exact HQQ/KV 122B)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3.5-122B-A10B
`tests/q122b-k4v4-hqq6-int4-benchmark.conf`, INT4 GPU/CPU experts, HQQ6
attention, `k4v4` KV cache, INT8 shared/dense/lm-head, layer group size 2,
graph replay enabled, timing instrumentation off.

| Variant | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Log |
|--------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|-----|
| Q122B stage-exact HQQ6/k4v4, INT4 experts | HQQ6 | k4v4 | 1377.3 | 16.41 | 29.46 | 1755/12288 (14.3%) | 716 MB | [log](20260502_phase2fq_q122b_k4v4_hqq6_stage_exact_chunkfix_benchmark.log) |

Notes:
- Stage-exact HQQ compaction was active: 122B HQQ6 runtime registered
  `device_mb=3696.47` with `prefill_mb=4904.93` and `decode_mb=3696.47`.
- Stage-exact k4v4 prefill used temporary FP8 KV and bulk-exported to compact
  decode KV. The invalid post-HCS measured chunk cap was removed after
  instrumentation showed it forced the first 25K prefill warmup into
  `196` chunks of `128` tokens.
- Compared with Phase 2FM on the same HQQ6/k4v4 benchmark config, best prefill
  improved from `541.9` to `1377.3 tok/s`; decode moved from `17.66` to
  `16.41 tok/s`.
- VRAM remains tight on the 2 GB k4v4 config: warmup emitted VRAM monitor lows
  of `530 MB` and `512 MB`, below the configured `600 MB` safety margin.
- This is still below the old BF16-attention/FP8-KV 122B control surface
  (`2765.6 tok/s` prefill, `23.20 tok/s` decode).

---

## Standard Benchmarks — 2026-05-02 (Phase 2FM 122B HQQ6/k4v4 refresh)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3.5-122B-A10B
`tests/q122b-k4v4-hqq6-int4-benchmark.conf`, INT4 GPU/CPU experts, HQQ6
attention, `k4v4` KV cache, INT8 shared/dense/lm-head, layer group size 2,
graph replay enabled, timing instrumentation off.

| Variant | Attention | KV | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Log |
|--------|-----------|----|----------------:|---------------:|-------------------:|-----|--------------:|-----|
| Q122B k4v4 HQQ6, INT4 experts | HQQ6 | k4v4 | 541.9 | 17.66 | 31.23 | 2187/12288 (17.8%) | 736 MB | [log](20260502_phase2fm_q122b_k4v4_hqq6_int4_benchmark.log) |

Notes:
- This run was requested as a README v0.1.63 comparison for the 122B release
  row (`2897 tok/s` prefill, `27.7 tok/s` decode).
- HQQ6/k4v4 is not the faster 122B surface: best prefill was `541.9 tok/s` at
  5K tokens and best internal decode was `17.66 tok/s`.
- Long calibration reached `39,920` prompt tokens with prefill min free
  `692 MB`; timed prefill later emitted VRAM monitor lows below the configured
  `600 MB` safety margin but completed.
- Q235 HQQ6/k4v4 was also attempted in Phase 2FM but no completed standard
  benchmark row was recorded: 2 GB KV failed long calibration with a measured
  scratch OOM, and 1 GB KV loaded/calibrated but was stopped after spending an
  extended GPU-bound period in the first `17,324`-token prefill warmup.

---

## Standard Benchmarks — 2026-05-01 (Phase 2EU MoE decode stages)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: current `./dev speed-test` surface, Qwen3-Coder-Next
`tests/qcn-k4v4-hqq8-int4-benchmark.conf`, INT4 GPU/CPU experts, HQQ8
attention, `k4v4` KV cache, graph replay enabled, timing instrumentation off.

| Stage | Variant | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Log |
|-------|---------|----------------:|---------------:|-------------------:|-----|--------------:|-----|
| Baseline | nearest same-config FA2/HQQ8/k4v4 row | 6,191.3 | 80.03 | n/a | 14256/24576 (58.0%) | 732 MB | [log](20260430_1502_qcn_k4v4_hqq8_int4_benchmark.log) |
| 2 | grouped batched MoE path also allowed for `k_splits=1` | 6,501.4 | 79.75 | 149.47 | 14256/24576 (58.0%) | 732 MB | [log](20260501_phase2eu_stage2_grouped_ksplit1_speed_test.log) |
| 3 | top-k=10 weighted-add specialization, tested then reverted | 6,039.4 | 78.30 | 144.36 | 14256/24576 (58.0%) | 732 MB | [log](20260501_phase2eu_stage3_topk10_accum_speed_test.log) |

Notes:
- Stage 1 graph-internal CUDA event markers were attempted with decode timing
  enabled, but graph replay failed with
  `CUDA_ERROR_INVALID_VALUE` from `cuEventElapsedTime`; the markers were
  reverted and no speed row was recorded.
- Stage 2 is effectively neutral on QCN decode (`79.75` vs `80.03 tok/s`) but
  removes the graph-replay `k_splits=1` rejection encountered during the 122B
  first-run probe.
- Stage 3 made decode slower (`78.30 tok/s`), so the specialization was removed
  and is not part of the retained code.
- Stage 4 was not enabled because the existing GPU-side route/classify path is
  explicitly disabled in code due to prior no-gain measurements and
  first-token illegal-address faults. Re-enabling it would risk degrading a
  working path rather than fixing the underlying correctness issue.

---

## Standard Benchmarks — 2026-05-01 (Phase 2ES cuDNN SDPA opt-in probe)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: current `./dev speed-test` surface, Qwen3-Coder-Next
`tests/qcn-k4v4-hqq8-int4-benchmark.conf`, INT4 GPU/CPU experts, HQQ8
attention, `k4v4` KV cache, graph replay enabled, timing instrumentation off.
cuDNN SDPA was explicitly enabled with `KRASIS_CUDNN_SDPA=1`; the sidecar was
built against `nvidia-cudnn-cu12 9.21.1.3` and `nvidia-cudnn-frontend 1.23.0`.

| Variant | Attention | KV | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|-----------|----|----------------:|---------------:|-----|--------------:|-----|
| QCN k4v4 HQQ8, INT4 experts, cuDNN SDPA opt-in | HQQ8 | k4v4 | 5,246.6 | 80.48 | 14175/24576 (57.7%) | 674 MB | [log](20260501_cudnn_sdpa_speed_test.log) |

Notes:
- Direct sidecar correctness smoke against existing FA2 passed for BF16 causal
  GQA (`B=1`, `S=128`, `HQ=16`, `HKV=2`, `D=128`): both calls returned `0`,
  outputs were finite, mean absolute difference was `6.7e-05`, and max
  absolute difference was `0.00390625`.
- Compared with the nearest same-config FA2/HQQ8/k4v4 QCN benchmark row
  (`20260430_1502_qcn_k4v4_hqq8_int4_benchmark.log`: `6,191.3 tok/s`
  prefill, `80.03 tok/s` decode, `732 MB` min free), cuDNN SDPA reduced
  prefill by about `15.3%`, left decode effectively unchanged, and reduced
  HCS coverage slightly due to lower available VRAM.
- Decision: keep cuDNN SDPA as an explicit opt-in prototype only. It is useful
  because the current latest cuDNN stack builds and runs through Krasis' Rust
  sidecar path, but it is not a default replacement for FA2 on this measured
  surface.

---

## Standard Benchmarks — 2026-05-01 (Phase 2EN FA2/FLA prefill variants)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3.5-35B-A3B `tests/q35b-4-4-hqq8-benchmark.conf`, INT4
GPU/CPU experts, INT8 shared/dense/lm-head, FP8 KV (`fp8_e4m3`, 4v4), layer
group size 2, graph replay enabled, timing instrumentation off for benchmark
rows. Q35B reached full HCS coverage in every benchmark row.

| Step | Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|------|---------|----------------:|---------------:|-----|--------------:|-----|
| Baseline | Phase 2EJ Q35B HQQ8 | 8,029.9 | 116.01 | 10240/10240 (100.0%) | 7180 MB | [log](20260430_2124_q35b_k4v4_hqq8_int4_benchmark.log) |
| 1 | FA2 exact `sm80` restored after failed extra-arch probe | 8,103.7 | 114.27 | 10240/10240 (100.0%) | 7180 MB | [log](20260501_0047_q35b_k4v4_hqq8_int4_fa2_exact_sm80_benchmark.log) |
| 2 | Fixed-length single-sequence FA2 for `start_pos=0` | 8,095.0 | 115.06 | 10240/10240 (100.0%) | 7180 MB | [log](20260501_0054_q35b_k4v4_hqq8_int4_fa2_fixed_benchmark.log) |
| 3 | Opt-in FLA state `BV=64` (`KRASIS_FLA_STATE_BV64=1`) | 5,416.7 | 114.29 | 10240/10240 (100.0%) | 5074 MB | [log](20260501_0103_q35b_k4v4_hqq8_int4_fla_bv64_benchmark.log) |
| 4 | Opt-in FA2 hdim128 causal `64x64` tile (`KRASIS_FA2_HDIM128_CAUSAL_TILE=64x64`) | 8,140.7 | 116.78 | 10240/10240 (100.0%) | 7180 MB | [log](20260501_0128_q35b_k4v4_hqq8_int4_fa2_tile64_benchmark.log) |

Timing notes:
- Step 1 attempted a targeted `sm89/sm90/sm120` FA2 hdim128 fatbin. It failed
  visibly on the 5090 with `FlashAttention-2 forward failed with code -2`
  after the FA2 shim was fixed to return launch errors. Default FA2 is restored
  to the previous known-good `sm80` build; extra-arch FA2 builds are opt-in via
  `KRASIS_FA2_HDIM128_EXTRA_ARCHES=1` or `KRASIS_FA2_ALL_ARCHES=1`.
- Step 2 removed per-GQA-layer cu-seqlens upload for single-sequence prefill,
  but long-prompt FA2 time stayed flat (`598.2 ms`), so it is a small cleanup,
  not a kernel speed win.
- Step 3 was a hard negative: FLA state increased from about `317-339 ms` to
  `2708.9 ms` on the long timing diagnostic, so `BV=64` remains opt-in only.
- Step 4 was mildly positive in clean benchmark terms, but the timing diagnostic
  still showed FA2 at `598.4 ms`; the improvement is small enough that it should
  stay opt-in until more repeat runs and an opt-in witness gate justify changing
  defaults.

---

## Standard Benchmarks — 2026-04-30 (Phase 2EL HQQ6 group-sum + async pointer upload)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Configs: Qwen3.5-35B-A3B `tests/q35b-4-4-hqq6-benchmark.conf`, INT4
GPU/CPU experts, INT8 shared/dense/lm-head, FP8 KV (`fp8_e4m3`, 4v4), layer
group size 2, graph replay enabled, timing instrumentation off for benchmark
rows.

| Variant | Attention | KV | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|-----------|----|----------------:|---------------:|-----|--------------:|-----|
| Q35B k4v4 HQQ6, tiled group-sum | HQQ6 | fp8_e4m3 | 7,788.5 | 112.67 | 10240/10240 (100.0%) | 7148 MB | [log](20260430_2241_q35b_k4v4_hqq6_int4_groupsum_tiled_benchmark.log) |
| Q35B k4v4 HQQ6, tiled group-sum + async ptr upload | HQQ6 | fp8_e4m3 | 7,824.5 | 116.62 | 10240/10240 (100.0%) | 7148 MB | [log](20260430_2250_q35b_k4v4_hqq6_int4_groupsum_async_benchmark.log) |

Notes:
- Timing-enabled Q35B HQQ6 diagnostics showed the group-sum kernel sub-bucket
  improved from `111.6 ms` to `17.8 ms` over the long prefill (`130` calls).
- Clean timing-off benchmarks did not beat the earlier Q35B HQQ6 baseline
  (`7,873.0 tok/s`), so this is recorded as a sub-bucket cleanup rather than a
  headline production speed win.
- Async pointer upload moved the measured startup-calibration MoE time from
  `ptr_upload` into `dma_wait`; total MoE DMA time stayed effectively the same.
  This confirms it removes CPU blocking/attribution noise but does not remove
  the underlying GPU wait for cold expert staging.

---

## Standard Benchmarks — 2026-04-30 (Phase 2EJ Q35B no-cold-DMA HQQ ladder)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Configs: Qwen3.5-35B-A3B 1-GPU configs, INT4 GPU/CPU experts, INT8
shared/dense/lm-head, FP8 KV (`fp8_e4m3`, 4v4), layer group size 2, graph
replay enabled, timing instrumentation off for the benchmark rows.

| Variant | Attention | KV | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|-----------|----|----------------:|---------------:|-----|--------------:|-----|
| Q35B k4v4 HQQ4, INT4 experts | HQQ4 | fp8_e4m3 | 8,047.2 | 115.07 | 10240/10240 (100.0%) | 7852 MB | [log](20260430_2011_q35b_k4v4_hqq4_int4_benchmark.log) |
| Q35B k4v4 HQQ6, INT4 experts | HQQ6 | fp8_e4m3 | 7,873.0 | 115.54 | 10240/10240 (100.0%) | 7148 MB | [log](20260430_2118_q35b_k4v4_hqq6_int4_benchmark.log) |
| Q35B k4v4 HQQ8, INT4 experts | HQQ8 | fp8_e4m3 | 8,029.9 | 116.01 | 10240/10240 (100.0%) | 7180 MB | [log](20260430_2124_q35b_k4v4_hqq8_int4_benchmark.log) |

Notes:
- All three Q35B runs reached `100.0%` HCS coverage, so decode has no cold
  expert DMA and is a useful code-side control lane.
- HQQ4/HQQ6/HQQ8 decode is effectively tied at `115-116 tok/s` when cold DMA
  is removed. That suggests the remaining HQQ decode difference on QCN is not
  mainly attention quant kernel choice.
- Prefill is also tightly clustered (`7.87k-8.05k tok/s`), with HQQ4 and HQQ8
  slightly ahead of HQQ6.
- Timing-enabled Q35B HQQ8 diagnostic:
  `logs/manual/phase2ej_q35b_hqq8_component_timing_20260430.log`.
  Long calibration: `39,920` tokens in `3,889.3 ms` (`10,264 tok/s`),
  attention `2,301.7 ms`, MoE `1,366.5 ms`. Post-HCS decode had `0` cold
  DMA calls and `0.00 MB/tok` cold DMA; graph launch stayed around
  `0.13-0.15 ms/tok`, while graph sync wait accounted for most measured
  decode wall time.

---

## Standard Benchmarks — 2026-04-30 (Phase 2EI HQQ prefill fused correction)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Configs: QCN 1-GPU benchmark-style configs, INT4 GPU/CPU experts, INT8
shared/dense/lm-head, layer group size 2, graph replay enabled, timing
instrumentation off. These runs were made after the Phase 2EI HQQ correction
change that replaces the old custom intercept-correction pass with an FP32
correction GEMM plus a BF16 add kernel after the Marlin projection.

| Variant | Attention | KV | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|-----------|----|----------------:|---------------:|-----|--------------:|-----|
| QCN k4v4 HQQ6, INT4 experts | HQQ6 | k4v4 | 6,186.5 | 76.66 | 14256/24576 (58.0%) | 700 MB | [log](20260430_2030_qcn_k4v4_hqq6_int4_fused_correction_benchmark.log) |
| QCN k6v6 HQQ8, INT4 experts | HQQ8 | k6v6 | 6,314.2 | 79.89 | 14256/24576 (58.0%) | 740 MB | [log](20260430_2054_qcn_k6v6_hqq8_int4_fused_correction_benchmark.log) |

Notes:
- Runs executed via `./dev benchmark ...`, not timing-instrumented profiling.
- Decode values are the benchmark's internal engine numbers. Network round-trip
  numbers are present in the full logs but are not used as decode speed.
- The HQQ6/k4v4 prefill row improved materially over the previous Phase 2EG
  ladder row (`5,363.4 -> 6,186.5 tok/s`), while decode stayed in the same
  class (`75.61 -> 76.66 tok/s`).
- The HQQ8/k6v6 row improved over the previous Phase 2EG comparator
  (`5,992.3 -> 6,314.2 tok/s`), with decode essentially unchanged
  (`81.96 -> 79.89 tok/s`).
- A timing-enabled forced multi-chunk diagnostic measured compressed-KV
  cross-chunk staging at only `37.5 ms` over `108` calls during a
  `39,920`-token prefill capped to `4096`-token chunks. That is about `3.7%`
  of GQA time and about `0.4%` of total prefill time in that diagnostic, so KV
  cross-chunk staging is not currently the main prefill limiter.

---

## Standard Benchmarks — 2026-04-30 (Phase 2EG QCN k4v4 HQQ4/HQQ6/HQQ8 attention ladder + k6v6 comparators)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Configs: QCN 1-GPU benchmark-style configs, INT8 shared/dense/lm-head, layer
group size 2, graph replay enabled, timing instrumentation off. The first
three rows keep INT4 GPU/CPU experts and k4v4 KV fixed while varying only
`CFG_ATTENTION_QUANT`. The final two rows are k6v6/HQQ8 comparators requested
after the ladder, first with INT8 experts and then with INT4 experts.

| Variant | Attention | KV | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|-----------|----|----------------:|---------------:|-----|--------------:|-----|
| QCN k4v4 HQQ4, INT4 experts | HQQ4 | k4v4 | 5,851.5 | 78.76 | 14823/24576 (60.3%) | 698 MB | [log](20260430_1536_qcn_k4v4_hqq4_int4_benchmark.log) |
| QCN k4v4 HQQ6, INT4 experts | HQQ6 | k4v4 | 5,363.4 | 75.61 | 14256/24576 (58.0%) | 700 MB | [log](20260430_1542_qcn_k4v4_hqq6_int4_benchmark.log) |
| QCN k4v4 HQQ8, INT4 experts | HQQ8 | k4v4 | 6,191.3 | 80.03 | 14256/24576 (58.0%) | 732 MB | [log](20260430_1502_qcn_k4v4_hqq8_int4_benchmark.log) |
| QCN k6v6 HQQ8, INT8 experts | HQQ8 | k6v6 | 4,910.6 | 36.85 | 7216/24576 (29.4%) | 656 MB | [log](20260430_1552_qcn_k6v6_hqq8_int8_benchmark.log) |
| QCN k6v6 HQQ8, INT4 experts | HQQ8 | k6v6 | 5,992.3 | 81.96 | 14256/24576 (58.0%) | 740 MB | [log](20260430_1601_qcn_k6v6_hqq8_int4_benchmark.log) |

Notes:
- Runs executed via `./dev benchmark ...`, not timing-instrumented profiling.
- Decode values are the benchmark's internal engine numbers. Network round-trip
  numbers are present in the full logs but are not used as decode speed.
- On this k4v4/INT4 surface, HQQ8 is fastest overall: `6,191.3 tok/s`
  prefill and `80.03 tok/s` internal decode.
- HQQ4 has slightly higher HCS coverage than HQQ6/HQQ8 (`60.3%` versus
  `58.0%`) but still trails HQQ8 on both prefill and decode.
- HQQ6 is currently the slowest row in this clean ladder despite the fast
  prefill staging work; the HQQ6 run emitted prefill-time VRAM monitor lows of
  `518 MB`, below the configured `600 MB` safety margin.
- The INT8/k6v6/HQQ8 comparator is much slower on decode (`36.85 tok/s`)
  because INT8 experts cut HCS soft coverage to `29.4%`; this matches earlier
  INT8 expert findings and argues against INT8 experts for maximum decode speed
  on this hardware.
- The INT4/k6v6/HQQ8 comparator is essentially tied with k4v4/HQQ8 on this
  hardware and current tree: lower prefill (`5,992.3` vs `6,191.3 tok/s`) but
  slightly higher internal decode (`81.96` vs `80.03 tok/s`) with the same HCS
  coverage.

---

## Standard Benchmarks — 2026-04-30 (Phase 2EE QCN INT4 / k4v4 / HQQ4 / HQQ8 sweep)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Configs: QCN 1-GPU benchmark-style configs with INT4 GPU/CPU experts, INT8
shared/dense/lm-head, layer group size 2, graph replay enabled, timing
instrumentation off. These runs were made on the current working tree after
the Phase 2EC/2ED speed patches; those patches were not committed at run time.
Post-run decision: AWQ attention and Polar4 KV are now deprecated and disabled
for new runs. The AWQ/Polar4 rows below are retained only as historical
baselines; current speed work should use HQQ attention with `k4v4`, `k6v6`, or
BF16 KV.

| Variant | Attention | KV | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|-----------|----|----------------:|---------------:|-----|--------------:|-----|
| QCN INT4 baseline (deprecated) | AWQ | Polar4 | 7,325.1 | 96.69 | 16848/24576 (68.6%) | 654 MB | [log](20260430_1458_qcn_polar4_awq_int4_benchmark.log) |
| QCN k4v4 | HQQ8 | k4v4 | 6,191.3 | 80.03 | 14256/24576 (58.0%) | 732 MB | [log](20260430_1502_qcn_k4v4_hqq8_int4_benchmark.log) |
| QCN HQQ4 | HQQ4 | FP8 E4M3 | 5,746.9 | 77.19 | 14823/24576 (60.3%) | 710 MB | [log](20260430_1508_qcn_hqq4_int4_benchmark.log) |
| QCN HQQ8 (deprecated KV) | HQQ8 | Polar4 | 6,131.7 | 80.62 | 14256/24576 (58.0%) | 732 MB | [log](20260430_1513_qcn_polar4_hqq8_int4_benchmark.log) |

Notes:
- Runs executed via `./dev benchmark ...`, not timing-instrumented profiling.
- Decode values are the benchmark's internal engine numbers. Network round-trip
  numbers are present in the full logs but are not used as decode speed.
- AWQ/Polar4 was the fastest measured row in this sweep, but is deprecated
  after this run and should not be used for future speed targets.
- HQQ8 is now in the same prefill speed class across `k4v4` and Polar4 KV:
  `6,191.3` versus `6,131.7 tok/s`; decode is also essentially tied
  (`80.03` versus `80.62 tok/s`).
- HQQ4 is a little slower than HQQ8 on this benchmark surface despite slightly
  higher HCS coverage (`60.3%` versus `58.0%`), landing at `5,746.9 tok/s`
  prefill and `77.19 tok/s` decode.
- The HQQ8 runs emitted prefill-time VRAM monitor lows of `514 MB`, below the
  configured `600 MB` safety margin. The table's min-free VRAM value is the
  benchmark summary's decode min-free value.

---

## Standard Benchmarks — 2026-04-29 (Phase 2DK QCN INT4/HQQ8 KV comparison)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Configs: QCN HQQ8 attention with INT4 GPU/CPU experts, INT8 shared/dense/lm
head, graph replay enabled, timing instrumentation off. Only the KV format
differs. Both runs used the default HQQ8 prefill path with
`KRASIS_HQQ8_PREFILL_MODE` unset.

| Variant | KV bpe | Context @ 1GB KV | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|-------:|-----------------:|----------------:|---------------:|-----|--------------:|-----|
| QCN HQQ8/k4v4, INT4 experts | 5.0 | 136,528 | 5,941.9 | 78.14 | 14256/24576 (58.0%) | 732 MB | [log](20260429_154355_qcn_k4v4_hqq8_int4_benchmark.log) |
| QCN HQQ8/k6v6, INT4 experts | 7.0 | 97,520 | 6,014.9 | 78.05 | 14256/24576 (58.0%) | 740 MB | [log](20260429_154908_qcn_k6v6_hqq8_int4_benchmark.log) |

Notes:
- Runs executed via `./dev benchmark ...`, not timing-instrumented profiling.
- Decode values are the benchmark's internal engine numbers. Network round-trip
  numbers are present in the full logs but are not used as decode speed.
- `k4v4` and `k6v6` are effectively tied on internal decode in this run
  (`78.14` vs `78.05 tok/s`); `k6v6` is slightly faster on best prefill
  (`6,014.9` vs `5,941.9 tok/s`).
- With a fixed `1000 MB` KV cache, `k4v4` provides a larger context window
  (`136,528` tokens) than `k6v6` (`97,520` tokens), matching the expected
  `5.0` versus `7.0` bpe footprint.
- Both runs emitted a prefill-time VRAM monitor warning below the configured
  `600 MB` safety margin (`514 MB` for `k4v4`, `522 MB` for `k6v6`). The table's
  min-free VRAM value is the benchmark summary's decode min-free value.
- A benchmark-report metadata bug was fixed during this pass: newer KV formats
  were previously displayed as `FP8 E4M3` in `benchmark_report.log` even when
  runtime logs correctly showed `Shared k4v4/k6v6 KV cache`.

---

## Standard Benchmarks — 2026-04-28 (Phase 2CT QCN k8v4 HQQ8 faster prefill mode)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: QCN HQQ8 attention with `k8v4` KV cache, graph replay enabled, INT4
GPU/CPU experts, timing instrumentation off. The run used
`KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin-twoscale`, which keeps the
two-scale HQQ8 Marlin prefill slope correction but removes the intercept
correction pass.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| QCN HQQ8/k8v4, INT4 experts, two-scale no-intercept | 5,922.8 | 78.24 | 14256/24576 (58.0%) | 740 MB | [log](20260428_204438_qcn_k8v4_hqq8_twoscale_int4_benchmark.log) |

Notes:
- Accuracy gate before the benchmark: `PASS`, avg exact `34.07`, total exact
  `477/653`, containment `556/653` on `phase2bn_qcn_64tok`.
- Follow-up: this mode is now the default HQQ8 prefill path when
  `KRASIS_HQQ8_PREFILL_MODE` is unset.
- Compared with the previous k8v4 HQQ8 INT4 benchmark using
  `native-fused-marlin-twoscale-intercept`, prefill improved
  `5,245.5 -> 5,922.8 tok/s` and decode moved `76.74 -> 78.24 tok/s`.
- This mode still emitted a prefill-time VRAM monitor warning at `522 MB` free,
  below the configured `600 MB` safety margin; decode min-free was `740 MB`.
- Decode remains well below the old AWQ/Polar4 speed-test baseline
  (`91.77 tok/s`), but timing attribution shows the remaining decode gap is
  dominated by graph replay sync wait and cold expert DMA rather than HQQ
  attention math.
- Reduction: `logs/manual/phase2ct_qcn_hqq8_speed_followup_20260428.md`.

---

## Standard Benchmarks — 2026-04-28 (Phase 2CS QCN k8v4 HQQ8 expert-bit comparison)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Configs: QCN HQQ8 attention with `k8v4` KV cache, graph replay enabled, timing
instrumentation off. Both runs used
`KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin-twoscale-intercept`; only the
GPU/CPU expert bits differ.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| QCN HQQ8/k8v4, INT8 experts | 4,238.1 | 35.00 | 7175/24576 (29.2%) | 752 MB | [log](20260428_194459_qcn_k8v4_hqq8_int8_benchmark.log) |
| QCN HQQ8/k8v4, INT4 experts | 5,245.5 | 76.74 | 14256/24576 (58.0%) | 708 MB | [log](20260428_195237_qcn_k8v4_hqq8_int4_benchmark.log) |

Notes:
- Runs executed via `./dev benchmark ...`, not timing-instrumented profiling.
- Decode values are the benchmark's internal engine numbers. Network round-trip
  numbers are present in the full logs but are not used as decode speed.
- INT4 experts are substantially faster on this surface because HCS soft
  coverage roughly doubles (`29.2% -> 58.0%`) and expert cache bandwidth is
  lower.
- The INT4 run emitted a prefill-time VRAM monitor warning at `564 MB` free,
  below the configured `600 MB` safety margin. The benchmark summary's min free
  VRAM row is the decode min-free value.
- Reduction: `logs/manual/phase2cs_qcn_k8v4_hqq8_benchmark_reduction_20260428.md`.

---

## Standard Benchmarks — 2026-04-27 (Phase 2BR QCN Polar4 HQQ speed-test variants)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Configs: speed-test-equivalent QCN runs with the AWQ attention choice replaced
by HQQ4SC or HQQ8. The rest of the surface matches the QCN AWQ/Polar4 speed
test: INT4 GPU/CPU experts, Polar4 KV, INT8 shared/dense/lm-head, layer group
size 2, timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| QCN AWQ/Polar4 speed-test baseline, 2026-04-26 | 7,295.6 | 91.77 | 16848/24576 (68.6%) | 688 MB | [log](20260426_211755_qcn_polar4_awq_speed_regression_check.log) |
| QCN HQQ4SC/Polar4 speed-test variant | 466.2 | 79.17 | 14823/24576 (60.3%) | 706 MB | [log](20260427_201735_qcn_polar4_hqq4sc_speed_variant.log) |
| QCN HQQ8/Polar4 speed-test variant, scalar HQQ prefill | 486.8 | 75.16 | 14256/24576 (58.0%) | 734 MB | [log](20260427_202957_qcn_polar4_hqq8_speed_variant.log) |
| QCN HQQ8/Polar4 speed-test variant, Marlin prefill prototype | 5,011.6 | 79.80 | 14256/24576 (58.0%) | 734 MB | [log](20260427_215803_qcn_polar4_hqq8_fastprefill_speed.log) |
| QCN HQQ8/Polar4 speed-test variant, native fused Marlin experiment | 7,132.9 | 79.68 | 14256/24576 (58.0%) | 734 MB | [log](20260427_230455_qcn_polar4_hqq8_native_fused_speed.log) |
| QCN HQQ8/Polar4 speed-test variant, two-scale + intercept Marlin experiment | 5,340.1 | 83.45 | 14256/24576 (58.0%) | 702 MB | [log](20260428_063003_qcn_polar4_hqq8_twoscale_intercept_speed.log) |

Notes:
- Runs executed via `./dev benchmark tests/qcn-polar4-hqq4sc.conf` and
  `./dev benchmark tests/qcn-polar4-hqq8.conf` because `./dev speed-test` is
  still the fixed AWQ config.
- Decode values are the benchmark's internal engine numbers. Network round-trip
  numbers are present in the full logs but are not used as decode speed.
- HQQ4SC/HQQ8 decode is usable but below AWQ on this exact Polar4 speed-test
  surface: HQQ4SC is about `86%` of AWQ decode, HQQ8 about `82%`.
- The first HQQ8/Polar4 row used the old scalar HQQ prefill kernel and was not
  acceptable: about `6-7%` of AWQ prefill.
- The Marlin prefill prototype replaces the scalar HQQ8 prefill GEMM with a
  two-pass Marlin U8B128 path plus grouped zero correction. It improves HQQ8
  prefill from `486.8` to `5,011.6 tok/s` (`10.3x`) on this surface, reaching
  about `68.7%` of the AWQ/Polar4 baseline prefill.
- The native fused Marlin experiment uses Marlin U8 with BF16 float zero-points
  as a single GEMM. It improves HQQ8 prefill from `486.8` to `7,132.9 tok/s`
  (`14.7x`) and reaches `97.8%` of the AWQ/Polar4 baseline prefill. Accuracy
  did not fully hold versus the residual Marlin HQQ8 prototype (`14.29` average
  exact prefix versus `15.79`), so this is a speed/architecture result rather
  than a production default.
- The two-scale + intercept Marlin experiment uses one U8 float-zp Marlin GEMM
  with a second BF16 scale plane plus a compact FP32 intercept correction. It
  improves the QCN HQQ8 64-token witness gate over both residual Marlin and
  native fused v1 (`18.14` average exact prefix, `326/653` decode containment),
  while prefill lands between those two speed points at `5,340.1 tok/s`.
- The Marlin prefill prototype passed the QCN 64-token witness gate before this
  speed run (`14/14` first-token, `15.79` average exact prefix), but selected
  first-token logprob delta remains worse than the previous fixed HQQ8 path; it
  should stay treated as a prototype until that residual quality gap is closed.
- VRAM monitor lows during timed prefill: HQQ4SC reached `532 MB` free; HQQ8
  scalar and Marlin-prefill runs reached `516 MB` free.
- Reduction: `logs/manual/phase2br_qcn_polar4_hqq_speed_reduction_20260427.md`.

---

## Standard Benchmarks — 2026-04-26 (Phase 2BK INT8 exception top-k validation)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Configs: QCN and Qwen3.5 HQQ4 baselines versus explicit top-4/top-8/top-16 INT8
exception manifests for layer-0 `in_proj_qkvz`. Timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| QCN HQQ4 baseline | 7,242.6 | 70.79 | 12960/24576 (52.7%) | 686 MB | [log](20260426_220134_qcn_hqq4_phase2bk_base.log) |
| QCN HQQ4 + INT8 exceptions top-4 | 6,936.2 | 71.14 | 12960/24576 (52.7%) | 692 MB | [log](20260426_220448_qcn_hqq4_phase2bk_int8_top4.log) |
| QCN HQQ4 + INT8 exceptions top-8 | 7,854.1 | 70.92 | 12960/24576 (52.7%) | 690 MB | [log](20260426_220800_qcn_hqq4_phase2bk_int8_top8.log) |
| QCN HQQ4 + INT8 exceptions top-16 | 7,833.9 | 68.59 | 12879/24576 (52.4%) | 758 MB | [log](20260426_221107_qcn_hqq4_phase2bk_int8_top16.log) |
| Q35 HQQ4 baseline | 7,146.8 | 114.01 | 10240/10240 (100.0%) | 5452 MB | [log](20260426_221419_q35_hqq4_phase2bk_base.log) |
| Q35 HQQ4 + INT8 exceptions top-4 | 7,186.4 | 116.59 | 10240/10240 (100.0%) | 5428 MB | [log](20260426_221755_q35_hqq4_phase2bk_int8_top4.log) |
| Q35 HQQ4 + INT8 exceptions top-8 | 7,138.4 | 114.16 | 10240/10240 (100.0%) | 5426 MB | [log](20260426_222218_q35_hqq4_phase2bk_int8_top8.log) |
| Q35 HQQ4 + INT8 exceptions top-16 | 7,428.9 | 104.14 | 10240/10240 (100.0%) | 5396 MB | [log](20260426_222643_q35_hqq4_phase2bk_int8_top16.log) |

Notes:
- Runs executed via `./dev benchmark ...`, not timing-instrumented profiling.
- Decode values are the benchmark's internal engine numbers. Network round-trip
  numbers are present in the full logs but are not used as decode speed.
- Top-4 was the only variant that improved witness selected-logprob on both QCN
  and Q35 in the matching Phase 2BK witness set.
- Top-8 regressed witness selected-logprob on both models. Top-16 improved
  selected-logprob less than top-4 and had a meaningful decode-speed cost,
  especially on Q35.
- The associated quality reduction is
  `logs/manual/phase2bk_int8_exception_topk_validation_reduction_20260426.md`.

---

## Standard Benchmarks — 2026-04-26 (QCN AWQ/Polar4 speed regression check)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: standard `./dev speed-test` QCN AWQ/Polar4 path after Phase 2BJ decode INT8 exception work. Timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| QCN AWQ/Polar4 speed-test check | 7,295.6 | 91.77 | 16848/24576 (68.6%) | 688 MB | [log](20260426_211755_qcn_polar4_awq_speed_regression_check.log) |

Notes:
- Run executed via `./dev speed-test`.
- This checks the historical `90+ tok/s` QCN path directly after Phase 2BJ.
- Result: standard QCN AWQ/Polar4 decode remains in the expected `90+ tok/s` class; the `71.30 tok/s` number belongs to the separate QCN HQQ4 + INT8 exception top-4 config.
- Decode values are the benchmark's internal engine numbers. Network round-trip numbers are present in the full log but are not used as decode speed.

---

## Standard Benchmarks — 2026-04-26 (Phase 2BJ INT8 exception prefill+decode top-4)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Configs: QCN and Qwen3.5 HQQ4 with explicit top-4 INT8 exception manifests after decode-side exception execution was implemented. Timing instrumentation off. Baselines are the same-day Phase 2BH HQQ4 baseline runs below.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| QCN HQQ4 baseline | 7,912.5 | 73.97 | 12960/24576 (52.7%) | 686 MB | [log](20260426_193218_qcn_hqq4_int8_exception_phase2bh_baseline.log) |
| QCN HQQ4 + INT8 exceptions top-4 | 6,984.6 | 71.30 | 12960/24576 (52.7%) | 692 MB | [log](20260426_210438_qcn_hqq4_int8_exception_phase2bj_top4_prefill_decode.log) |
| Q35 HQQ4 baseline | 9,360.5 | 116.24 | 10240/10240 (100.0%) | 5452 MB | [log](20260426_193851_q35_hqq4_int8_exception_phase2bh_baseline.log) |
| Q35 HQQ4 + INT8 exceptions top-4 | 9,523.2 | 111.88 | 10240/10240 (100.0%) | 5428 MB | [log](20260426_210751_q35_hqq4_int8_exception_phase2bj_top4_prefill_decode.log) |

Notes:
- Runs executed via `./dev benchmark ...`, not timing-instrumented profiling.
- QCN top-4 prefill+decode path had lower internal prefill throughput than the HQQ-only baseline (`-11.7%`) and lower decode throughput (`-3.6%`).
- Q35 top-4 prefill+decode path had slightly higher internal prefill throughput (`+1.7%`) but lower decode throughput (`-3.8%`).
- Decode values are the benchmark's internal engine numbers. Network round-trip numbers are present in the full logs but are not used as decode speed.
- The associated implementation and quality reduction is `logs/manual/phase2bj_int8_exception_decode_runtime_reduction_20260426.md`.

---

## Standard Benchmarks — 2026-04-26 (Phase 2BH INT8 exception prefill prototype)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Configs: QCN and Qwen3.5 HQQ4 baselines versus explicit single-block INT8 exception prefill manifests. Timing instrumentation off. These runs were used only to measure the opt-in Phase 2BH prefill path; no default/runtime promotion was made.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| QCN HQQ4 baseline | 7,912.5 | 73.97 | 12960/24576 (52.7%) | 686 MB | [log](20260426_193218_qcn_hqq4_int8_exception_phase2bh_baseline.log) |
| QCN HQQ4 + INT8 exception group 12 | 7,832.5 | 71.95 | 12960/24576 (52.7%) | 682 MB | [log](20260426_193537_qcn_hqq4_int8_exception_phase2bh_g12.log) |
| Q35 HQQ4 baseline | 9,360.5 | 116.24 | 10240/10240 (100.0%) | 5452 MB | [log](20260426_193851_q35_hqq4_int8_exception_phase2bh_baseline.log) |
| Q35 HQQ4 + INT8 exception group 14 | 9,332.0 | 116.06 | 10240/10240 (100.0%) | 5452 MB | [log](20260426_194201_q35_hqq4_int8_exception_phase2bh_g14.log) |

Notes:
- Runs executed via `./dev benchmark ...`, not timing-instrumented profiling.
- QCN INT8 exception prefill overhead was about `-1.0%` internal prefill throughput and `-4 MB` measured min-free VRAM difference.
- Q35 INT8 exception prefill overhead was about `-0.3%` internal prefill throughput with unchanged measured min-free VRAM.
- Decode values are the benchmark's internal engine numbers. Network round-trip numbers are present in the full logs but are not used as decode speed.
- The associated quality reduction is `logs/manual/phase2bh_int8_exception_runtime_reduction_20260426.md`.

---

## Standard Benchmarks — 2026-04-16 (QCN Polar4 AWQ after QK FP32 decision rerun)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3-Coder-Next, 1 GPU, AWQ attention, Polar4 KV, GPU decode, HCS on, timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| post QK FP32 decision rerun | 7,308.0 | 95.06 | 16848/24576 (68.6%) | 682 MB | [log](20260416_065456_qcn_polar4_awq_5090_qk_fp32_policy_rerun.log) |

Notes:
- Run executed via `./dev speed-test` on branch `gpu-debug-trace` at `8e50e32`.
- Internal prefill runs:
  - `1K`: `524.9 tok/s`
  - `5K`: `2372.6 tok/s`
  - `10K`: `4166.5 tok/s`
  - `20K`: `5913.7 tok/s`
  - `35K`: `7308.0 tok/s`
  - `50K`: `6655.0 tok/s`
- Internal decode runs:
  - `50`: `91.84 tok/s`
  - `100`: `95.06 tok/s`
  - `250`: `81.76 tok/s`
- Round-trip HTTP runs:
  - `50`: `185.41 tok/s`
  - `100`: `111.49 tok/s`
  - `250`: `88.65 tok/s`
- Calibration summary:
  - short decode probe: `60.7 tok/s`
  - long decode probe: `49.2 tok/s`
  - transient deltas: short prefill `23678 MB`, long prefill `26206 MB`, short decode `50 MB`, long decode `2 MB`
  - worst-case prefill scratch reservation: `26743 MB` at `50000` tokens
- HCS load summary:
  - `16848/24576` experts loaded (`68.6%`)
  - soft HCS footprint `26199.4 MB`
- Standard benchmark log archived at `benchmarks/20260416_065456_qcn_polar4_awq_5090_qk_fp32_policy_rerun.log`.

---

## Standard Benchmarks — 2026-04-15 (QCN Polar4 AWQ linear-attention AWQ fold review)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3-Coder-Next, 1 GPU, AWQ attention, Polar4 KV, GPU decode, HCS on, timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| LA AWQ runtime disabled | 7,233.2 | 94.01 | 16848/24576 (68.6%) | 682 MB | [log](20260415_224343_qcn_polar4_awq_5090_la_awq_runtime_disabled.log) |
| LA AWQ fold restored | 7,241.5 | 94.65 | 16848/24576 (68.6%) | 682 MB | [log](20260415_224836_qcn_polar4_awq_5090_la_awq_fold_restored.log) |

Notes:
- Both runs used `./dev speed-test` on branch `gpu-debug-trace`.
- This pair was run specifically to evaluate whether linear-attention input projections should participate in the same AWQ input-scale-and-fold contract as calibration.
- Result:
  - prefill improved slightly: `7233.2 -> 7241.5 tok/s`
  - internal decode improved slightly: `94.01 -> 94.65 tok/s`
  - HCS coverage and minimum free VRAM were unchanged
- Internal prefill runs with fold restored:
  - `1K`: `520.9 tok/s`
  - `5K`: `2395.2 tok/s`
  - `10K`: `4070.0 tok/s`
  - `20K`: `5832.3 tok/s`
  - `35K`: `7241.5 tok/s`
  - `50K`: `6379.7 tok/s`
- Internal decode runs with fold restored:
  - `50`: `90.66 tok/s`
  - `100`: `94.65 tok/s`
  - `250`: `86.17 tok/s`
- Round-trip HTTP best with fold restored:
  - `167.68 tok/s` at `50` tokens

---

## Standard Benchmarks — 2026-04-13 (QCN Polar4 AWQ after BF16 policy / dead TRTLLM cleanup)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3-Coder-Next, 1 GPU, AWQ attention, Polar4 KV, GPU decode, HCS on, timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| post BF16 policy cleanup; decode benchmark EOS-early failure | 7,777.2 | FAILED (EOS at 2 tokens) | 16848/24576 (68.6%) | 696 MB | [log](20260413_171619_qcn_polar4_awq_5090_eos_early_decode_failure.log) |

Notes:
- Run executed via `./dev speed-test` on branch `gpu-debug-trace` after pushing `5e80acb`.
- Internal prefill runs:
  - `1K`: `549.1 tok/s`
  - `5K`: `2680.3 tok/s`
  - `10K`: `4194.3 tok/s`
  - `20K`: `6500.0 tok/s`
  - `35K`: `7777.2 tok/s`
  - `50K`: `7694.4 tok/s`
- Internal decode benchmark did not produce a valid throughput number:
  - `50`: failed, EOS at `2` tokens
  - `100`: failed, EOS at `2` tokens
  - `250`: failed, EOS at `2` tokens
- Round-trip HTTP benchmark also failed the same way:
  - `50`: failed, EOS at `2` tokens
  - `100`: failed, EOS at `2` tokens
  - `250`: failed, EOS at `2` tokens
- Standard benchmark log archived at `benchmarks/20260413_171619_qcn_polar4_awq_5090_eos_early_decode_failure.log`.

---

## Standard Benchmarks — 2026-04-13 (QCN Polar4 AWQ after capture-box and Nemotron reference fixes)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3-Coder-Next, 1 GPU, AWQ attention, Polar4 KV, GPU decode, HCS on, timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| post capture-box hardening and Nemotron compat fixes | 7,554.2 | 92.59 | 16848/24576 (68.6%) | 732 MB | [log](20260413_005327_qcn_polar4_awq_5090.log) |

Notes:
- Run executed via `./dev speed-test` on branch `gpu-debug-trace` after pushing `9cc7a91`.
- Internal prefill runs:
  - `1K`: `510.2 tok/s`
  - `5K`: `2446.1 tok/s`
  - `10K`: `4012.7 tok/s`
  - `20K`: `5810.8 tok/s`
  - `35K`: `7554.2 tok/s`
  - `50K`: `6579.1 tok/s`
- Internal decode runs:
  - `50`: `92.59 tok/s`
  - `100`: `87.11 tok/s`
  - `250`: `91.96 tok/s`
- Round-trip HTTP best:
  - `129.55 tok/s` at `50` tokens
- Standard benchmark log archived at `benchmarks/20260413_005327_qcn_polar4_awq_5090.log`.

---

## Standard Benchmarks — 2026-04-04 (QCN Polar4 AWQ after HCS async pointer lifetime fix)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3-Coder-Next, 1 GPU, AWQ attention, Polar4 KV, GPU decode, HCS on, timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| accuracy async ptr lifetime fix | 7,891.4 | 98.59 | 17010/24576 (69.2%) | 682 MB | [log](20260404_011629_qcn_polar4_awq_5090_accuracy_hcs_async_ptr_fix.log) |

Notes:
- Run executed via `./dev speed-test` on branch `accuracy`.
- Fix restored stable host backing for async `cuMemcpyHtoDAsync_v2` expert-pointer table uploads in `src/gpu_decode.rs`.
- This was run after a broken `release-test` on current main/accuracy had shown QCN collapsing into repeated `S` tokens and failing mini reference validation immediately.
- Post-fix QCN AWQ reference-test result on the same branch state:
  - `./dev reference-test qcn-a4`
  - `13/13` prompts PASS
  - first-token match `12/13`
  - prefill argmax match `249/273 (91%)`
  - prefill top-10 containment `273/273 (100%)`
  - report: `logs/reference-test_20260404_011152/reference_test.html`

---

## Standard Benchmarks — 2026-04-03 (QCN Polar4 AWQ speed-test rerun on c35d9b0)

Hardware: EPYC 7742, 995 GB RAM, 1x RTX 5090 32 GB used for benchmark, 2x RTX 5090 present.

Config: Qwen3-Coder-Next, 1 GPU, AWQ attention, Polar4 KV, GPU decode, HCS on, timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| c35d9b0 speed-test rerun | FAIL before timed benchmark | 84.8 tok/s short calibration only | not reached | 3262 MB during short prefill probe | [log](20260403_174355_qcn_polar4_awq_5090_failed_illegal_address.log) |

Notes:
- Run executed via `./dev speed-test` on detached `c35d9b0`.
- Load, warmup, decode-store setup, and short calibration passed.
- Failure occurred in long VRAM calibration at 39,920 prompt tokens inside `gpu_store.rust_prefill_tokens(...)`.
- Error: `CUDA_ERROR_ILLEGAL_ADDRESS (grid=(39920, 1, 1), block=(1024, 1, 1), smem=4096, nparams=6)`.
- Cleanup also hit a Rust destructor panic while tearing down `GpuDecodeStore` after the illegal address.

---

## Standard Benchmarks — 2026-04-01 (QCN Polar4 AWQ padding rewrite)

Hardware: EPYC 7742, 995 GB RAM, 1x RTX 5090 32 GB used for benchmark, 2x RTX 5090 present.

Config: Qwen3-Coder-Next, 1 GPU, AWQ attention, Polar4 KV, GPU decode, HCS on, timing instrumentation off.

| Variant | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|--------|----------------:|---------------:|-----|--------------:|-----|
| Intermediate rewrite (cached dummy ptrs + real-expert alias padding) | 7,398.3 | 90.99 | 17010/24576 (69.2%) | 686 MB | [report](../logs/dev-benchmark_20260401_083831/benchmark_report.log) |
| Final rewrite (cached dummy ptrs + dummy-only zero-weight padding) | 7,769.5 | 96.43 | 17010/24576 (69.2%) | 686 MB | [report](20260401_084451_qcn_polar4_awq_5090_padding_rewrite.log) |

Notes:
- The first rewrite removed only the per-step `cuMemcpyDtoH` and did not recover the regression.
- The final rewrite shows the remaining loss came from aliasing zero-weight slots onto a real expert during replay.
- Standard benchmark log archived at `benchmarks/20260401_084451_qcn_polar4_awq_5090_padding_rewrite.log`.

---

## Standard Benchmarks — 2026-04-01 (QCN Polar4 AWQ after decode harness work)

Hardware: EPYC 7742, 1007 GB RAM, 1x RTX 5090 32 GB selected for the run.

Config: Qwen3-Coder-Next, 1 GPU, INT4 GPU experts, INT4 CPU experts, AWQ attention, Polar4 KV, layer group size 2, timing off.

| Model | GPUs | GPU/CPU bits | Attention | KV | Prefill (tok/s) | Decode (tok/s) | HCS | Min free VRAM | Log |
|-------|-----:|-------------:|----------:|---:|----------------:|---------------:|----:|--------------:|-----|
| Qwen3-Coder-Next | 1 | INT4/INT4 | AWQ | Polar4 | 7645.2 | 96.10 | 17010/24576 (69.2%) | 686 MB | [log](20260401_101548_qcn_polar4_awq_5090_decode_harness.log) |

Notes:
- Internal decode results were 96.10 tok/s at 50 tokens, 94.35 tok/s at 100 tokens, and 93.84 tok/s at 250 tokens.
- Internal prefill peaked at 7645.2 tok/s on the 35K-token prompt.
- This confirms the decode padding rewrite and decode-harness changes did not knock QCN Polar4 AWQ out of its expected mid-90 tok/s decode class.

---

## GPU Decode Benchmark — 2026-03-02 (5090, 40% HCS, pinned memory)

**Hardware:** EPYC 7742, 995 GB RAM, 1x RTX 5090 32 GB, PCIe 4.0 x16 (27 GB/s peak).

**Config:** QCN (Qwen3-Coder-Next), INT4 GPU/CPU, BF16 attention, LGS=2, GPU decode (Rust, zero GIL), HCS 40.2% (9,869/24,576 experts), pinned expert memory for async DMA, no debug/timing instrumentation.

### Decode Speed

| Prompt | Tokens | Decode Time | Decode Speed | TTFT |
|--------|-------:|------------:|-------------:|-----:|
| Short (math) | 199 | 7.04s | 28.1 tok/s | 5.31s |
| Medium (caches) | 499 | 16.43s | 30.3 tok/s | 5.33s |
| Code (BST) | 499 | 17.29s | 28.8 tok/s | 5.30s |
| Long (essay) | 799 | 23.16s | 34.5 tok/s | 5.30s |
| **Average** | | | **30.4 tok/s** | **5.31s** |

### Prefill Speed (from server log, includes ~5.3s layer streaming overhead)

| Input Tokens | TTFT | Prefill Compute (est) | Prefill Speed (est) |
|-------------:|-----:|----------------------:|--------------------:|
| 64 | 5.29s | ~0.0s | n/a (streaming dominated) |
| 576 | 5.31s | ~0.01s | n/a |
| 1,126 | 5.28s | ~0.0s | n/a |
| 2,236 | 5.29s | ~0.0s | n/a |
| 4,456 | 5.32s | ~0.02s | ~838 tok/s |
| 8,906 | 5.32s | ~0.02s | ~1,674 tok/s |
| 17,796 | 5.69s | ~0.39s | ~3,129 tok/s |
| 27,796 | 7.64s | ~2.34s | ~3,636 tok/s |

### Notes
- TTFT is dominated by layer group streaming (~5.3s constant overhead for lgs=2, 48 layers, 24 groups)
- Actual prefill compute only becomes visible above ~8K tokens
- Peak prefill throughput: ~3,636 tok/s at 28K tokens
- Decode speed varies 28-35 tok/s, higher on longer outputs (routing stabilizes)
- VRAM: 24,923 MB used, 7,196 MB free during decode, lowest watermark 2,888 MB (during 28K prefill)
- Rust KV cache limited to 8,192 tokens (prompts >8K skip decode)

Full log: [20260302-gpu-decode-5090-qcn-40pct-hcs.log](../logs/benchmarks/20260302-gpu-decode-5090-qcn-40pct-hcs.log)

---

## Standard Benchmarks — 2026-02-27 (Rust server, unified timing)

**Hardware:** EPYC 7742 (64 cores, 4 NUMA nodes), DDR4-2666 8-channel, 1x RTX 2000 Ada 16 GB, PCIe 4.0 x8.

Config: 20K–50K token prompts, FP8 KV cache, BF16 attention, INT8 shared_expert/dense_mlp/lm_head, 40 CPU threads, NUMA thread pinning + interleaved allocation, LGS=2, pure CPU decode, Rust HTTP server with ring buffer SSE.

| Model | GPUs | GPU/CPU bits | Engine Prefill | Engine Decode | Network Prefill | Network Decode | Overhead | Log |
|-------|-----:|-------------:|---------------:|--------------:|----------------:|---------------:|---------:|-----|
| Qwen3-Coder-Next | 1 | INT4/INT4 | 1,003 tok/s | 12.97 tok/s | 932 tok/s | 12.13 tok/s | 7.1% / 6.5% | [log](../logs/benchmarks/Qwen3-Coder-Next_native_1gpu_int4gpu_int4cpu_stream_lgs2.log) |

### Key changes from previous benchmarks

- **Rust-internal timing**: Both engine and network decode use Rust `Instant` timers. Previous Python timing included `torch.cuda.synchronize()` overhead, making engine decode appear 33% slower than network (impossible).
- **Ring buffer SSE**: Decode loop pushes to mpsc channel, writer thread flushes every 100ms. First token flushed immediately for accurate TTFT.
- **Unified tokenization**: Both paths use `apply_chat_template(enable_thinking=False)`. Network sends text (not pre-tokenized IDs).
- **Model warmup before benchmarks**: Full generate cycle runs before any measurement, paying all cold-start costs.

---

## Standard Speed Benchmark — 2026-04-03 (resolution, BF-01 host ptr-table base import)

Hardware: 1x RTX 5090

Config: Qwen3-Coder-Next, INT4 experts, AWQ attention, Polar4 KV, standard command `./dev speed-test`, timing instrumentation off.

| Date | Commit | Change | Prefill (tok/s) | Decode (tok/s) | Round trip (tok/s) | HCS | Min free VRAM | Status | Log |
|------|--------|--------|----------------:|---------------:|-------------------:|-----|--------------:|--------|-----|
| 2026-04-03 18:53 | 83dd3b0 + local BF-01 | Pointer-table fused MoE host base set to null in ptr-table mode; BF-02 already present | 7,584.7 | 100.38 | 138.95 | 16929/24576 (68.9%) | 738 MB | PASS | [log](20260403_185345_qcn_polar4_awq_5090_bf01_host_null_base.log) |
| 2026-04-03 19:20 | 83dd3b0 + local BF-01 + BF-03 cache | Cache fused MoE `C_tmp` floor calculation once per model/device config and reuse in hot path | FAIL before timed benchmark | 80.6 tok/s short calibration only | not reached | not reached | 3248 MB during short prefill probe | FAIL | [log](20260403_192025_qcn_polar4_awq_5090_bf03_cached_ctmp_failed_illegal_address.log) |
| 2026-04-03 19:26 | 83dd3b0 + local BF-01, BF-03 cache reverted | Revert the one-time `C_tmp` cache follow-up and rerun standard speed test | FAIL before timed benchmark | 80.6 tok/s short calibration only | not reached | not reached | 3248 MB during short prefill probe | FAIL | [log](20260403_192634_qcn_polar4_awq_5090_bf03_revert_failed_illegal_address.log) |
| 2026-04-03 19:38 | 83dd3b0 + local BF-01 + BF-03 cache | Post-reboot rerun with BF-03 one-time `C_tmp` cache reapplied | 7,844.1 | 99.29 | 182.51 | 17010/24576 (69.2%) | 686 MB | PASS | [log](20260403_193344_qcn_polar4_awq_5090_bf03_reapplied_post_reboot.log) |
| 2026-04-03 19:51 | 3b36240 + local BF-04 clean import | Replace drifting ptr-table fused-MoE `B` progression with explicit expert base + signed slice rebasing on expert/block transitions | 7,513.1 | 98.68 | 127.13 | 17010/24576 (69.2%) | 686 MB | PASS | [log](20260403_195132_qcn_polar4_awq_5090_bf04_clean_rebase.log) |
| 2026-04-03 20:11 | fb49b0f + local BF-05 clean import | Keep ptr-table `B` fetch source indices signed through slice rewinds and guard the hazard path with `cp_async4_pred` | 7,515.0 | 97.82 | 128.37 | 17010/24576 (69.2%) | 686 MB | PASS | [log](20260403_201155_qcn_polar4_awq_5090_bf05_signed_fetch_guard.log) |
| 2026-04-03 20:36 | 4aa3bee + local no-valid-block guard | Exit `update_next_moe_block_data()` cleanly when invalid-block scanning reaches the padded tail without finding another valid expert block | 7,606.7 | 99.88 | 137.25 | 17010/24576 (69.2%) | 686 MB | PASS | [log](20260403_203640_qcn_polar4_awq_5090_no_valid_block_guard.log) |
| 2026-04-03 20:54 | 7d09912 + local BF-09 | Feed the active decode-store CUDA ordinal into `PrefillModelConfig` so fused-MoE shared-memory capability queries stop assuming GPU 0 | 7,745.6 | 95.55 | 120.95 | 17010/24576 (69.2%) | 686 MB | PASS | [log](20260403_205423_qcn_polar4_awq_5090_bf09_device_ordinal.log) |
| 2026-04-03 21:23 | df6c259 + local BF-13 | Split BF16 shared-expert cuBLAS ownership so `shared_stream` uses a dedicated handle instead of retargeting the main prefill handle across streams | 7,498.6 | 100.62 | 137.97 | 17010/24576 (69.2%) | 682 MB | PASS | [log](20260403_212300_qcn_polar4_awq_5090_bf13_shared_cublas_handle.log) |
| 2026-04-03 22:03 | 68f1557 + local BF-10 | Split fused-MoE sorted scatter finalization into a second same-stream kernel so padding and `expert_ids` are written only after scatter completion | 7,510.0 | 100.18 | 135.55 | 17010/24576 (69.2%) | 682 MB | PASS | [log](20260403_220350_qcn_polar4_awq_5090_bf10_scatter_finalize_split.log) |
| 2026-04-03 22:33 | 1532389 + local FLA fail-closed | Fail startup for linear-attention models when vendored FLA cannot load, keeping `KRASIS_NO_FLA=1` as the only explicit opt-out to the slower custom LA path | 7,572.8 | 97.61 | 134.15 | 17010/24576 (69.2%) | 682 MB | PASS | [log](20260403_223342_qcn_polar4_awq_5090_fla_fail_closed.log) |
| 2026-04-03 23:00 | 1532389 + local FLA fail-closed + C-02 | Preserve raw `q`/`k` in canonical head-major layout before non-FLA `la_apply_beta`, and emit canonical `k_beta` directly from the beta kernel | 7,943.9 | 98.70 | 133.80 | 17010/24576 (69.2%) | 682 MB | PASS | [log](20260403_230008_qcn_polar4_awq_5090_c02_raw_k_canonical.log) |

Notes:
- The BF-03 cache edit built cleanly through `./dev build`.
- This benchmark failed in long VRAM calibration at `39,920` prompt tokens before the timed benchmark section.
- Failure remained `CUDA_ERROR_ILLEGAL_ADDRESS (grid=(39920, 1, 1), block=(1024, 1, 1), smem=4096, nparams=6)`.
- BF-13 cleared warmup, long calibration, HCS load, and the full timed benchmark on the standard QCN AWQ Polar4 path.
- BF-13 behaved like a correctness/stability fix with no obvious throughput regression; decode returned to ~100 tok/s while preserving async shared-expert overlap.
- Reverting the BF-03 cache follow-up did not restore a successful run on this attempt; the same long-calibration illegal-address fault reproduced with the same short-calibration numbers.
- After reboot, the same BF-03 cache state completed the full standard benchmark cleanly and produced the best prefill result in this local series.
- The BF-04 clean import also completed the full standard benchmark cleanly on the same branch state, but did not improve throughput versus the earlier post-reboot BF-03 pass.
- The BF-05 clean import also completed the full standard benchmark cleanly; throughput stayed effectively flat versus BF-04 while preserving the signed rewind-safe pointer-table fetch path.
- The no-valid-block guard also completed the full standard benchmark cleanly; it is a cheap control-flow correctness fix and did not regress standard prefill or decode throughput.
- The BF-09 actual-device-ordinal wiring also completed the full standard benchmark cleanly; on this single-GPU path it behaves like a correctness/generalization fix rather than a speed optimization.
- BF-10 also completed the full standard benchmark cleanly; the extra finalize launch did not materially change throughput on the standard QCN AWQ Polar4 path and removed the grid-wide race from the sorted scatter/finalize step.
- The FLA fail-closed change also completed the full standard benchmark cleanly; startup still succeeds on the shipped QCN path with vendored FLA present, but LA models will now fail visibly instead of silently degrading to the older custom LA kernels when FLA sidecar loading breaks.
- The C-02 layout-preservation change also completed the full standard benchmark cleanly on the combined branch state. This benchmark still exercises the shipped FLA path, so it confirms no regression in the standard product path but does not by itself prove non-FLA correctness.

## Standard Benchmarks — 2026-02-25 (NUMA-optimized, 1 GPU)

**Hardware:** EPYC 7742 (64 cores, 4 NUMA nodes), DDR4-2666 8-channel, 1x RTX 2000 Ada 16 GB, PCIe 4.0 x8.

Config: 10K–50K token prompts, FP8 KV cache, BF16 attention, INT8 shared_expert/dense_mlp/lm_head, 40 CPU threads, NUMA thread pinning + interleaved allocation, LGS=2, pure CPU decode.

| Model | GPUs | GPU/CPU bits | Prefill (tok/s) | TTFT @ 20K | Decode (tok/s) | ms/tok | Log |
|-------|-----:|-------------:|----------------:|:----------:|:--------------:|:------:|-----|
| Qwen3-Coder-Next | 1 | INT4/INT4 | 1,056.6 | 18.9s | 15.81 | 63.6 | [log](../logs/benchmarks/Qwen3-Coder-Next_native_1gpu_int4gpu_int4cpu_stream_lgs2.log) |
| Qwen3-Coder-Next | 1 | INT8/INT8 | 873.2 | 40.1s | 12.41 | 80.6 | [log](../logs/benchmarks/Qwen3-Coder-Next_native_1gpu_int8gpu_int8cpu_stream_lgs2.log) |
| DeepSeek-V2-Lite | 1 | INT4/INT4 | 1,476.5 | 13.6s | 20.18 | 49.7 | [log](../logs/benchmarks/DeepSeek-V2-Lite_native_1gpu_int4gpu_int4cpu_stream_lgs2.log) |
| DeepSeek-V2-Lite | 1 | INT8/INT8 | 1,316.9 | 15.2s | 17.84 | 56.2 | [log](../logs/benchmarks/DeepSeek-V2-Lite_native_1gpu_int8gpu_int8cpu_stream_lgs2.log) |

### Key improvements over previous benchmarks

- **NUMA-aware thread pinning**: rayon threads pinned round-robin across 4 NUMA nodes via sched_setaffinity. Eliminates cross-node memory traffic.
- **MPOL_INTERLEAVE**: Weight mmap pages spread across all memory controllers. 4x aggregate DRAM bandwidth.
- **MLA AVX2 kernels**: w_kc/w_vc absorption and attention vectorized with parallel head dispatch.
- **Combined effect**: QCN decode 7.89 → 15.81 tok/s (+100%), V2-Lite decode 6.22 → 20.18 tok/s (+224%).
- **Note**: Decode numbers from Feb 25 used Python timing that included cuda.synchronize() — may be slightly pessimistic. Feb 27 numbers use Rust-internal timing.

---

## Previous Benchmarks — 2026-02-22 (pre-NUMA, multi-GPU)

**Hardware:** EPYC 7742, DDR4-2666 8-channel, 3x RTX 2000 Ada 16 GB, 1 NUMA node (NPS1), 48 CPU threads.

Config: 10K token prompt, FP8 KV cache, INT8 attention/shared_expert/dense_mlp/lm_head.
Default: pure CPU MoE decode (no HCS), streamed attention with double buffering.

| Model | GPUs | GPU/CPU bits | LGS | HCS | Prefill (tok/s) | TTFT (s) | Decode (tok/s) | ms/tok | Status | Log |
|-------|-----:|-------------:|----:|-----|----------------:|---------:|---------------:|-------:|--------|-----|
| DeepSeek-V2-Lite | 1 | INT8/INT8 | 2 | ON | 1882.8 | 5.32 | 3.04 | 328.8 | PASS | [log](../logs/benchmarks/) |
| DeepSeek-V2-Lite | 2 | INT4/INT4 | 2 | ON | 1623.1 | 6.16 | 6.22 | 160.9 | PASS | [log](../logs/benchmarks/) |
| Qwen3-Coder-Next | 1 | INT8/INT8 | 2 | ON | 696.4 | 14.36 | 5.93 | 168.6 | PASS | [log](../logs/benchmarks/) |
| Qwen3-Coder-Next | 1 | INT4/INT4 | 2 | ON | 979.6 | 10.21 | 7.89 | 126.8 | PASS | [log](../logs/benchmarks/Qwen3-Coder-Next_native_1gpu_int4gpu_int4cpu.log) |
| Qwen3-Coder-Next | 1 | INT4/INT4 | 2 | OFF | 1097.4 | 18.23 | 8.12 | 123.4 | PASS | [log](../logs/benchmarks/Qwen3-Coder-Next_native_1gpu_int4gpu_int4cpu_stream_lgs2.log) |
| Qwen3-Coder-Next | 2 | INT4/INT4 | 2 | ON | 880.2 | 11.36 | 8.15 | 122.8 | FAIL* | [log](../logs/benchmarks/Qwen3-Coder-Next_native_2gpu_int4gpu_int4cpu.log) |
| Qwen3-Coder-Next | 2 | INT4/INT4 | 2 | multi | 806.8 | 12.39 | 9.14 | 109.4 | PASS | [log](../logs/benchmarks/Qwen3-Coder-Next_native_2gpu_int4gpu_int4cpu_multigpu_hcs.log) |
| Qwen3-Coder-Next | 2 | INT4/INT4 | 2 | ON | 859.6 | 11.63 | 7.21 | 138.8 | PASS | [log](../logs/benchmarks/Qwen3-Coder-Next_native_2gpu_int4gpu_int4cpu_stream_lgs2.log) |
| Qwen3-Coder-Next | 2 | INT4/INT4 | 4 | ON | 845.2 | 11.83 | 7.21 | 138.7 | PASS | [log](../logs/benchmarks/Qwen3-Coder-Next_native_2gpu_int4gpu_int4cpu_stream_lgs4.log) |
| gpt-oss-120b | 1 | INT8/INT8 | 2 | ON | 516.1 | 19.38 | 3.59 | 278.7 | PASS | [log](../logs/benchmarks/) |
| gpt-oss-120b | 2 | INT4/INT4 | 2 | ON | 825.7 | 12.11 | 5.17 | 193.6 | PASS | [log](../logs/benchmarks/) |
| Qwen3-235B-A22B | 1 | INT4/INT4 | 2 | OFF | 369.7 | 27.05 | 1.58 | 632.1 | PASS | [log](../logs/benchmarks/Qwen3-235B-A22B_native_1gpu_int4gpu_int4cpu_stream_lgs2.log) |
| Qwen3-235B-A22B | 2 | INT4/INT4 | 2 | OFF | 214.2 | 46.69 | 1.58 | 635.3 | PASS | [log](../logs/benchmarks/Qwen3-235B-A22B_native_2gpu_int4gpu_int4cpu_stream_lgs2.log) |

### Column Legend

- **LGS**: Layer Group Size — number of layers streamed through GPU at a time (double-buffered). Lower = less VRAM, more DMA rounds.
- **HCS**: Hot-Cache Strategy — ON = GPU-cached experts for decode, OFF = pure CPU decode, multi = HCS on all GPUs.

### Notes

- **Pure CPU decode** (HCS OFF) is now default. QCN pure CPU decode (7.82 tok/s) beats HCS ON (7.21 tok/s) because GPU Marlin M=1 overhead exceeds CPU AVX2 INT4 cost for QCN's tiny experts (intermediate=512).
- **Heatmap overhead fix**: Disabling heatmap collection during normal inference improved QCN decode from 7.38 to 7.82 tok/s (+6%). Heatmap accumulation called torch.unique() per MoE layer per token — unnecessary when HCS is off.
- **Qwen3-235B-A22B** now runs on 1 GPU thanks to streaming attention (94 MLA layers streamed through ~136 MB double buffers instead of 6.5 GB persistent). Previously OOM'd.
- **Qwen3-235B-A22B 1 GPU vs 2 GPU**: Decode identical (1.58 tok/s, all CPU). Prefill 73% faster on 1 GPU (369.7 vs 214.2 tok/s) — second GPU adds cross-device DMA overhead with no benefit.
- **QCN 2gpu INT4/INT4 FAIL***: Prefill and decode speeds are valid, but decode output is garbage (cross-GPU HCS expert corruption).
- **QCN 2gpu multi-HCS**: HCS experts on both GPUs (11,279 total). Decode 9.14 tok/s (slower than pure CPU 10.57 due to CPU bounce overhead).
- **QCN 2gpu stream lgs=2 vs lgs=4**: Nearly identical performance. lgs=2 slightly better for VRAM headroom.
