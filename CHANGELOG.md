# Changelog

## Unreleased

- Verified the restored Gemma HQQ4 k4v4 clean baseline after rejecting the
  HD512 q2-BC32 K/V-alias prototype. The verification gate was recorded in
  `krasis-internal/DEBUGLOG.md` before launch. Ran
  `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` with only
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and
  `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` enabled, with attribution clocks and q2
  prototype envs explicitly unset. The run passed `14/14`,
  `ALL TESTS PASSED`, and produced `5619.6` prefill, `92.43` internal decode,
  and `155.69` HTTP, confirming restored speed close to the accepted
  `5594.9/92.42` k4 marker. HCS stayed `3840/3840`, min free decode VRAM was
  `11474 MB`, and all Dynamic HCS rows had `copy_failures=0`. The log had no
  prefill attribution or q2 candidate markers. Cleanup required `./dev kill`
  for a residual benchmark server/zombie child, after which no tmux/server
  process remained and GPUs were idle.

- Implemented and rejected the env-gated HD512 q2-BC32 K/V-alias prefill
  prototype behind `KRASIS_PREFILL_HD512_Q2_BC32_KV_ALIAS=1`. The gate was
  recorded in `krasis-internal/DEBUGLOG.md` before source edits. The default
  env-off q2-BC16 path and the existing capacity-blocked
  `KRASIS_PREFILL_HD512_Q2_BC32=1` path were kept separate. `./dev build`
  passed with `20260614_1339_gemma4_hqq4_hd512_q2_bc32_kv_alias_build.log`
  (`duration_s=133`). Required k4 attribution then passed correctness `14/14`
  with `5270.0` prefill, `92.42` internal decode, `160.15` HTTP, HCS
  `3840/3840`, min free decode VRAM `11474 MB`, and zero copy failures; the
  log explicitly selected `hd512_q2_mode=bc32_kv_alias`, `hd512_tile_cols=32`,
  and `hd512_q_heads_per_block=2`. Representative `11824`-token HD512 q2
  rows were `373.1-374.4 ms/layer` with K load `93.9-94.2`, softmax
  `58.7-58.9`, V load `93.9-94.2`, and PV `97.4-97.8`. The required clean
  k4 speed run passed correctness but failed the full gate: `4817.6` prefill,
  `92.38` internal decode, `160.30` HTTP versus accepted clean k4 baseline
  `5594.9/92.42/165.46`. Reverted the alias source changes only and rebuilt
  restored source successfully with
  `20260614_1357_restore_after_hd512_q2_bc32_kv_alias_reject_build.log`
  (`duration_s=132`). No k6v6 or QCN guard was run because clean k4 speed
  failed.

- Completed a design-only inspection after the q2-BC32 capacity block. Current
  clean markers are k4 `5594.9/92.42/165.46` and k6v6
  `5243.5/65.08/119.07`; the q2-BC32 prototype correctly failed visibly on
  this RTX 5090 because it requires `106496` bytes opt-in shared memory and the
  device reports `101376`. q2-BC24 was rejected as a safe narrow follow-up even
  though its derived current-layout shared memory is `88576` bytes, because the
  current q2 WMMA loops require a 16-column tile multiple and `BC=24` would
  require padded/remainder handling. Selected one future design-only follow-up,
  not implemented here: q2-BC32 with temporal K/V shared-memory aliasing under a
  new env gate such as `KRASIS_PREFILL_HD512_Q2_BC32_KV_ALIAS=1`. The derived
  shared-memory requirement is `73728` bytes, below the `101376` byte device
  limit, by loading K for QK and then reusing the same shared tile buffer for V
  before PV. No performance code was edited and no benchmark was run.

- Implemented the planned env-gated HD512 q2-BC32 prefill prototype behind
  `KRASIS_PREFILL_HD512_Q2_BC32=1`. The default env-off HD512 q2-BC16 path
  remains unchanged. The implementation adds q2-BC32 normal/timed CUDA entry
  points, Rust symbol registration, runtime shared-memory capacity checks, and
  visible failure instead of silent fallback when the env flag is set but the
  device cannot supply the derived q2-BC32 shared memory. `./dev build` passed
  with final-source log `20260614_1316_gemma4_hqq4_hd512_q2_bc32_build.log`
  (`duration_s=128`). The required first k4 attribution run was attempted with
  the accepted dual-norm + GPU-softcap baseline plus `KRASIS_PREFILL_TIMING=1`,
  `KRASIS_PREFILL_HD512_KERNEL_CLOCKS=1`, and
  `KRASIS_PREFILL_HD512_Q2_BC32=1`, but it stopped before benchmark/correctness
  because the selected RTX 5090 reports `101376` bytes opt-in shared memory and
  q2-BC32 requires `106496` bytes. This confirmed the intended visible capacity
  gate and no hidden fallback to q2-BC16. No clean speed, k6v6, or QCN guard
  was run because k4 attribution could not start on this hardware.

- Recorded a design-only HD512 q2 prefill redesign plan before any performance
  code edit. The gate was written to `krasis-internal/DEBUGLOG.md` first.
  Source/log inspection confirmed the active long-row path is the Gemma HD512
  q2 kernel with `BR=16`, `BC=16`, `q_heads_per_block=2`, and q2 K/V sharing.
  k4 HD512 kernel clocks showed capped long-row per-layer total
  `627.0-628.4 ms`, dominated by PV `246.9-247.3 ms`, K/V tile load
  `177.7-178.0 ms`, and softmax/rescale/probability `156.9-157.2 ms`; k6v6
  attribution showed the same HD512 `custom_tiled` area at
  `1285.4-1287.2 ms` over five calls on capped `10524` rows. Selected one
  concrete future prototype plan, not an implementation:
  `KRASIS_PREFILL_HD512_Q2_BC32=1`, an opt-in q2 retile that keeps `BR=16`,
  `q_heads_per_block=2`, online-softmax math, chunk sizing, calibration, HCS,
  decode graph behavior, and non-Gemma behavior unchanged, while widening only
  the K/V tile to `BC=32` when runtime opt-in shared memory supports the
  derived requirement. If the env is set and shared memory is insufficient, the
  future prototype must fail visibly rather than silently falling back to
  q2-BC16. No performance code was edited and no benchmark was run in this
  design pass.

- Ran a focused source-free Gemma4 HQQ4/k6v6 attribution pass before any HD512
  redesign work. The gate and full tmux command were recorded in
  `krasis-internal/DEBUGLOG.md` before launch. The run used
  `tests/gemma-4-4-hqq4-k6v6-a16.conf` through `./dev test --timing` with
  accepted gates `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and
  `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`, plus existing attribution clocks
  `KRASIS_PREFILL_TIMING=1`, final, MoE-route, route-prep, MoE-W2, W2-preload,
  and GQA clocks. The heavy per-block HD512 kernel clock and rejected candidate
  envs were explicitly unset. Full log
  `20260614_1255_gemma4_hqq4_k6v6_dual_norm_softcap_attr_timing.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5537.2` prefill, `64.04` internal decode,
  `116.30` HTTP, HCS `3840/3840`, min free decode VRAM `11748 MB`, zero cold
  DMA, and `copy_failures=0`. The split showed capped `10524`-token prefill
  dominated by HD512 `custom_tiled` launch `1285.4-1287.2 ms` over 5 calls,
  plus GQA projection `157.8-157.9 ms`, FA2 `112.3 ms`, O projection
  `82.8-83.1 ms`, and MoE `257.3-257.5 ms`; decode rows showed total
  `15.43/15.89/16.28 ms/tok`, MoE expert `3.34`, GQA path
  `3.07/3.34/3.87`, route/topk `2.86`, and final segment `1.05`. Decision:
  attribution-only. The data did not expose a distinct non-rejected
  `>0.2 ms/tok` target; no source changes were made and no performance
  candidate was started.

- Ran a clean timing-off Gemma4 HQQ4/k6v6 baseline check before any broad
  HD512 redesign planning. Current k4 clean marker was `5594.9` prefill,
  `92.42` internal decode, `165.46` HTTP. Inspected available Gemma HQQ k6v6
  configs and selected `tests/gemma-4-4-hqq4-k6v6-a16.conf` for comparability
  with the accepted HQQ4 k4 marker. Only
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and
  `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` were enabled; known attribution clocks
  and rejected candidate envs were explicitly unset, `--timing` was omitted,
  and the metric gate plus full tmux command were recorded in
  `krasis-internal/DEBUGLOG.md` before launch. Full log
  `20260614_1247_gemma4_hqq4_k6v6_dual_norm_softcap_clean_speed.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5243.5` prefill, `65.08` internal decode,
  `119.07` HTTP, HCS `3840/3840`, min free decode VRAM `11748 MB`, zero cold
  DMA, and `copy_failures=0`. Startup built and loaded the Gemma Marlin expert
  cache for k6v6 before benchmark timing. No source changes were made and no
  performance candidate was started.

- Ran a clean timing-off Gemma4 HQQ4/k4v4 accepted-baseline speed check with
  only `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and
  `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1` enabled. Known attribution clocks and
  rejected candidate envs were explicitly unset, `--timing` was omitted, and
  the metric gate plus full tmux command were recorded in
  `krasis-internal/DEBUGLOG.md` before launch. Full log
  `20260614_1238_gemma4_hqq4_k4v4_dual_norm_softcap_clean_speed.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5594.9` prefill, `92.42` internal decode,
  `165.46` HTTP, HCS `3840/3840`, min free `11474 MB`, zero cold DMA, and
  `copy_failures=0`. This replaces the anomalous accepted experimental
  timing-off marker `4931.0/92.31`; startup rebuilt the Gemma Marlin expert
  cache, but the benchmark speed rows were timing-off and attribution-free. No
  source changes were made.

- Recorded a design-only inspection of the Gemma4 HQQ4/k4v4 HD512 q2 prefill
  fallback after kernel attribution showed PV accumulation `246.7-247.0 ms`,
  K/V tile load `177.7-177.9 ms`, and softmax/rescale/probability
  `156.8-157.1 ms` per capped `14780`-token HD512 layer. The design gate was
  recorded in `krasis-internal/DEBUGLOG.md` before inspection. Source review
  found the active q2 path uses `BC=16`, `q_heads_per_block=2`, and `70656`
  bytes of dynamic shared memory; existing single-head BC32/48/64 variants
  require `86016/120320/154624` bytes. No performance candidate was selected:
  forcing BC32 is mechanically narrow but not clear-upside because it gives up
  q2 K/V sharing, while the real PV/softmax improvement path requires a broader
  HD512 attention retile/redesign. No performance code, build, or benchmark was
  run for this design-only pass.

- Added timing-only Gemma4 HQQ4/k4v4 HD512 `custom_tiled` q2 kernel
  attribution under accepted decode baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`.
  The metric gate was recorded before source changes. The debug path is opt-in
  with `KRASIS_PREFILL_HD512_KERNEL_CLOCKS=1` and splits the HD512 q2 fallback
  into Q load, K/V tile load, QK score, softmax/rescale/probability write, PV
  accumulation, final write, and residual overhead without changing default
  prefill kernel selection, calibration, HCS, graph/decode behavior, chunk
  sizing, or non-Gemma behavior. Initial build
  `20260614_1219_gemma4_hqq4_k4v4_prefill_hd512_kernel_attr_build.log` failed
  with Rust `E0308` from passing an `Option<f64>` event time into the report
  helper; fixed build
  `20260614_1221_gemma4_hqq4_k4v4_prefill_hd512_kernel_attr_fix_build.log`
  passed with `duration_s=129`. Attribution run
  `20260614_1224_gemma4_hqq4_k4v4_prefill_hd512_kernel_attr_timing.log`
  passed `14/14`, `ALL TESTS PASSED`, with `3865.8` prefill, `92.17`
  internal decode, `157.09` HTTP, HCS `3840/3840`, min free `11474 MB`, zero
  cold DMA, and `copy_failures=0`. The prefill number is instrumentation-heavy
  and not a speed-regression marker. Representative `14780`-token q2 rows
  showed each HD512 layer at `627.0-627.9 ms`, split into Q load `1.2 ms`,
  K/V tile load `177.7-177.9 ms`, QK score `44.4-44.5 ms`,
  softmax/rescale/probability write `156.8-157.1 ms`, PV accumulation
  `246.7-247.0 ms`, final write `0.1 ms`, and residual `0.0 ms`. Decision:
  attribution only; no performance candidate was attempted because the split
  points at a larger HD512 attention redesign, not a distinct narrow safe
  implementation path.

- Ran source-free Gemma4 HQQ4/k4v4 prefill attribution under accepted decode
  baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`
  with `KRASIS_PREFILL_TIMING=1`. Command:
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1 KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf`.
  The run rebuilt and loaded the Gemma Marlin expert cache during startup, then
  passed `14/14`, `ALL TESTS PASSED`, with `5562.6` prefill, `92.53`
  internal decode, `160.33` HTTP, HCS `3840/3840`, min free `11474 MB`, zero
  cold DMA, and `copy_failures=0`. The low prior timing-off prefill marker
  `4931.0` was not reproduced. Benchmark prefill rows were `2860.1` at
  `1000` tokens, `5562.6` at `4999`, `4660.3` at `10000`, and
  `3796.9/3794.6/3767.9` on capped `14780/14780/14779` rows. Long-row
  attribution showed the known HD512 custom-tiled fallback launch at
  `2503.4-2505.0 ms` over 5 calls, while KV append was only `3.4-3.5 ms`
  (`~0.3 ms` wall-minus-kernel). Decision: attribution only; no new safe
  non-rejected prefill optimization candidate was selected.

- Added timing-only Gemma4 HQQ4/k4v4 MoE hidden write/post-FFN attribution
  under accepted experimental baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`.
  The new debug split widens the Gemma MoE-route clock layout to separate
  `post_ffn_norm2`, dense+MoE add, `post_ffn_norm`, residual add, optional
  layer scalar, and endpoint overhead. Initial build
  `20260614_1140_gemma4_hqq4_k4v4_moe_hidden_attr_build.log` passed, but the
  first timing run
  `20260614_1144_gemma4_hqq4_k4v4_moe_hidden_attr_timing.log` was stopped as
  invalid because the new sub-span accumulators were not reset per request,
  producing negative residuals. Fixed build
  `20260614_1150_gemma4_hqq4_k4v4_moe_hidden_attr_fix_build.log` passed with
  `duration_s=128`. Valid attribution
  `20260614_1154_gemma4_hqq4_k4v4_moe_hidden_attr_timing.log` passed `14/14`,
  `ALL TESTS PASSED`, with `5507.8` prefill, `79.59` internal decode,
  `175.39` HTTP, HCS `3840/3840`, min free `11468 MB`, zero cold DMA, and
  `copy_failures=0`. Representative rows showed hidden write/post-FFN
  `0.58-0.68 ms/tok`, `post_ffn_norm2` `0.21-0.23`, dense+MoE add
  `0.05-0.07`, `post_ffn_norm` `0.21-0.23`, residual add `0.05-0.07`,
  layer scalar `0.05-0.07`, endpoint `0.02-0.03`, and residual `~0.00`.
  Decision: attribution only. The RMSNorms are real `>0.2 ms/tok` pieces, but
  no distinct safe non-rejected implementation path was identified from this
  split.

- Added timing-only Gemma4 HQQ4/k4v4 MoE post-output attribution under
  accepted experimental baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`.
  The new debug split widens the Gemma MoE-route clock layout to separate
  shared gate, shared W13/reduce, shared W2, routed scaling, hidden
  write/post-FFN work, and endpoint overhead. Build
  `20260614_1125_gemma4_hqq4_k4v4_moe_post_attr_build.log` passed with
  `duration_s=127`. Focused attribution
  `20260614_1130_gemma4_hqq4_k4v4_moe_post_attr_timing.log` passed `14/14`,
  `ALL TESTS PASSED`, with `5293.8` prefill, `80.69` internal decode,
  `186.31` HTTP, HCS `3840/3840`, min free `11470 MB`, zero cold DMA, and
  `copy_failures=0`. Representative rows showed post-output `0.55-0.63
  ms/tok`, shared gate/W13/reduce/W2 all `0.00`, routed scale `0.02-0.03`,
  hidden write/post-FFN `0.49-0.54`, endpoint `0.04-0.06`, and routed
  weighted combine `0.06-0.08`. Decision: attribution only. Hidden
  write/post-FFN is the only newly isolated `>0.2 ms/tok` piece, but it still
  groups multiple kernels and needs a further split before there is a distinct
  safe implementation path.

- Added an attribution-only Gemma4 HQQ4/k4v4 focused MoE timing pass under
  accepted experimental baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`.
  No source change was made; the run used existing MoE-route and W2 preload
  clocks. Run `20260614_1116_gemma4_hqq4_k4v4_moe_focus_attr_timing.log`
  passed `14/14`, `ALL TESTS PASSED`, with `5005.2` prefill, `81.13`
  internal decode, `187.11` HTTP, HCS `3840/3840`, min free `11470 MB`, zero
  cold DMA, and `copy_failures=0`. Representative rows showed MoE expert
  `3.36-3.53 ms/tok`, W13 `1.08-1.12`, W13 reduce `0.05-0.07`,
  activation+W2 `1.66-1.69`, activation `1.19`, W2 compute/output `0.41`,
  load `0.31-0.35`, SiLU/multiply `0.62-0.66`, shared store `0.21-0.23`,
  weighted accumulation `0.06-0.08`, graph/launch `0.04-0.06`, and
  post-output `0.50-0.57`. Decision: attribution only. W13 direct,
  activation precompute/reuse, paired-output W2, and FP32 shared activation
  are already rejected; weighted accumulation/launch are too small; post-output
  needs a separate split before it is a safe candidate.

- Added an attribution-only Gemma4 HQQ4/k4v4 route/final focused timing pass
  under accepted experimental baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`.
  No source change was made; the run used existing final, MoE-route, and
  route-prep clocks to inspect route-prep dense/overhead and final LM-head
  BF16 cuBLAS after moving off non-active GQA. Run
  `20260614_1108_gemma4_hqq4_k4v4_route_final_attr_timing.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5618.9` prefill, `83.63` internal
  decode, `192.76` HTTP, HCS `3840/3840`, min free `11466 MB`, zero cold DMA,
  and `copy_failures=0`. Representative rows showed route-prep dense pre
  `0.25-0.26`, gate `0.34-0.35`, up `0.34-0.35`, activation `0.06-0.07`,
  down `0.37-0.38`, dense post norms `0.42-0.43`, router norm `0.02-0.03`,
  remaining overhead `0.22-0.25`, final LM-head BF16 cuBLAS `0.92`, D2H
  logits `0.13`, graph residual `0.12`, and final segment/sync `1.05 ms/tok`.
  Decision: attribution only. Route-prep does not expose a clean non-rejected
  `>0.2 ms/tok` candidate, and final LM-head optimization remains blocked by
  the previously rejected Marlin INT8 path.

- Added timing-only focused attribution for Gemma4 HQQ4/k4v4 non-active GQA
  under accepted experimental baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`.
  The new debug split expands existing non-active GQA markers into projection,
  Q/K/V norm, RoPE, KV-cache write, attention/reduce, BF16/gated conversion,
  O-input prep, O projection, endpoint/gap, and the existing HD256
  score/weight-V/final-output clocks. Build
  `20260614_1052_gemma4_hqq4_k4v4_gqa_other_detail_attr_build.log` passed
  with `duration_s=128`. Focused attribution
  `20260614_1055_gemma4_hqq4_k4v4_gqa_other_detail_attr_timing.log` passed
  `14/14`, `ALL TESTS PASSED`, with `4950.5` prefill, `77.07` internal
  decode, `156.54` HTTP, HCS `3840/3840`, min free `11462 MB`, zero cold
  DMA, and `copy_failures=0`. Representative 49/99/249/511 rows showed
  projection `0.92/0.95/0.95/0.92 ms/tok`, Q/K/V norm
  `0.10/0.12/0.12/0.10`, RoPE `0.04/0.06/0.05/0.04`, KV write
  `0.11/0.13/0.12/0.11`, attention `0.68/0.85/1.30/1.88`, O projection
  `0.44/0.46/0.45/0.44`, and HD256 weight/V `0.42/0.56/0.89/1.33`.
  Decision: attribution only. Newly split norm/RoPE/KV pieces are below the
  `>0.2 ms/tok` useful-upside threshold; larger projection/attention areas map
  to already rejected or not-yet-safe GQA work, so no optimization candidate
  was attempted.

- Rejected the decode-only Gemma4 HQQ4/k4v4 route-prep fused pre-norm copy
  candidate. Scope used accepted experimental baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`
  and targeted the remaining non-rejected route-prep dense pre-norm bucket
  from fresh post-softcap attribution
  `20260614_1018_gemma4_hqq4_k4v4_post_softcap_fresh_attr_timing.log`
  (`0.23-0.26 ms/tok`, fresh summary `5597.4` prefill, `77.58` internal
  decode, `153.06` HTTP, HCS `3840/3840`, min free `11464 MB`, zero cold DMA,
  `copy_failures=0`). Candidate env
  `KRASIS_DECODE_ROUTE_PREP_FUSED_PRE_NORM=1` replaced only the Gemma graph
  route-prep residual copy plus pre-FFN RMSNorm pair with existing
  `fused_add_rmsnorm(..., first_layer=1)`, gated by both accepted env flags,
  Gemma k4v4 graph scope, nonzero `pre_ffn_norm_ptr`, and runtime
  hidden-size/shared-memory capacity. Build
  `20260614_1037_gemma4_hqq4_k4v4_route_fused_pre_norm_build.log` passed with
  `duration_s=129`. Timing gate
  `20260614_1040_gemma4_hqq4_k4v4_route_fused_pre_norm_timing.log` passed
  correctness (`14/14`, `ALL TESTS PASSED`) with `5611.7` prefill, `77.80`
  internal decode, `157.72` HTTP, HCS `3840/3840`, min free `11462 MB`, and
  `copy_failures=0`, but failed the performance gate: dense pre-norm only
  moved to mostly `0.23 ms/tok` with occasional `0.21-0.22`, route
  prep/overhead stayed mostly `2.60 ms/tok` and regressed on some rows, and
  the `77.58 -> 77.80 tok/s` timing delta was too small/noisy to accept.
  Reverted only this candidate; restore build
  `20260614_1048_restore_after_route_fused_pre_norm_reject_build.log` passed
  with `duration_s=127`, and candidate env/source symbols are absent. No
  timing-off Gemma speed or QCN guard was run because the timing gate failed.

- Rejected the decode-only Gemma4 HQQ4/k4v4 final LM-head Marlin INT8
  candidate. Scope used accepted experimental baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`
  and first ran fresh post-softcap attribution
  `20260614_1018_gemma4_hqq4_k4v4_post_softcap_fresh_attr_timing.log`, which
  passed `14/14` with `5597.4` prefill, `77.58` internal decode, `153.06`
  HTTP, HCS `3840/3840`, min free `11464 MB`, zero cold DMA, and
  `copy_failures=0`. The remaining final LM-head bucket was about
  `0.93 ms/tok`, D2H logits were `0.15-0.16`, host softcap was `0.00`, and
  weighted expert accumulation remained below threshold. The candidate was
  env-gated by `KRASIS_DECODE_FINAL_LM_HEAD_MARLIN_INT8=1`, Gemma-final only,
  and dimension-gated from the source INT8 LM-head tuple plus Marlin
  compatibility checks (`rows % 64 == 0`, `cols % 16 == 0`,
  `cols % group_size == 0`). Build
  `20260614_1029_gemma4_hqq4_k4v4_final_lm_head_marlin_int8_build.log`
  passed with `duration_s=7`, but timing gate
  `20260614_1030_gemma4_hqq4_k4v4_final_lm_head_marlin_int8_timing.log`
  failed before benchmark completion: LM head regressed to `9.40-9.52
  ms/tok`, then graph replay failed with `CUDA_ERROR_ILLEGAL_ADDRESS` at final
  sync and the boundary sync reported a fatal CUDA context error. HCS was
  still clean before failure (`3840/3840`, representative `240/0`,
  `copy_failures=0`). Reverted only this candidate; restore build
  `20260614_1033_restore_after_final_lm_head_marlin_int8_reject_build.log`
  passed with `duration_s=8`, and candidate env/source symbols are absent. No
  timing-off Gemma speed or QCN guard was run because the timing gate failed.

- Accepted an env-gated Gemma4 HQQ4/k4v4 GPU final-logit softcap path. A
  focused final/sync attribution pass was run first under accepted experimental
  baseline `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1`. Initial attribution build
  `20260614_0948_gemma4_hqq4_k4v4_final_attr_build.log` failed with
  `error[E0425]` for an out-of-scope `avg_sync_final` report variable; build2
  `20260614_0952_gemma4_hqq4_k4v4_final_attr_build2.log` passed with
  `duration_s=127`. Focused attribution
  `20260614_0955_gemma4_hqq4_k4v4_final_attr_timing.log` passed correctness
  (`14/14`, `ALL TESTS PASSED`) with `5626.1` prefill, `58.90` internal
  decode, `119.52` HTTP, HCS `3840/3840`, min free `11464 MB`, zero cold DMA,
  and `copy_failures=0`; it split final graph work into final RMSNorm
  `~0.01 ms/tok`, LM head `~0.92`, graph residual `~0.12`, D2H logits
  `0.12-0.13`, and host final-logit softcap typically `3.86-4.32 ms/tok`
  with one `4.84` row. Added `KRASIS_DECODE_FINAL_SOFTCAP_GPU=1`, gated to
  Gemma graph final segments with finite positive `final_logit_softcap` and
  runtime `vocab_size`, applying the same `tanh(logit / softcap) * softcap`
  formula to `d_logits` before D2H and skipping CPU softcap only when the GPU
  path captured. Candidate build
  `20260614_0952_gemma4_hqq4_k4v4_final_softcap_gpu_build.log` passed with
  `duration_s=136`. Timing gate
  `20260614_0958_gemma4_hqq4_k4v4_final_softcap_gpu_timing.log` passed
  `14/14`, moved host softcap to `0.00 ms/tok`, and improved timing-enabled
  internal decode from `58.90` to `77.84 tok/s` with HCS still clean. Timing-
  off Gemma run `20260614_1004_gemma4_hqq4_k4v4_final_softcap_gpu_speed.log`
  passed `14/14` with `4931.0` prefill, `92.31` internal decode, `160.16`
  HTTP, HCS `3840/3840`, min free `11474 MB`, and `copy_failures=0`; prefill
  was lower in that run despite no prefill-path changes and is recorded as a
  follow-up data point. QCN guard
  `20260614_1011_qcn_speed_test_after_final_softcap_gpu.log` passed via
  `./dev speed-test` with `6685.1` prefill, `90.07` internal decode,
  `198.82` HTTP, HCS `15957/24576`, min free `896 MB`, and `copy_failures=0`.

- Rejected the decode-only Gemma4 HQQ4/k4v4 MoE W2 FP32 shared-activation
  candidate. Scope stayed on the accepted experimental baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and targeted the fresh measured MoE
  activation/W2 bucket: fused activation+W2 `1.67-1.69 ms/tok`, with preload
  detail load `~0.34`, SiLU/multiply `~0.64`, and shared store
  `~0.21 ms/tok`; weighted expert accumulation stayed below threshold at
  `0.06-0.08 ms/tok`. The candidate was env-gated by
  `KRASIS_DECODE_MOE_W2_F32_ACT=1`, kept one output tile per block, did not
  retry paired-output W2, and runtime-gated shared-memory use from
  `topk/intermediate/expert_hs` dimensions and device capacity. Initial build
  `20260614_0909_gemma4_hqq4_k4v4_moe_w2_f32_act_build.log` passed with
  `duration_s=137`, but the first timing attempt failed before serving because
  the new PTX symbols were not added to `KERNEL_NAMES`. After fixing that
  integration issue, build2
  `20260614_0917_gemma4_hqq4_k4v4_moe_w2_f32_act_build2.log` passed with
  `duration_s=129`. Valid timing gate
  `20260614_0920_gemma4_hqq4_k4v4_moe_w2_f32_act_timing2.log` passed
  correctness (`14/14`, `ALL TESTS PASSED`) but failed the full metric gate:
  `4683.2` prefill, `58.41` internal decode, and `99.72` HTTP versus fresh
  broad-clock baseline `4731.8/59.29/119.22`. HCS stayed clean at `3840/3840`,
  min free `11464 MB`, representative `240/0`, zero cold DMA, and
  `copy_failures=0`. The target W2 bucket improved locally
  (`1.54/1.54/1.54/1.53 ms/tok`, internal `1.47-1.48`), but total internal
  decode regressed, so the candidate was reverted. Restore build
  `20260614_0927_restore_after_moe_w2_f32_act_reject_build.log` passed with
  `duration_s=135`; no timing-off Gemma speed or QCN guard was run.

- Rejected the decode-only Gemma4 HQQ4/k4v4 GQA fused-QKV split-elision
  candidate. First ran a fresh broad timing attribution with accepted
  experimental baseline `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1`:
  `20260614_0842_gemma4_hqq4_k4v4_dual_norm_fresh_attr_timing.log` passed
  correctness (`14/14`, `ALL TESTS PASSED`) with `4731.8` prefill, `59.29`
  internal decode, `119.22` HTTP, HCS `3840/3840`, min free `11464 MB`, zero
  cold DMA, and `copy_failures=0`. Weighted expert accumulation remained only
  `0.06-0.08 ms/tok`, so no weighted-add candidate was attempted under the
  requested `>0.2 ms/tok` upside rule. Excluding paired W2, W13 direct, dense
  gate+up fusion, dense-down custom GEMV, and rejected GQA tile-cap,
  score-cache, and HD256-specialization variants, the chosen candidate targeted
  non-active GQA projection by env-gating fused-QKV K/V split-copy elision under
  `KRASIS_DECODE_GQA_FUSED_QKV_SPLIT_ELIDE=1`. Candidate build
  `20260614_0852_gemma4_hqq4_k4v4_gqa_fused_qkv_elide_build.log` passed with
  `duration_s=127`. Timing gate
  `20260614_0855_gemma4_hqq4_k4v4_gqa_fused_qkv_elide_timing.log` passed
  correctness (`14/14`, `ALL TESTS PASSED`) but failed the metric gate:
  `5605.5` prefill, `58.56` internal decode, and `110.82` HTTP versus fresh
  broad-clock baseline `4731.8/59.29/119.22`. HCS stayed clean at `3840/3840`,
  min free `11464 MB`, representative `240/0`, zero cold DMA, and
  `copy_failures=0`. The target non-active GQA projection did not improve
  (`0.95/0.95/0.95/0.93 ms/tok` versus baseline
  `0.95/0.93/0.95/0.95`), non-active endpoint regressed
  (`2.44/2.59/3.04/3.56` versus `2.44/2.53/2.96/3.36`), and total internal
  decode regressed. Reverted only this candidate; restore build
  `20260614_0902_restore_after_gqa_fused_qkv_elide_reject_build.log` passed
  with `duration_s=127`, and the candidate env/symbols are absent. No
  timing-off Gemma speed or QCN guard was run.

- Rejected the decode-only Gemma4 HQQ4/k4v4 MoE W13 direct-BF16 candidate.
  Scope used the accepted experimental baseline
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` and targeted the measured W13 bucket
  in Gemma graph MoE. The candidate was env-gated by
  `KRASIS_DECODE_MOE_W13_DIRECT=1`, runtime-gated to Gemma4 k4v4 graph MoE
  with INT4 gated experts and `w13_ksplits_batched == 1`, and attempted to
  write W13 output directly as BF16 instead of launching the one-split
  `reduce_ksplits_bf16_batched` pass. It did not change calibration, HCS
  policy, graph capture structure, prefill, final segment behavior, or
  non-Gemma behavior, and avoided paired-W2, dense gate+up fusion, dense-down
  custom GEMV, GQA tile-cap, score-cache, and HD256 specialization. Candidate
  build `20260614_0827_gemma4_hqq4_k4v4_moe_w13_direct_build.log` passed with
  `duration_s=136`. Timing gate
  `20260614_0830_gemma4_hqq4_k4v4_moe_w13_direct_timing.log` used
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_MOE_W13_DIRECT=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed correctness (`14/14`, `ALL TESTS PASSED`) but failed the metric
  gate: `5626.1` prefill, `62.16` internal decode, and `112.01` HTTP versus
  accepted dual-norm timing baseline `5628.2/64.75/112.01`. HCS stayed clean
  at `3840/3840`, min free `11466 MB`, representative `240/0`, zero cold DMA
  and `copy_failures=0`. W13+reduce stayed essentially unchanged at about
  `1.15-1.18 ms/tok`, and total internal decode regressed, so the candidate
  was reverted. Restore build
  `20260614_0836_restore_after_moe_w13_direct_reject_build.log` passed with
  `duration_s=136`; no timing-off Gemma speed or QCN guard was run.

- Rejected the decode-only Gemma4 HQQ4/k4v4 route-prep dense-down custom BF16
  GEMV candidate. The candidate was env-gated by
  `KRASIS_DECODE_ROUTE_PREP_DENSE_DOWN_CUSTOM=1` on top of the accepted
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` baseline and was runtime-gated to the
  Gemma graph route-prep dense-down BF16 weight dimensions. It avoided paired
  W2, dense gate+up fusion, GQA tile-cap, score-cache, and HD256 specialization
  candidates, and did not change calibration, HCS policy, graph capture,
  prefill, final segment behavior, or non-Gemma behavior. Candidate build
  `20260614_0806_gemma4_hqq4_k4v4_route_dense_down_custom_build.log` passed
  with `duration_s=137`. Timing gate
  `20260614_0812_gemma4_hqq4_k4v4_route_dense_down_custom_timing.log` used
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_ROUTE_PREP_DENSE_DOWN_CUSTOM=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed correctness (`14/14`, `ALL TESTS PASSED`) but failed the full
  metric gate: `5622.1` prefill, `62.12` internal decode, and `112.43` HTTP
  versus accepted dual-norm timing baseline `5628.2/64.75/112.01`. HCS stayed
  clean at `3840/3840`, min free `11466 MB`, representative `240/0`, zero cold
  DMA and `copy_failures=0`. Dense-down improved from about
  `0.36-0.37 ms/tok` to `0.29-0.31 ms/tok`, but route prep did not improve
  enough and total internal decode regressed, so the candidate was reverted.
  Restore build `20260614_0830_restore_after_dense_down_custom_reject_build.log`
  passed with `duration_s=136`; no timing-off Gemma speed or QCN guard was run.

- Rejected the decode-only Gemma4 HQQ4/k4v4 HD256 GQA specialized
  single-kernel candidate. The candidate was env-gated by
  `KRASIS_DECODE_GQA_HD256_SPECIALIZED=1` on top of the accepted
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1` baseline, and was runtime-gated to
  Gemma graph route GQA with k4v4, `head_dim == 256`, sliding/gated GQA, and
  active shared-memory capacity. It avoided the rejected tile-cap and
  score-cache GQA variants and did not change calibration, HCS policy, graph
  capture, prefill, final segment behavior, or non-Gemma paths. Candidate
  build `20260614_0748_gemma4_hqq4_k4v4_hd256_spec_build.log` passed with
  `duration_s=138`. Timing gate
  `20260614_0752_gemma4_hqq4_k4v4_hd256_spec_timing.log` used
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_GQA_HD256_SPECIALIZED=1 KRASIS_DECODE_GQA_HD256_ATTN_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed correctness (`14/14`, `ALL TESTS PASSED`) but failed performance:
  `5616.7` prefill, `61.87` internal decode, and `110.76` HTTP versus the
  accepted dual-norm timing baseline `5628.2/64.75/112.01`. HCS stayed clean
  at `3840/3840`, min free `11464 MB`, representative `240/0`, zero cold DMA
  and `copy_failures=0`. The measured GQA rows showed no consistent
  improvement (`3.22/3.33/3.97 ms/tok` versus baseline `2.97/3.35/3.89`), and
  total decode regressed. Reverted only this candidate; restore build
  `20260614_0807_restore_after_hd256_spec_reject_build.log` passed with
  `duration_s=136`. No timing-off speed or QCN guard was run.

- Added an env-gated decode-only Gemma4 HQQ4/k4v4 route-prep dual residual
  RMSNorm path under `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1`. Inspection showed
  two adjacent residual RMSNorm passes in the Gemma route-prep graph path:
  `pre_ffn_norm2` writes the BF16 scratch path and router-input RMSNorm+scale
  writes `d_hidden`, both reading the same `d_residual` with the same RMS
  denominator. The candidate fuses only those two norms and avoids the
  previously failed dense gate+up fusion path and all W2 paired-tile work. It
  does not change calibration, HCS policy, graph capture, prefill, final
  segment behavior, or QCN/non-Gemma paths. Build
  `20260614_0720_gemma4_hqq4_k4v4_route_dual_norm_build.log` passed with
  `duration_s=135`. Timing gate
  `20260614_0724_gemma4_hqq4_k4v4_route_dual_norm_timing.log` used
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed `14/14`, `ALL TESTS PASSED`, with `5628.2` prefill, `64.75`
  internal decode, `112.01` HTTP, HCS `3840/3840`, min free `11466 MB`, graph
  HCS `240/0`, and zero cold DMA/DMA calls or copy failures. The target
  duplicated residual-norm bucket improved from baseline dense-post+router
  `0.65/0.65/0.65/0.60 ms/tok` to `0.46/0.43/0.46/0.46`; route prep was
  `2.63/2.41/2.63/2.63` versus baseline `2.82/2.82/2.82/2.53`. Timing-off
  Gemma validation
  `20260614_0737_gemma4_hqq4_k4v4_route_dual_norm_speed.log` passed `14/14`
  with `5626.1` prefill, `66.96` internal decode, `120.00` round trip, HCS
  `3840/3840`, min free `11474 MB`, and `copy_failures=0`. QCN guard
  `KRASIS_DECODE_ROUTE_PREP_DUAL_NORM=1 ./dev speed-test` completed in
  `20260614_0733_qcn_speed_test_after_route_dual_norm.log` with `6452.1`
  prefill, `87.61` internal decode, `149.48` round trip, HCS `15957/24576`,
  min free `928 MB`, and `copy_failures=0`, confirming the Gemma-only gate did
  not route QCN through the new path.

- Rejected the decode-only Gemma4 HQQ4/k4v4 INT4 fused-W2 paired-output-tile
  candidate. Candidate build
  `20260614_0654_gemma4_hqq4_k4v4_w2_pair2_build.log` passed with
  `duration_s=137`. Timing gate
  `20260614_0700_gemma4_hqq4_k4v4_w2_pair2_timing.log` used
  `KRASIS_DECODE_MOE_W2_PAIRED_TILES=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and failed correctness: `13/14`, `multi_turn_t5_all` missing `Biscuit`,
  `1 TEST(S) FAILED`. Benchmark summary before failure was `5454.8 tok/s`
  prefill, `59.12 tok/s` internal decode, and `97.46 tok/s` round trip. HCS
  stayed clean with `3840/3840`, representative `240/0`, zero cold DMA/DMA
  calls, and `copy_failures=0`. The paired path was active
  (`paired_tiles:on`, `physical_tiles/seg:88.0`, `blocks/seg:704.0`) but
  regressed target timing: fused activation+W2 became
  `2.01/2.01/2.01/1.99 ms/tok` versus accepted attribution baseline
  `1.69/1.69/1.69/1.67`, internal W2 total became about `1.94-1.95` versus
  `1.63`, and SiLU/multiply rose from `0.65` to about `1.18`. Reverted only
  this candidate; restore build
  `20260614_0710_restore_after_w2_pair2_reject_build.log` passed with
  `duration_s=135`, and no paired-tile symbols or env gate remain. No
  timing-off `./dev speed-test` or QCN guard was run.

- Added diagnostic-only Gemma4 HQQ4/k4v4 fused W2 activation/preload
  sub-attribution under `KRASIS_DECODE_MOE_W2_PRELOAD_CLOCKS=1`, layered on the
  existing `KRASIS_DECODE_MOE_W2_CLOCKS=1` and MoE route graph timing gate.
  Two initial timing attempts were discarded: `20260613_231855...timing.log`
  used per-iteration clocks that perturbed activation wall time, and
  `20260613_232606...timing2.log` exposed a four-slot/eight-slot clock-base
  stride bug. After fixing the stride, build
  `20260613_233023_gemma4_hqq4_k4v4_moe_w2_preload_attr_build3.log` passed
  with `duration_s=127`, and valid timing run
  `20260613_233253_gemma4_hqq4_k4v4_moe_w2_preload_attr_timing3.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5602.3 tok/s` prefill, `59.89 tok/s`
  internal decode, `108.08 tok/s` HTTP, HCS `3840/3840`, min free
  `11470 MB`, graph HCS `240/0`, and zero cold DMA/DMA calls or copy failures.
  Representative 49/99/249/511 rows showed fused activation+W2
  `1.69/1.69/1.69/1.67 ms/tok`, activation `1.20` ms/tok, W2 prep `0.02`,
  W2 GEMV/output `0.41`, and activation detail load `0.33`, SiLU/multiply
  `0.65`, shared store `0.21`, sync `0.04`, residual `0.00`, repeated block
  work about `53.9 ms/tok`, and `1408.0` blocks/segment. Exactly one next
  candidate is selected: a decode-only Gemma INT4 fused-W2 paired-output-tile
  block variant for graph MoE, avoiding the rejected global activation
  precompute/reuse path.

- Rejected the decode-only Gemma4 HQQ4/k4v4 INT4 expert activation
  precompute/reuse candidate. Inspection confirmed the premise from the actual
  `fused_silu_w2_batched` kernel: launch shape is
  `(ceil(expert_hs / 16), 1, topk)`, and every `(output tile, expert)` block
  recomputes `SiLU(gate) * up` for the full `moe_intermediate_size` into
  block-local shared memory. Candidate build
  `20260613_225526_gemma4_hqq4_k4v4_act_precompute_build.log` passed with
  `duration_s=137`. Timing gate
  `20260613_225804_gemma4_hqq4_k4v4_act_precompute_timing.log` used
  `KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed `14/14`, `ALL TESTS PASSED`, with `5165.6 tok/s` prefill,
  `64.24 tok/s` internal decode, `109.83 tok/s` HTTP, HCS `3840/3840`, min
  free `11470 MB`, graph HCS `240/0`, and zero cold DMA/DMA calls or copy
  failures. The target bucket did not improve: activation dropped to
  `0.01 ms/tok`, but W2 prep rose to `1.07-1.08 ms/tok`, leaving
  activation+W2 at `1.48/1.51/1.49/1.50 ms/tok` versus the accepted marker
  `1.48/1.48/1.48/1.46`. Reverted only this candidate; restore build
  `20260613_230519_restore_after_act_precompute_reject_build.log` passed with
  `duration_s=135`, and no activation-precompute symbols remain. No timing-off
  Gemma speed or QCN guard was run.

- Added diagnostic-only Gemma4 HQQ4/k4v4 MoE activation/W2 attribution for the
  next proven decode bucket after rejecting dense gate+up. The instrumentation
  is gated by `KRASIS_DECODE_MOE_W2_CLOCKS=1`, graph timing, k4v4 captured
  graphs containing `Gemma4MoE`, target `GRAPH_SEG_ROUTE_GQA`, and the runtime
  gated INT4 expert path; normal decode stays on `fused_silu_w2_batched`.
  Build attempt `20260613_223530_gemma4_hqq4_k4v4_moe_w2_attr_build.log`
  failed with Rust `E0277` because the diagnostic timed kernel exceeded
  cudarc's tuple launch arity; the launch was corrected to use the raw
  pointer-vector form. Build
  `20260613_223732_gemma4_hqq4_k4v4_moe_w2_attr_build2.log` passed with
  `duration_s=128`. Timing run
  `20260613_224001_gemma4_hqq4_k4v4_moe_w2_attr_timing.log` used
  `KRASIS_DECODE_MOE_W2_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed `14/14`, `ALL TESTS PASSED`, with `4915.2 tok/s` prefill,
  `62.16 tok/s` internal decode, `110.99 tok/s` HTTP, HCS `3840/3840`, min
  free `11470 MB`, graph HCS `240/0`, zero cold DMA/DMA calls, and dynamic-HCS
  `promotions=0 evictions=0 copy_failures=0`. Representative 49/99/249/511
  rows showed activation `1.05/1.06/1.06/1.05 ms/tok`, W2 prep
  `0.02/0.02/0.02/0.02`, W2 GEMV/output `0.34/0.34/0.34/0.34`, weighted
  accumulation `0.08/0.08/0.08/0.06`, and graph/launch overhead
  `0.06/0.06/0.06/0.05`. Exactly one next candidate is selected:
  decode-only Gemma INT4 expert activation precompute/reuse for graph MoE.

- Rejected the decode-only Gemma4 HQQ4/k4v4 route-prep dense gate+up
  dual-projection candidate. Inspection confirmed the current path uses BF16
  dense MLP tensors registered with `dtype=0`, two separate cuBLAS BF16 GEMM
  launches sharing the same `d_hidden` input, and adjacent `[gate | up]` output
  slices in `d_expert_gate_up`. Candidate build attempt
  `20260613_221418_gemma4_hqq4_k4v4_dense_gate_up_build.log` failed with Rust
  `E0133`; the wrapper was fixed and build
  `20260613_221555_gemma4_hqq4_k4v4_dense_gate_up_build2.log` passed with
  `duration_s=127`. Timing gate
  `20260613_221837_gemma4_hqq4_k4v4_dense_gate_up_timing.log` used
  `KRASIS_DECODE_ROUTE_PREP_CLOCKS=1 KRASIS_DECODE_MOE_ROUTE_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  but failed correctness during warmup graph replay with
  `sync routing[1]: CUDA_ERROR_ILLEGAL_ADDRESS`, followed by fatal
  decode/prefill boundary sync. Partial diagnostics showed HCS `3840/3840`,
  `240/0` hit/miss, and zero cold DMA/DMA calls, but no full benchmark summary
  was valid. Reverted only this candidate; restore build
  `20260613_222310_restore_after_dense_gate_up_reject_build.log` passed with
  `duration_s=127`, and no grouped dense gate+up symbols remain. No timing-off
  Gemma speed or QCN guard was run.

- Added diagnostic-only Gemma4 HQQ4/k4v4 route-prep sub-split attribution for
  the largest unresolved non-GQA decode bucket. The instrumentation is gated by
  `KRASIS_DECODE_ROUTE_PREP_CLOCKS=1`, graph timing, k4v4 captured graphs with
  `Gemma4MoE`, and target `GRAPH_SEG_ROUTE_GQA` layers; it also forces the
  parent MoE/route clocks so the denominator is recorded in the same run. Build
  `20260613_215750_gemma4_hqq4_k4v4_route_prep_attr_build.log` passed in
  `127s`. Timing run
  `20260613_220014_gemma4_hqq4_k4v4_route_prep_attr_timing.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5611.2 tok/s` prefill, `63.87 tok/s`
  internal decode, `109.14 tok/s` HTTP, HCS `3840/3840`, min free
  `11464 MB`, graph HCS `240/0`, zero cold DMA/DMA calls, and dynamic-HCS
  `promotions=0 evictions=0 copy_failures=0`. Representative 49/99/249/511
  rows showed route prep/overhead `2.82/2.82/2.82/2.53 ms/tok`, pre-GQA norm
  `0.26/0.26/0.26/0.23`, post-GQA norm/add `0.26/0.26/0.26/0.24`, dense
  pre-norm `0.26/0.26/0.26/0.23`, dense gate `0.35/0.35/0.35/0.33`, dense up
  `0.35/0.35/0.35/0.33`, dense activation `0.06/0.06/0.06/0.05`, dense down
  `0.38/0.38/0.37/0.36`, dense post norms `0.42/0.42/0.42/0.39`, router-input
  norm `0.23/0.23/0.23/0.21`, and remaining debug/graph overhead
  `0.25/0.25/0.25/0.17`. Exactly one next candidate is selected:
  decode-only Gemma dense gate+up dual-projection candidate for the route-prep
  graph path.

- Added diagnostic-only Gemma4 HQQ4/k4v4 MoE and route attribution for the
  next large decode buckets after rejecting HD256 GQA micro-optimizations. The
  instrumentation is timing/debug gated behind
  `KRASIS_DECODE_MOE_ROUTE_CLOCKS=1` and limited to k4v4
  `GRAPH_SEG_ROUTE_GQA` graph segments with `Gemma4MoE` layers. Build
  `20260613_214141_gemma4_hqq4_k4v4_moe_route_attr_build.log` passed in
  `125s`. Timing run
  `20260613_214409_gemma4_hqq4_k4v4_moe_route_attr_timing.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5598.9 tok/s` prefill,
  `62.57 tok/s` internal decode, `114.27 tok/s` HTTP, HCS `3840/3840`, min
  free `11470 MB`, graph HCS `240/0`, zero cold DMA/DMA calls, and dynamic-HCS
  `promotions=0 evictions=0 copy_failures=0`. Representative 49/99/249/511
  rows showed MoE expert `3.49/3.49/3.49/3.41 ms/tok`, GQA path
  `3.09/3.35/3.94/4.63`, route/top-k `3.09/3.09/3.09/2.96`, MoE W13
  `1.11/1.11/1.11/1.09`, activation+W2 `1.66/1.66/1.66/1.65`, route logits
  `0.15/0.15/0.15/0.14`, top-k `0.31/0.31/0.31/0.30`, scale/classify
  `0.13/0.13/0.13/0.12`, and route prep/overhead
  `2.49/2.49/2.49/2.39`. Exactly one next candidate is selected:
  decode-only Gemma route-prep sub-split attribution before optimization.

- Rejected the decode-only Gemma4 HQQ4/k4v4 sliding HD256 shared-score/cache
  candidate. The candidate was gated to CUDA graph GQA route decode only:
  `GRAPH_SEG_ROUTE_GQA`, `layer_idx == range_end`, k4v4, sliding
  `head_dim == 256`, gated GQA, and shared-score cache memory derived from
  `graph.kv_cache_len_for_layer(layer_idx)` plus active GPU shared-memory
  capacity. Candidate build
  `20260613_211248_gemma4_hqq4_k4v4_hd256_score_cache_build.log` passed in
  `135s`. Initial timing run
  `20260613_211550_gemma4_hqq4_k4v4_hd256_score_cache_timing.log` passed
  `14/14`; gate wiring was corrected so the candidate kernel opt-in updated
  the runtime capacity used by the gate. Corrected timing gate
  `20260613_212310_gemma4_hqq4_k4v4_hd256_score_cache_timing2.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5617.2 tok/s` prefill, `63.46 tok/s`
  internal decode, `114.75 tok/s` HTTP, HCS `3840/3840`, min free
  `11472 MB`, graph HCS `240/0`, zero cold DMA/DMA calls, and dynamic-HCS
  `promotions=0 evictions=0 copy_failures=0`. Representative 49/99/249/511
  rows showed sliding HD256 attention graph `0.70/0.87/1.28/1.89 ms/tok`,
  internal `0.64/0.81/1.24/1.84`, score `0.18/0.21/0.32/0.48`, weight+V
  `0.42/0.57/0.88/1.33`, and final `0.04/0.04/0.04/0.04`. Because
  weight+V did not improve versus the accepted marker, no timing-off Gemma
  speed or QCN guard was run. Reverted only the score-cache candidate and
  restore build `20260613_212911_restore_after_hd256_score_cache_reject_build.log`
  passed in `135s`; no score-cache symbols remain.

- Added diagnostic-only Gemma4 HQQ4/k4v4 sliding HD256 single-kernel attention
  attribution for the non-active attention/reduce bucket. The instrumentation
  is timing/debug gated behind `KRASIS_DECODE_GQA_HD256_ATTN_CLOCKS=1`, run
  with `KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1` and
  `KRASIS_DECODE_GQA_OTHER_CLOCKS=1`, and changes no prefill, HCS policy,
  final segment behavior, QCN/non-HD512 speed path, or output math. Corrected
  build `20260613_205442_gemma4_hqq4_k4v4_hd256_attn_attr_build5.log` passed
  in `126s`. Corrected timing run
  `20260613_205709_gemma4_hqq4_k4v4_hd256_attn_attr_timing4.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5605.5 tok/s` prefill, `63.80 tok/s`
  internal decode, `115.13 tok/s` HTTP, HCS `3840/3840`, min free
  `11472 MB`, clean graph HCS (`240/0` hit/miss, zero cold DMA/DMA calls), and
  dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. Representative
  49/99/249/511 rows showed sliding HD256 attention graph
  `0.70/0.87/1.28/1.89 ms/tok`, internal kernel
  `0.64/0.81/1.24/1.84`, graph overhead `0.06/0.06/0.04/0.05`, score/max
  `0.18/0.21/0.32/0.48`, weight+V accumulation `0.42/0.57/0.88/1.33`, and
  final reduce/output `0.04/0.04/0.04/0.04`. Runtime tiles averaged
  `1.00/1.00/1.22/1.63` against allocated graph max `59.00`. The actual
  accepted hot path is `gqa_attention_k4v4_single_g`; the rejected tile-cap
  branch targeted the live non-active HD256 route layers but replaced an
  already single-kernel path with tiled graph work. Exactly one next candidate
  is selected: decode-only HD256 k4v4 single-kernel shared-score/cache variant
  for sliding GQA, gated by runtime/config max attention length and active-GPU
  shared-memory capacity.

- Rejected the decode-only Gemma4 HQQ4/k4v4 sliding HD256 graph attention
  tile-cap candidate. The candidate was tightly gated to k4v4 graph decode,
  route-layer GQA, sliding `head_dim == 256`, allocated tiled buffers, and a
  per-layer `tile_cap > 1` derived from
  `ceil(kv_cache_len_for_layer(layer_idx) / graph.gqa_tile_size)` rather than
  a model hardcode. Build
  `20260613_201050_gemma4_hqq4_k4v4_hd256_tilecap_build.log` passed in
  `125s`. Timing gate
  `20260613_201352_gemma4_hqq4_k4v4_hd256_tilecap_timing.log` used
  `KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 KRASIS_DECODE_GQA_OTHER_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed `14/14`, `ALL TESTS PASSED`, with `5616.5 tok/s` prefill,
  `63.97 tok/s` internal decode, `114.51 tok/s` HTTP, HCS `3840/3840`, min
  free `11472 MB`, clean graph HCS (`240/0` hit/miss, zero cold DMA/DMA
  calls), and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. The
  target non-active attention/reduce bucket across representative 49/99/249/511
  rows was `0.76/0.96/1.46/2.16 ms/tok`, versus prior marker
  `0.75/0.95/1.47/2.16`, so it did not improve. Per gate, no timing-off
  Gemma speed or QCN guard was run. The candidate was reverted and restore
  build `20260613_201820_restore_after_hd256_tilecap_reject_build.log` passed
  in `124s`; no tile-cap candidate branch remains.

- Added diagnostic-only Gemma4 HQQ4/k4v4 non-active GQA geometry attribution
  for the remaining `other_mixed_gqa` bucket. The new report is timing/debug
  gated behind `KRASIS_DECODE_GQA_OTHER_CLOCKS=1`, run with
  `KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1`, and allocation is limited to k4v4
  graph sessions with tiled GQA buffers and at least one HD512 GQA layer. It
  changes no prefill, HCS policy, final segment behavior, QCN/non-HD512 path,
  output math, or speed-path behavior. Build
  `20260613_195304_gemma4_hqq4_k4v4_other_gqa_attr_build.log` passed in
  `127s`. Timing run
  `20260613_195613_gemma4_hqq4_k4v4_other_gqa_attr_timing.log` used
  `KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 KRASIS_DECODE_GQA_OTHER_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed `14/14`, `ALL TESTS PASSED`, with `5608.8 tok/s` prefill,
  `63.48 tok/s` internal decode, `114.54 tok/s` HTTP, HCS `3840/3840`, min
  free `11472 MB`, clean graph HCS (`240/0` hit/miss, zero cold DMA/DMA
  calls), and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`.
  Representative 49/99/249/511 rows showed mixed GQA total
  `3.23/3.48/4.10/4.80 ms/tok`, active HD512 span `0.70/0.74/0.85/0.89`,
  other mixed GQA `2.53/2.73/3.25/3.92`, and active endpoint/name coverage
  `100.0%`. The non-active split was projection `0.95/0.95/0.95/0.94`,
  norm/RoPE/KV `0.26/0.26/0.26/0.25`, attention/reduce
  `0.75/0.95/1.47/2.16`, BF16 `0.05/0.05/0.05/0.05`, O-input prep
  `0.02/0.02/0.02/0.02`, O projection `0.45/0.45/0.45/0.45`, endpoint
  `2.49/2.69/3.21/3.88`, and `24` non-active segments. The measured geometry
  is Gemma's sliding GQA path (`head_dim=256`, `num_attention_heads=16`,
  `num_key_value_heads=8`, `sliding_window=1024`, gated k4v4). Exactly one
  next optimization candidate is selected: decode-only sliding HD256 k4v4
  graph attention tile-cap specialization using per-layer max attention length
  to reduce inactive tile blocks in the row-growing attention/reduce bucket.

- Added diagnostic-only Gemma4 HQQ4/k4v4 mixed-segment coverage attribution for
  the graph GQA route path. The new report is timing/debug gated behind
  `KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1`, reuses existing mixed/path graph
  clock markers, and changes no prefill, HCS policy, final segment behavior,
  QCN/non-HD512 path, or output math. Build
  `20260613_180708_gemma4_hqq4_k4v4_coverage_attr_build.log` passed in
  `123s`. Timing run
  `20260613_180941_gemma4_hqq4_k4v4_coverage_attr_timing.log` used
  `KRASIS_DECODE_GQA_COVERAGE_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed `14/14`, `ALL TESTS PASSED`, with `5602.5 tok/s` prefill,
  `64.32 tok/s` internal decode, `114.32 tok/s` HTTP, HCS `3840/3840`, min
  free `11474 MB`, clean graph HCS (`240/0` hit/miss, zero cold DMA/DMA
  calls), and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`.
  Representative 49/99/249/511 coverage rows showed mixed GQA total
  `3.08/3.24/3.96/4.41 ms/tok`, active HD512 mixed span
  `0.70/0.72/0.86/0.95`, other mixed GQA `2.38/2.52/3.11/3.46`, active
  endpoint `0.69/0.71/0.85/0.94`, active named submarkers
  `0.69/0.71/0.85/0.94`, active internal gap `0.00/0.00/0.00/0.00`, count
  coverage `17.2%`, active time coverage `22.8/22.3/21.6/21.5%`, and named
  endpoint coverage `100.0%`. The earlier path residual was a
  coverage/denominator mismatch: active HD512 markers fully cover the active
  path, but that path is only about one fifth of mixed GQA time. Exactly one
  next candidate is selected: decode-only non-active GQA layer geometry
  attribution before any optimization.

- Added diagnostic-only Gemma4 HQQ4/k4v4 boundary/stream residual attribution
  for the HD512 graph GQA path. The new report is timing/debug gated behind
  `KRASIS_DECODE_GQA_BOUNDARY_CLOCKS=1`, reuses existing graph clock markers,
  and changes no prefill, HCS policy, final segment behavior, QCN/non-HD512
  path, or output math. Build
  `20260613_175047_gemma4_hqq4_k4v4_boundary_attr_build.log` passed in
  `126s`. Timing run
  `20260613_175323_gemma4_hqq4_k4v4_boundary_attr_timing.log` used
  `KRASIS_DECODE_GQA_SPLIT_CLOCKS=1 KRASIS_DECODE_GQA_BOUNDARY_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  and passed `14/14`, `ALL TESTS PASSED`, with `5566.7 tok/s` prefill,
  `64.09 tok/s` internal decode, `115.54 tok/s` HTTP, HCS `3840/3840`, min
  free `11472 MB`, clean graph HCS (`240/0` hit/miss, zero cold DMA/DMA
  calls), and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`.
  Boundary split across representative 49/99/249/511 rows showed GQA-entry
  gap `0.00/0.00/0.00/0.00 ms/tok`, post-O-proj exit gap
  `0.00/0.00/0.00/0.00`, and HQQ/graph stream debt
  `0.00/-0.00/-0.00/-0.00`. The active HD512 marked path was only
  `0.70/0.75/0.82/0.87`, while the mixed GQA-path total was
  `3.09/3.34/3.85/4.60`, proving the earlier path residual is a
  mixed-segment coverage/reporting artifact rather than hidden post-attention
  work. Exactly one next candidate is selected: decode-only coverage
  attribution for the mixed `GRAPH_SEG_ROUTE_GQA` label before any new
  optimization.

- Added diagnostic-only Gemma4 HQQ4/k4v4 post-attention residual attribution
  for the HD512 graph GQA path. The instrumentation remains timing/debug gated
  behind `KRASIS_DECODE_GQA_PATH_CLOCKS=1` and changes no prefill, HCS policy,
  final segment behavior, QCN/non-HD512 path, or output math. First build
  `20260613_172655_gemma4_hqq4_k4v4_post_attn_attr_build.log` passed in
  `124s`, but the first timing run
  `20260613_172936_gemma4_hqq4_k4v4_post_attn_attr_timing.log` did not emit
  the new path split because the added timing gate was too narrow; it still
  passed `14/14` and was cleaned up with `./dev kill` after the server stayed
  alive. After restoring the prior working path-clock gate shape, build
  `20260613_173518_gemma4_hqq4_k4v4_post_attn_attr_build2.log` passed in
  `124s`, and timing run
  `20260613_173800_gemma4_hqq4_k4v4_post_attn_attr_timing2.log` passed
  `14/14`, `ALL TESTS PASSED`, with `5622.8 tok/s` prefill, `65.96 tok/s`
  decode, `115.79 tok/s` HTTP, HCS `3840/3840`, min free `11472 MB`, clean
  graph HCS (`240/0` hit/miss, zero cold DMA/DMA calls), and dynamic-HCS
  `promotions=0 evictions=0 copy_failures=0`. Post-attention split across
  representative 49/99/249/511 rows: `apply_gated_attn_bf16`
  `0.01/0.01/0.01/0.01 ms/tok`, O-proj input prep
  `0.00/0.00/0.00/0.00`, O projection `0.17/0.17/0.17/0.17`, marker residual
  `2.31/2.61/3.06/3.78`, and final graph sync debt
  `1.04/1.04/1.04/1.04`. This confirms the rejected gated BF16 reduce/output
  candidate was not targeting the dominant cost. Exactly one next candidate is
  selected: a decode-only boundary/stream residual attribution to split the
  remaining marker residual into GQA-entry gap, post-O-proj exit gap, and
  HQQ/graph stream debt before any new optimization.

- Measured and rejected a decode-only Gemma4 HQQ4/k4v4 gated BF16
  reduce/output candidate. The candidate inspected the existing boundary
  `gqa_attention_k4v4_tiled_g` -> `gqa_attention_polar4_reduce_g` writing
  FP32 `d_gqa_out` -> `apply_gated_attn_bf16` writing BF16 `d_scratch` ->
  BF16 O projection, then tried to fuse the gated BF16 conversion into the
  k4v4 Polar4 reduce/output stage. The runtime gate was captured graph GQA
  decode only: `GRAPH_SEG_ROUTE_GQA`, `layer_idx == range_end`,
  `kv_format == 9`, effective `head_dim == 512`, gated attention,
  `gqa_tile_size == 256`, `gqa_max_tiles > 1`, and allocated tiled buffers.
  Candidate `./dev build` passed in `133s`. Timing-enabled Gemma k4
  `KRASIS_DECODE_GQA_SPLIT_CLOCKS=1 KRASIS_DECODE_MIXED_SEGMENT_CLOCKS=1 KRASIS_DECODE_GQA_PATH_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  passed `14/14` and `ALL TESTS PASSED`, but regressed to `5399.5 tok/s`
  prefill, `63.25 tok/s` internal decode, and `115.34 tok/s` HTTP. HCS stayed
  clean (`3840/3840`, `240/0` hit/miss, `0` cold DMA/DMA calls,
  `copy_failures=0`) and min free remained `11474 MB`. The GQA-path residual
  bucket did not improve consistently: `2.41/2.62/3.03/3.78 ms/tok` across
  49/99/249/511 rows versus the attribution marker `2.32/2.60/3.09/3.78`.
  The candidate was rejected at the timing gate; no timing-off Gemma speed or
  QCN guard was run. Only the candidate source was reverted, no fused
  gated-reduce symbols remain, and restore `./dev build` passed in `132s`.

- Added diagnostic-only Gemma4 HQQ4/k4v4 HD512 graph GQA path sub-split timing
  behind `KRASIS_DECODE_GQA_PATH_CLOCKS=1`. The instrumentation is active only
  with graph internal timing and the runtime gate `kv_format == 9`, effective
  `head_dim == 512`, a mixed `GRAPH_SEG_ROUTE_GQA` segment, and active tiled
  graph buffers; prefill, HCS policy, final segment behavior, QCN/non-HD512
  paths, and output math are unchanged. Build passed in `124s`, then
  `KRASIS_DECODE_GQA_SPLIT_CLOCKS=1 KRASIS_DECODE_MIXED_SEGMENT_CLOCKS=1 KRASIS_DECODE_GQA_PATH_CLOCKS=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  passed `14/14` and `ALL TESTS PASSED`, with `5614.9 tok/s` prefill,
  `64.23 tok/s` decode, `115.66 tok/s` HTTP, HCS `3840/3840`, min free
  `11474 MB`, clean graph HCS (`240/0` hit/miss, zero cold DMA/DMA calls), and
  dynamic-HCS `promotions=0 evictions=0 copy_failures=0`. The requested GQA
  path split showed projection `0.23/0.23/0.23/0.23 ms/tok`, norm+RoPE+KV
  `0.05/0.06/0.06/0.06`, attention/reduce `0.22/0.27/0.37/0.41`, O projection
  `0.17/0.17/0.17/0.17`, and residual `2.32/2.60/3.09/3.78` across
  49/99/249/511 rows. The selected next candidate is a decode-only HD512 k4v4
  graph path that fuses the post-attention gated BF16 conversion into the k4v4
  reduce/output stage while preserving output math and BF16 O-projection input;
  do not retry rejected GQA/thread/score/single-tile/all-HCS candidates.

- Added diagnostic-only Gemma4 HQQ4/k4v4 mixed graph-segment attribution behind
  `KRASIS_DECODE_MIXED_SEGMENT_CLOCKS=1`. The timing pass split the coarse
  `experts L{n-1} + route L{n} GQA` label into MoE expert compute, full GQA
  path, route/top-k work, segment residual, and final sync without changing
  runtime math, HCS residency, prefill, or graph correctness checks. Gemma k4
  passed `14/14` with `5592.5 tok/s` prefill, `64.24 tok/s` decode,
  `113.71 tok/s` HTTP, HCS `3840/3840`, `240/0` hit/miss, zero cold DMA/DMA
  calls, and min free `11474 MB`. The split showed the full GQA path is the
  row-growing component (`3.07/3.24/3.92/4.58 ms/tok` across 49/99/249/511)
  while tiled attention plus Polar4 reduce remain tiny. The selected next
  candidate is a decode-only HD512 k4v4 graph GQA path sub-split into
  projection, Q/K norm + RoPE + KV-cache write, attention/reduce, and
  O projection before any new optimization; this avoids retrying the rejected
  tiled-attention/thread/score/single-tile/all-HCS paths or the earlier prefill
  post-projection fusions.

- Measured and rejected a decode-only Gemma4 HQQ4/k4v4 all-HCS graph replay
  fast-path candidate. The candidate reused the existing graph-side
  `expert_classify_prepare` pointer-fill path and replayed captured graph
  segments back-to-back only under the tight runtime gate: graph-mode k4v4,
  effective HD512, tiled graph buffers active, GPU classify/pointer-table
  support present, complete HCS coverage for the current decode segment, and
  no cold DMA required. QCN is excluded by `head_dim=256`, k6 by `kv_format`,
  and prefill was not touched. The first timing attempt was stopped before a
  correctness/speed decision because the gate blocked on dynamic-HCS
  bookkeeping even though Gemma showed `3840/3840` resident experts and zero
  cold DMA; the gate was adjusted to allow dynamic-HCS bookkeeping only after
  complete all-resident coverage was proven, while heatmap collection remained
  blocked. The second timing-enabled Gemma run passed `14/14` and `ALL TESTS
  PASSED`, with `5622.5 tok/s` prefill, `65.38 tok/s` internal decode,
  `116.42 tok/s` HTTP, HCS `3840/3840`, `240/0` hit/miss, `0` cold DMA,
  `0` DMA calls, and dynamic-HCS `promotions=0 evictions=0 copy_failures=0`.
  The candidate removed per-segment route sync/upload attribution
  (`inter: 0.00 ms`, classify/upload/launch `0.00 ms`) but moved the wait to
  the final sync. Residual GQA-route rows versus the split attribution marker
  were mixed/worse: 49-token `9.16` vs `9.08`, 99-token `9.36` vs `9.03`,
  249-token `9.71` vs `9.82`, and 511-token `10.41` vs `10.18`. Because the
  required residual bucket did not improve consistently and internal decode
  stayed below the accepted `65.89 tok/s` marker, the candidate was rejected at
  the timing gate. No timing-off Gemma speed or QCN guard was run. The
  all-HCS fast-path source was reverted and `./dev build` passed afterward to
  restore the accepted installed extension.

- Added diagnostic-only Gemma4 HQQ4/k4v4 graph decode split timing for the
  accepted HD512 tiled-GQA path. The final accepted measurement uses an
  env-gated `record_globaltimer_u64_g` marker kernel, enabled only by
  `KRASIS_DECODE_GQA_SPLIT_CLOCKS=1`, to split
  `gqa_attention_k4v4_tiled_g` from `gqa_attention_polar4_reduce_g` inside
  graph replay without changing output math, HCS residency, prefill, or graph
  correctness checks. Earlier CUDA-event instrumentation was rejected after
  graph replay failed with `cuEventElapsedTime` invalid-value errors, and an
  Nsight Systems attempt stalled before usable kernel rows. The final
  timing-enabled Gemma run passed `14/14` and `ALL TESTS PASSED`, with
  `5429.2 tok/s` prefill, `64.67 tok/s` internal decode, `117.31 tok/s` HTTP,
  HCS `3840/3840`, min free `11474 MB`, and clean graph HCS (`240/0` hit/miss,
  `0` cold DMA, `0` copy failures). The split showed the two measured k4v4
  kernels are small versus the coarse GQA-route segment: 49-token row
  `0.20 + 0.02 = 0.21 ms/tok` versus `9.30 ms/tok` GQA route, 99-token
  `0.24 + 0.02 = 0.25` versus `9.29`, 249-token `0.35 + 0.02 = 0.37` versus
  `10.18`, and 511-token `0.39 + 0.02 = 0.40` versus `10.58`. The selected
  next candidate is therefore not another attention/reducer micro-kernel
  change: target the all-HCS-resident graph replay residual by removing the
  per-layer CPU route sync/upload loop when runtime HCS coverage is complete.
  No timing-off speed run or QCN guard was run for this attribution-only pass.

- Measured and rejected a decode-only Gemma4 HQQ4/k4v4 single-tile
  reducer-bypass candidate. The exact gate was graph-mode k4v4 decode with
  effective `hd == 512`, `tile_size == 256`, `max_tiles > 1`, allocated tiled
  buffers, and runtime device-side `num_tiles == 1`; prefill, HCS residency,
  graph correctness checks, final logits, k6, non-HD512 paths, and QCN
  `head_dim=256` were not intended to change. First `./dev build` failed
  before runtime testing because the added output pointer exceeded `cudarc`
  tuple launch arity; the launch plumbing was fixed with the existing raw
  parameter-vector style and the second build passed. Timing-enabled Gemma
  validation passed `14/14` and `ALL TESTS PASSED`, with benchmark summary
  `4904.0 tok/s` prefill, `64.09 tok/s` internal decode, `115.89 tok/s` HTTP,
  and min free `11474 MB`. HCS stayed clean, but GQA route timing did not
  improve consistently versus the accepted tiled timing marker, and internal
  decode regressed versus the accepted timing run (`64.94 tok/s`). Per the
  gate, no timing-off Gemma speed or QCN guard was run. The candidate was
  reverted and `./dev build` passed afterward to restore the accepted
  installed extension.

- Measured and rejected a decode-only Gemma4 HQQ4/k4v4 HD512 score-unroll
  candidate. The exact affected path was graph-mode k4v4 decode with effective
  `hd == 512`, `tile_size == 256`, `max_tiles > 1`, and existing tiled buffers;
  prefill, HCS residency, output math, graph segmentation, final logits, and
  non-HD512 paths were not changed. `./dev build` passed. Timing-enabled Gemma
  validation passed `14/14` with benchmark summary `4876.5 tok/s` prefill,
  `65.18 tok/s` internal decode, `114.41 tok/s` HTTP, and min free
  `11474 MB`; visible graph rows showed lower GQA route cost in representative
  rows while final stayed flat around `1.04-1.07 ms/tok`, so the timing-off
  speed gate and QCN guard were run. Timing-off Gemma validation passed
  network `14/14` but regressed against the accepted tiled-GQA marker:
  `5604.3 tok/s` prefill, `65.32 tok/s` decode, `116.10 tok/s` HTTP,
  HCS `3840/3840`, min free `11474 MB`, and `0` copy failures versus
  accepted `5613.8/65.89/118.74`. QCN guard completed with `6354.9 tok/s`
  prefill, `90.90 tok/s` decode, `153.41 tok/s` HTTP, HCS `15957/24576`,
  min free `896 MB`, and `0` copy failures; QCN `head_dim=256` is gated away
  from the HD512 branch. The candidate was rejected, the score-unroll branch
  was reverted, and `./dev build` passed to restore the accepted installed
  extension.

- Measured and rejected a decode-only Gemma4 HQQ4/k4v4 HD512 thread-count
  candidate. Latest tiled timing showed GQA route dominates graph-mode decode
  across 49/99/249/511-token rows (`9.20/9.52/10.11/10.65 ms/tok`) while the
  final segment stays flat around `1.04-1.05 ms/tok`, so the candidate tried
  launching the existing graph-mode k4v4 HD512 tiled GQA kernel with `512`
  attention threads instead of `256`. `./dev build` passed, and timing-enabled
  Gemma validation passed `14/14` with `5559.5 tok/s` prefill, `64.47 tok/s`
  internal decode, `117.56 tok/s` HTTP, and min free `11474 MB`; long-row GQA
  buckets improved (`249` `10.01` vs `10.11 ms/tok`, `511` `10.49` vs
  `10.65`), but the 49-token GQA bucket regressed and the final segment did
  not move. Timing-off Gemma validation passed `14/14` but regressed versus
  the accepted tiled-GQA marker: `5471.1 tok/s` prefill, `65.25 tok/s`
  internal decode, `118.26 tok/s` HTTP, HCS `3840/3840`, min free
  `11474 MB`, and `0` copy failures, versus accepted `5613.8/65.89/118.74`.
  QCN guard completed with `6348.3 tok/s` prefill, `88.73 tok/s` internal
  decode, `148.18 tok/s` HTTP, HCS `15957/24576`, min free `896 MB`, and `0`
  copy failures; QCN is gated away by `head_dim=256`. The candidate was
  reverted and `./dev build` passed afterward, restoring the accepted
  tiled-GQA decode source.

- Accepted a small decode-only Gemma4 HQQ4/k4v4 graph-mode tiled-GQA dispatch
  candidate after measured attribution showed the gap is GQA route/final
  segment replay time rather than HCS misses. The candidate is deliberately
  narrow: for graph-mode k4v4 HD512 decode with runtime tiled buffers present,
  use the existing tiled k4v4 GQA kernel plus Polar4 reduce path; otherwise keep
  the existing single-block k4v4 route kernel. Prefill code, HCS residency,
  output math, and correctness checks are not changed. `./dev build` passed.
  Timing-enabled validation
  `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  passed `14/14`, with summary `4691.3 tok/s` prefill, `64.94 tok/s`
  internal decode, `119.80 tok/s` HTTP, and min free `11474 MB`; HCS remained
  clean (`240/0` hit/miss, `0` cold DMA, `0` copy failures). Timing buckets
  were mixed but improved enough to run the speed gate: 49-token graph decode
  `15.23 ms/tok` vs prior `15.27`, 99-token `15.47` vs `15.63`, 249-token
  internal `16.12` vs `15.97`, and 511-token code-gen `16.61` vs `17.19`.
  Timing-off Gemma validation passed network `14/14` with `5613.8 tok/s`
  prefill, `65.89 tok/s` internal decode, `118.74 tok/s` HTTP, HCS
  `3840/3840`, min free `11474 MB`, and `0` copy failures. QCN guard
  `./dev speed-test` completed with `6351.4 tok/s` prefill, `87.72 tok/s`
  internal decode, `199.77 tok/s` HTTP, HCS `15957/24576`, min free `896 MB`,
  and `0` copy failures; QCN `head_dim=256`, so it is gated away from the new
  HD512 branch. Gemma decode remains far below the `200 tok/s` target.

- Ran a timing-enabled Gemma4 HQQ4/k4v4 q2 attribution pass on the current
  source before further optimization:
  `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`.
  The run passed `14/14` and `ALL TESTS PASSED`, with summary `5561.1 tok/s`
  prefill, `64.76 tok/s` internal decode, `116.67 tok/s` HTTP, and min free
  VRAM `11474 MB`. Graph-mode decode showed HCS is not the bottleneck:
  `100.00%` HCS hit rate, `240/0` hit/miss, `0.00 MB/tok` DMA, and `0` copy
  failures. The measured decode limiter is graph replay/sync wait around GQA
  route segments and the final segment: the 249-token graph row was
  `15.97 ms/tok`, with GPU compute `4.44 ms`, sync wait `11.23 ms`, GQA route
  `10.04 ms/tok`, and final sync about `1.04 ms`. The long code-gen row was
  `17.19 ms/tok` with sync wait `12.48 ms`. Long-prefill residuals remain in
  the five HD512 custom-tiled launches (`~2504.6 ms` over 5 calls on a
  14780-token inner row); KV append was only `3.2 ms` over 30 calls. No runtime
  code was changed in this attribution pass.

- Accepted the Gemma4 HQQ4 HD512 q2/wide full-attention specialization toward
  the RTX 5090 target (`10000 tok/s` prefill, `200 tok/s` decode, correctness
  required). `./dev build` passed first. Timing-enabled k4 validation
  `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  passed network `14/14` and showed `5563.7 tok/s` best prefill. Timing-off
  k4 validation `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` passed
  network `14/14` with `5613.0 tok/s` prefill
  (`2932.6`, `5613.0`, `4679.2`, `3794.3`, `3649.8`, `3572.9 tok/s` rows),
  `65.37 tok/s` decode, `117.61 tok/s` HTTP, HCS `3840/3840`, min free
  `11474 MB`, and `0` copy failures. The k6 guard
  `./dev test tests/gemma-4-4-hqq4-k6v6-a16.conf` passed network `14/14` with
  `4942.5 tok/s` prefill, `65.50 tok/s` decode, `117.91 tok/s` HTTP, HCS
  `3840/3840`, min free `11748 MB`, and `0` copy failures. Production QCN
  guard `./dev speed-test` passed with `6349.0 tok/s` prefill,
  `90.21 tok/s` decode, `199.75 tok/s` HTTP, HCS `15957/24576`, min free
  `928 MB`, and `0` copy failures. The Gemma prefill gain is accepted; Gemma
  decode remains far below target and Gemma min-free VRAM remains far above
  the default `600 MB` safety target because the tested Gemma configs already
  keep all `3840/3840` experts hot.

- Ran connectivity-only Zephyrus diagnosis after the pre-release matrix
  blocked. Existing notes still identify Zephyrus as `main@192.168.1.228`
  from `/home/main/Documents/BOX_3070.txt`. Local networking has an on-link
  route (`192.168.1.228 dev enp65s0 src 192.168.1.181`), but reachability
  fails below SSH: ping returned `Destination Host Unreachable`, neighbour
  lookup reported `192.168.1.228 dev enp65s0 FAILED`, ARP showed
  `(incomplete)`, and SSH still returned
  `ssh: connect to host 192.168.1.228 port 22: No route to host`. Full log:
  `benchmarks/20260613_093522_zephyrus_connectivity_diagnosis.log`. The
  pre-release matrix remains blocked; podman and model validation were not
  run.

- Began the pre-release environment validation path with no optimization
  changes. Existing docs/scripts identify the remote variants as Zephyrus
  installed-command Qwen3.5-35B-A3B HQQ4/k4v4 500 MB KV plus HQQ4+10% k4v4
  500 MB KV, and the podman variant as `krasis-run` with the Lore
  Qwen3.6-35B-A3B HQQ6+10% k6v6 config. Clean `./dev build` passed and was
  logged to `benchmarks/20260613_093124_prerelease_build.log`. The first
  formal Zephyrus preflight failed before any model run with
  `ssh: connect to host 192.168.1.228 port 22: No route to host`; full log:
  `benchmarks/20260613_093124_prerelease_zephyrus_preflight_failed.log`. Per
  the validation rule, the matrix stopped there and podman was not run as a
  substitute.

- Ran a final timing-off Gemma4 HQQ4 validation matrix on the current source
  state after the accepted k4v4 and k6v6 `head_dim=512` full-attention
  specializations. `./dev build` passed first. The k4v4 validation
  `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` passed network `14/14` with
  `4176.7 tok/s` prefill (`2824.5`, `4176.7`, `3013.0`, `2291.5`, `2283.2`,
  and `2214.2 tok/s` rows), `65.47 tok/s` decode, `117.99 tok/s` HTTP, HCS
  `3840/3840`, min free `11474 MB`, and `0` copy failures observed in the HCS
  rows. The k6v6 validation
  `./dev test tests/gemma-4-4-hqq4-k6v6-a16.conf` passed network `14/14` with
  `3864.2 tok/s` prefill (`1565.0`, `3864.2`, `2991.2`, `2865.7`, `2738.2`,
  and `2829.3 tok/s` rows), `65.69 tok/s` decode, `118.91 tok/s` HTTP, HCS
  `3840/3840`, min free `11748 MB`, and `0` copy failures observed in the HCS
  rows. Both remain above the accepted comparison points (`4056.2 tok/s` k4v4
  and `3750.8 tok/s` k6v6). No optimization candidate or QCN guard was run
  from this validation-only pass.

- Added and accepted a separate Gemma4 HQQ4/k6v6 stage-exact
  `head_dim=512` full-attention prefill specialization gate. Attribution showed
  k6v6 uses active FP8 stage-exact KV during prefill (`kv_format=1`,
  `decode_kv_format=7`, `prefill_kv_active=true`), while the attention fallback
  itself still consumes current BF16 `q/k/v` projection buffers before the
  stage-exact append. The existing k4v4 direct-cache gate remains unchanged;
  the new predicate is limited to Gemma4 HQQ4 k6v6 stage-exact full attention
  (`start_pos=0`, `window=0`, `head_dim=512`, `16` Q heads, `2` KV heads) and
  the same runtime opt-in shared-memory check. Timing-off
  `./dev test tests/gemma-4-4-hqq4-k6v6-a16.conf` passed network `14/14` and
  improved k6v6 prefill from the tracked `2268.3 tok/s` baseline to
  `3750.8 tok/s` best (`1579.0`, `3750.8`, `3013.3`, `2826.2`, `2735.9`, and
  `2864.0 tok/s` rows), with `65.68 tok/s` decode, `118.34 tok/s` HTTP, HCS
  `3840/3840`, min free `11748 MB`, and `0` copy failures. QCN guard
  `./dev speed-test` completed with `6357.4 tok/s` prefill, `88.91 tok/s`
  decode, `145.97 tok/s` HTTP, HCS `15957/24576`, min free `864 MB`, and `0`
  copy failures.

- Added timing-only Gemma4 HQQ4/k6v6 attribution by extending only the
  reporting gates that had been k4v4 direct-cache specific. The accepted
  `head_dim=512` specialization remains k4v4-only. Full
  `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k6v6-a16.conf --timing`
  passed `ALL TESTS PASSED` with timing-enabled benchmark summary
  `1647.9 tok/s` prefill, `64.31 tok/s` decode, `116.77 tok/s` HTTP, HCS
  `3840/3840`, min free `11748 MB`, and `0` copy failures. Timing rows were
  `1555.0`, `1647.9`, `1012.4`, `972.0`, `907.7`, and `892.2 tok/s`.
  The tracked timing-off k6 context remains baseline `2268.3 tok/s` and
  rejected candidate `2264.1 tok/s`. Attribution showed k6v6 prefill runs
  stage-exact (`kv_format=1`, `decode_kv_format=7`,
  `prefill_kv_active=true`), so the prior k6 HD512 gate extension did not
  match the real prefill path. The five `custom_no_fa2` full-attention layers
  are still `5/11/17/23/29`; on the 8419-token calibration row,
  `flash_attn_tiled_launch` owned `6313.9 ms` wall / `6314.0 ms` CUDA-event
  time over those five calls, while `kv_append_kernel` was only `2.1 ms` over
  30 calls. No optimization candidate or QCN guard was run.

- Rejected extending the Gemma4 HQQ4 `head_dim=512` full-attention
  specialization from k4v4 to k6v6. The existing k6v6 config
  `tests/gemma-4-4-hqq4-k6v6-a16.conf` was benchmarked timing-off first:
  baseline passed network `14/14` with `2268.3 tok/s` best prefill
  (`2268.3`, `1708.6`, `950.7`, `891.2`, `893.5`, `905.0 tok/s` rows),
  `65.82 tok/s` decode, `115.72 tok/s` HTTP, HCS `3840/3840`, min free
  `11748 MB`, and `0` copy failures. The k6v6 gate candidate passed network
  `14/14` but regressed slightly to `2264.1 tok/s` best prefill (`2264.1`,
  `1679.0`, `950.2`, `892.9`, `895.3`, `903.5 tok/s` rows), with
  `65.89 tok/s` decode, `118.24 tok/s` HTTP, HCS `3840/3840`, min free
  `11748 MB`, and `0` copy failures. The k6v6 gate was reverted and
  `./dev build` passed afterward, so the installed extension is back to the
  accepted k4v4-only gate. No QCN guard was run because no speed win was
  accepted.

- Added a gated Gemma4 HQQ4/k4v4 `head_dim=512` full-attention prefill
  specialization for the five unsupported-FA2 GQA layers. The new CUDA kernel
  keeps the existing causal fallback math but uses a fixed `head_dim=512` path
  with a larger `BC=32` KV tile, enabled only when runtime opt-in shared memory
  is sufficient and the layer is Gemma4 HQQ4/k4v4 direct single-chunk full
  attention (`start_pos=0`, `window=0`, `head_dim=512`, `16` Q heads, `2` KV
  heads, `prefill_kv_active=false`). Timing-off
  `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf` passed network `14/14` and
  improved Gemma HQQ4/k4v4 prefill to `4056.2 tok/s` best (`1409.2`, `4056.2`,
  `3004.7`, `2219.5`, `2222.5`, `2211.8 tok/s` rows), with `65.31 tok/s`
  decode, `118.03 tok/s` HTTP, HCS `3840/3840`, min free `11474 MB`, and `0`
  copy failures. QCN guard `./dev speed-test` completed with `6365.4 tok/s`
  prefill, `87.28 tok/s` decode, `146.51 tok/s` HTTP, HCS `15957/24576`, min
  free `928 MB`, and `0` copy failures.

- Added timing-only `custom_no_fa2` / `custom_tiled` sub-step attribution for
  the Gemma4 HQQ4/k4v4 direct-cache GQA path. The instrumentation brackets the
  custom fallback wrapper only: entry, head-dim validation, layout/shared-memory
  math, cache-pointer selection, argument packing, pre-launch checkpoint, and
  the `flash_attn_tiled` launch. Full
  `KRASIS_PREFILL_TIMING=1 ./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing`
  passed network `14/14` with timing-enabled benchmark summary `2220.7 tok/s`
  prefill, `64.68 tok/s` decode, `116.62 tok/s` HTTP, HCS `3840/3840`, min
  free `11474 MB`, and `0` copy failures. Timing-row speeds were `4764`,
  `1806`, `990`, `729`, `689`, and `728 tok/s`. The host-side sub-steps were
  `0.0 ms`; `flash_attn_tiled_launch` owned the wait with event time
  `116.1 ms` on the 1K row, `2424.7 ms` on 5K, `9380.5 ms` on 10K,
  `13091.7 ms` on the 11,824-token calibration row, and
  `19171.5/20346.8/19175.0 ms` on the capped 14,780/14,780/14,779 rows. No
  optimization candidate or QCN guard was run.

- Added timing-only attention-branch attribution for the Gemma4 HQQ4/k4v4
  direct-cache GQA path. The instrumentation logs every Gemma HQQ4/k4 GQA call
  with layer, token count, attention type/window, branch, path, and fixed-FA2
  status, then adds before/after sync-debt checkpoints only around non-fixed
  attention branches. Full
  `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` passed network
  `14/14` with timing-instrumented benchmark summary `2097.3 tok/s` prefill,
  `64.77 tok/s` decode, `116.76 tok/s` HTTP, HCS `3840/3840`, min free
  `11474 MB`, and `0` copy failures. Timing-row speeds were `4909`, `1863`,
  `1047`, `728`, `689`, and `690 tok/s`. The five non-fixed calls are layers
  `5/11/17/23/29`, all `custom_no_fa2` / `custom_tiled` because
  `fa2_head_dim=false`. Their after-branch sync debt owns the long-row wait:
  `13075.6 ms` on the 11,824-token row, `8830.2 ms` on the 10K row, and
  `19179.9/20318.5/20313.0 ms` on the measured 14,780/14,780/14,779-token
  rows. No optimization candidate or QCN guard was run.

- Added timing-only append-gap sync-debt bisection for the Gemma4 HQQ4/k4v4
  direct-cache GQA path. The new `KRASIS_PREFILL_TIMING=1` checkpoints bracket
  the exact interval from fixed-length FA2 return/bookkeeping through attention
  branch exit, optional reference/trace hooks, append setup, k4 cache geometry,
  append timing-event record, and the direct-cache append launch. Full
  `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf --timing` passed network
  `14/14` with timing-instrumented benchmark summary `1630.4 tok/s` prefill,
  `66.10 tok/s` decode, `115.73 tok/s` HTTP, HCS `3840/3840`, min free
  `11474 MB`, and `0` copy failures. The old timing-off speed markers remain:
  prior unreproduced best `2303.8 tok/s`, latest timing-off best `1653.2 tok/s`.
  Timing-row speeds in this diagnostic run were `4803`, `1862`, `1048`, `729`,
  `687`, and `728 tok/s`. The 11,824-token attribution row showed
  `after_fa2_bookkeeping=0.0 ms`, then `after_attention_branch=13080.6 ms`,
  while all following reference/trace/append setup/event/launch checkpoints were
  ~`0.0 ms`; append itself was `2.5 ms` wall / `2.2 ms` event. The 14,780-token
  rows showed the same pattern (`after_attention_branch` `19162.0-20399.1 ms`).
  Since fixed FA2 had 25 calls but `after_attention_branch` had 30 calls, the
  producer is narrowed to unbracketed non-fixed-FA2 attention branch/layer work,
  not trace/reference hooks, append setup, append timing events, or the append
  kernel. No optimization candidate or QCN guard was run.
- Added timing-only sync-debt bisection for the Gemma4 HQQ4/k4v4 direct-cache
  GQA path. The new `KRASIS_PREFILL_TIMING=1` checkpoints force a stream sync
  at Gemma4 layer handoff, pre-GQA, GQA entry, after individual Q/K/V
  projections, after fixed-length FA2, immediately before append, and after O
  projection. On the 11,824-token row, the debt was absent at layer entry,
  GQA entry, and after Q/K/V (`0.0 ms` each over 30 calls), small at pre-GQA
  (`6.7 ms`), and moved to `pre_append`: `13063.3 ms` over 30 calls before
  append was launched. The append itself was only `2.2 ms` wall / `2.0 ms`
  event. On the 14,780-token warmup row, `pre_append` was `19119.8 ms`, while
  append was `3.2/3.0 ms`. This confirms the old append attribution was a sync
  placement artifact and narrows the producer to the interval after the
  post-FA2 checkpoint and before append launch, not Q/K/V projection, GQA
  entry, prior Gemma4 handoff, or the append kernel. The run was stopped after
  attribution; no optimization candidate or QCN guard was run.
- Added timing-only CUDA event instrumentation around the Gemma4 HQQ4/k4v4 GQA
  queue boundary after direct append timing ruled out the append kernel body.
  The diagnostic run bracketed Q/K/V projection, QKV norm, RoPE, fixed-length
  FA2, direct k4/v4 append, O projection, and the explicit stream-sync waits
  while preserving existing wall buckets. On the 11,824-token row, projection
  was `181.3 ms` wall / `174.6 ms` event, QKV norm `11.7/11.5`, RoPE
  `6.8/6.6`, FA2 `137.6/137.4`, O projection `92.6/92.4`, but direct append
  was `13066.9 ms` wall with only `1.9 ms` append-kernel event time. On the
  14,780-token warmup row the same pattern held: append wall `20260.6 ms`,
  append event `2.9 ms`, while projection/FA2/O were close to their event
  times. Conclusion: the 13-20s debt is not produced by Q/K/V projection,
  norm/RoPE, FA2, O projection, or the append kernel body; it appears as an
  explicit stream-sync wait at the append boundary. The timing run was stopped
  after attribution and no optimization candidate or QCN guard was run.
- Added timing-only CUDA event instrumentation for the accepted Gemma4
  HQQ4/k4v4 direct-cache `kv_append` route. The diagnostic run showed the old
  wall/sync `kv_append` bucket was a synchronization attribution artifact, not
  a slow append kernel: on the 11,824-token row, wall `kv_append` was
  `13070.1 ms`, but `kv_append_kernel` was only `2.0 ms` over 30 calls
  (`13068.1 ms` wall-minus-kernel). On a 14,780-token warmup row, wall
  `kv_append` was `19114.7 ms` while `kv_append_kernel` was `3.0 ms`. No
  replacement k4/v4 append kernel was built because the kernel body is not the
  measured bottleneck. A timing-off validation run after the instrumentation
  passed network `14/14` with `65.41 tok/s` decode, `117.73 tok/s` HTTP, HCS
  `3840/3840`, `0` copy failures, and `11474 MB` minimum free VRAM, but did
  not produce a new speed win: best prefill was `1653.2 tok/s` because the 1K
  row did not reproduce the earlier `2303.8 tok/s` best. The longer rows were
  comparable to the accepted baseline, so this remains attribution-only
  instrumentation rather than an accepted speed change.
- Tested a deeper Gemma4 HQQ4/k4v4 projection-backend variant after the
  post-projection fusion attempts failed. A timing-enabled attribution run
  confirmed the long-row bottleneck is still KV append/staging: `11824` tokens
  spent `13082.4 ms` in `kv_append`, while HQQ projection internals were much
  smaller (`237.0 ms` Marlin float-zp, `5.2 ms` group sums, `10.4 ms`
  correction GEMM, `18.4 ms` correction add). A base-only HQQ4 Marlin
  projection experiment skipped the HQQ4 two-scale/intercept repair and passed
  network `14/14`, but regressed timing-off prefill to `1742.1 tok/s` best
  with visibly degraded code generation, versus the accepted HQQ4/k4v4
  stage-exact skip at `2303.8 tok/s`. The experimental backend path was
  reverted and `./dev build` passed afterward; no new runtime default was
  accepted.
- Ran a larger Gemma4 HQQ4/k4v4 fused cache-prep candidate after the isolated
  Q/K fusion failed. The candidate fused Q norm+half-split RoPE, K
  norm+half-split RoPE+final k4 cache write, and V no-scale RMSNorm+final v4
  cache write for the already validated single-chunk direct-cache HQQ4/k4v4
  route. It built and passed `./dev test tests/gemma-4-4-hqq4-k4v4-a16.conf`
  with network `14/14`, decode `65.13 tok/s`, HTTP `117.36 tok/s`, HCS
  `3840/3840`, and `0` copy failures, but regressed prefill to `1710.7 tok/s`
  best (`1552.5`, `1710.7`, `969.3`, `713.5`, `667.0`, `668.8` row speeds)
  versus the current accepted HQQ4/k4v4 skip at `2303.8 tok/s`. The fused
  kernels and gate were reverted, `./dev build` passed afterward, and no new
  runtime default was accepted.
- Ran a Gemma4 fused HQQ/GQA kernel pass. Rejected a fused Q/K RMSNorm plus
  half-split RoPE prefill kernel because it preserved startup but severely
  regressed timing-off HQQ4/k4v4 prefill: early rows were `1659.5`, `1633.2`,
  `958.4`, `673.8`, `668.5`, and `669.0 tok/s` before the run was stopped,
  versus the accepted HQQ4/k4v4 skip at `2303.8 tok/s`. Rejected broadening
  the stage-exact skip from Gemma4 HQQ4/k4v4 to HQQ8/k4v4 after the timed
  prefill row stalled with GPU0 pegged and no benchmark row output in the
  bounded run. Both candidates were reverted, `./dev build` passed afterward,
  and the only active runtime speed diff remains the previously validated
  HQQ4/k4v4 single-chunk stage-exact skip.
- Ran a Gemma4 HQQ prefill mode optimization pass. Rejected persistent BF16
  materialization from HQQ4 because it passed network `14/14` but regressed
  best prefill to `1707.7 tok/s` and reduced min free VRAM to `9330 MB`.
  HQQ8 `native-fused-marlin` improved HQQ8/k4v4 prefill from `1590.0` to
  `1734.0 tok/s` with `65.26` decode, `119.35` HTTP, network `14/14`, and
  HCS `3840/3840`. HQQ8 `native-fused-marlin-v2` was the best HQQ8 format:
  `1855.3` prefill, `66.67` decode, `119.10` HTTP, network `14/14`, and HCS
  `3840/3840`. Rejected HQQ8 `symmetric-marlin` because it was slower
  (`1652.3` prefill) and produced a visibly weaker multi-turn sample despite
  passing the short suite. No new runtime default was accepted because even the
  best HQQ8 mode remains below the current HQQ4/k4v4 skip (`2303.8`) and far
  below Gemma BF16/k4v4 (`5196.5`).
- Replaced the launcher Hugging Face model search/download flow with a curated
  supported-model downloader. The launcher now offers only the Krasis-supported
  model catalog, downloads each entry to the expected `~/.krasis/models/<name>`
  directory, and passes the catalog's pinned Hugging Face revision into
  `snapshot_download` so upstream `main` changes do not silently alter local
  installs. The initial catalog contains Qwen3-Coder-Next, Qwen3.6-35B-A3B,
  Qwen3.5-35B-A3B, Qwen3.5-122B-A10B, Qwen3-235B-A22B, and Gemma4 26B A4B IT.
- Started native Gemma4 text support on branch `gemma-dev`. Added Gemma config
  parsing for `top_k_experts`, `hidden_activation`, full-attention head
  geometry, `attention_k_eq_v`, and per-layer GQA helper methods; corrected the
  Gemma test config model path casing; taught the weight loader about Gemma's
  `router.proj`, router scale/per-expert-scale tensors, extra feed-forward
  norms, dense-MLP-plus-MoE layer shape, and full-attention `v_proj` aliasing.
  Extended the Marlin expert cache builder to detect Gemma's sibling
  `layers.N.experts.gate_up_proj/down_proj` stacked BF16 experts.
- Implemented the Gemma4 text hot path for INT4 routed experts, BF16
  attention, compact BF16 KV, and tied embedding/lm-head weights. The Rust
  prefill and decode paths now handle Gemma4 embedding scaling, final logit
  softcap, plain final RMSNorm, half-split RoPE, per-layer GQA geometry,
  full-attention `attention_k_eq_v`, no-scale V RMSNorm, Gemma4 router
  scale/per-expert scale, extra pre/post feed-forward norms, dense MLP plus
  routed MoE branch composition, and `layer_scalar` residual scaling.
- Added Gemma4 tokenizer handling for `extra_special_tokens` list configs and
  sibling `chat_template.jinja`, plus minijinja support for Python-style
  `.get(key[, default])` calls used by the Gemma chat template.
- Added Gemma4 variable per-layer compressed KV cache support for `k6v6` and
  `k4v4`. The default Gemma4 k6/k4 configs use full per-layer compressed KV
  storage: with `CFG_KV_CACHE_MB=1000`, k6v6 provides a `10624`-token context
  and k4v4 provides a `14880`-token context on the local RTX 5090.
- Implemented an experimental Gemma4 ring-window KV path for sliding-attention
  layers and gated it behind explicit `CFG_RING_WINDOW_KV=1` /
  `--ring-window-kv`. Fixed the k6v6 ring 25K long-prompt corruption by
  passing real prefill chunk starts into GQA attention and by routing
  ring-capped sliding layers through the custom local-window prefill path
  instead of the FA2 local-window path. k6v6 ring remains explicit/diagnostic
  pending witness validation and 100K quality review; k4v4 ring is now rejected
  before model load because it still produced invalid 25K output during
  validation. The k4v4 non-ring config remains supported.
- Gemma4 fixed HQQ attention modes are now supported for non-ring compressed
  KV. The model constructor accepts Gemma4 `attention_quant` values `bf16`,
  `hqq4`, `hqq6`, and `hqq8`, while continuing to reject mixed/auto HQQ modes,
  non-INT4 routed experts, and KV formats outside BF16/k6v6/k4v4. Gemma4 HQQ
  registration now uses split Q/K/V descriptors for GQA layers instead of the
  generic fused-QKV HQQ descriptor path, because Gemma4 has mixed
  sliding/full-attention geometry and the fused path failed startup
  calibration on full-attention layers. Validation covered all fixed-HQQ
  combinations with `./dev test`: HQQ8/k6v6 `1838.4` prefill / `66.56`
  decode, HQQ6/k6v6 `1632.5` / `63.85`, HQQ4/k6v6 `1706.5` / `65.39`,
  HQQ8/k4v4 `1590.0` / `65.42`, HQQ6/k4v4 `1628.2` / `64.43`, and HQQ4/k4v4
  `2150.7` / `65.25`; all passed `14/14` network prompts with HCS
  `3840/3840` and `0` copy failures. HQQ attention saves VRAM but is slower
  than the BF16 attention fast path in prefill. HQQ4 also produced a visibly
  weaker code-generation sample despite passing the short network test, so
  Gemma4 HQQ should not be treated as llama-witness validated.
- Gemma4 decode CUDA graphs are disabled by model capability rather than by an
  environment variable. The supported Gemma4 decode path is currently
  ungraphed; per-layer graph capture still needs a Gemma-aware graph segment
  implementation for dense-MLP-plus-routed-MoE layers.
- Improved the diagnostic Gemma4 k6v6 ring-window prefill path after timing
  showed the old custom path charging almost all 39,920-token prefill time to
  GQA/KV work (`151956 ms`, `263 tok/s`). Ring-capped sliding layers still use
  bounded local-window attention, now with an FA2 staging path when available,
  and full-attention layers no longer get routed through the custom ring branch.
  A 25K large-prompt validation improved from `421.6 tok/s` to
  `480.1 tok/s` prefill with unchanged `10.2 tok/s` decode, `100%` HCS hit
  rate, `0` copy failures, and `2098 MB` minimum free VRAM. A Gemma4 CUDA
  graph decode attempt failed during long calibration with
  `CUDA_ERROR_ILLEGAL_ADDRESS`, so the model capability guard remains in place
  and the failed graph experiment was removed from the shared graph path to
  avoid slowing existing graph-capable models.
- Improved Gemma4 k6v6 ring-window long-context decode without enabling CUDA
  graph capture. The ungraphed compressed-GQA decode path now reuses the
  existing tiled k6/k8 attention kernels for long sequences when tiled buffers
  and a device sequence-length scalar are available, and falls back to the
  original single-block kernel for short sequences or k4v4. Clean timing-off
  validation through `large_25k` passed with `498.2 tok/s` prefill,
  `28.4 tok/s` decode, `100%` HCS hit rate, `0` copy failures, and `2346 MB`
  minimum free VRAM. Follow-up prefill experiments were rejected: skipping old
  FP8 stage-cache rows produced invalid 25K output, and reducing only the
  export launch grid regressed 25K prefill to `470.2 tok/s`. A later
  ring-aware temp-KV cap reduced calibration prefill growth (`359.0 KB/tok`)
  but still regressed 25K prefill to `470.4 tok/s`; adding a 4096-token chunk
  cap reduced prefill growth further (`155.3 KB/tok`) but collapsed 25K
  throughput to `296.6 tok/s`. Both were reverted because they traded speed
  for memory headroom. A direct compressed k6 ring prefill prototype was also
  rejected: it completed startup calibration safely and reduced measured
  prefill memory growth to `345.1 KB/tok`, but benchmark warmup took `213.0s`
  and the first 1K timed prefill row still had not completed after a bounded
  wait. The active code was reverted to avoid degrading the accepted
  `498.2 tok/s` k6 ring path.
- Added a Qwen35-vs-Gemma4 timing diagnostic to explain the current Gemma
  prefill/decode gap. Added `tests/q35b-4-4-hqq6-k6v6-diagnostic.conf` because
  the older Qwen35 benchmark configs used disabled KV formats. With timing
  instrumentation enabled, Qwen35 HQQ6 k6v6 reached `9593.2 tok/s` internal
  prefill and `113.92 tok/s` internal decode with CUDA graph replay and `100%`
  HCS hits. Gemma4 k6v6 non-ring reached `4945.1 tok/s` best internal prefill
  only at the 1K row and fell to about `1000 tok/s` at 10K; timing showed the
  10K row spent `9404.8 ms` in stage-exact KV append over 30 GQA layers.
  Qwen35's comparable 10K row spent only `0.3 ms` in KV append over 10 GQA
  layers. Gemma4 decode remained ungraphed at `36.60 tok/s`; HCS was `100%`,
  so the decode gap is graph/layer-path cost, not expert cache misses.
  Fixed Gemma4 prefill timing attribution so the top-level `attn/gqa/moe`
  buckets account for the nested Gemma layer timers instead of charging the
  whole Gemma wrapper as broad MoE/other time. A smoke diagnostic on
  `tests/gemma-4-4-k6v6-a16.conf` now reports an 8419-token calibration row as
  `6434.9 ms` GQA/attention with `6304.7 ms` in KV append and only `63.2 ms`
  MoE, matching the root-cause attribution without changing timing-off paths.
- Ran the next Gemma4 prefill/graph speed implementation pass and rejected the
  unsafe or slower paths by measurement. BF16 temp-KV staging worsened the
  8419-token diagnostic (`6679.2 ms` KV append versus the prior `6304.7 ms`);
  FP8-window FA2 routing, vectorized FP8 append, and a direct decode-KV
  single-chunk bypass all left the 39,920-token timing row around `281 tok/s`
  with about `141 s` in KV append, so their active code was removed. A Gemma4
  CUDA graph decode implementation attempt reached roughly `64-73 tok/s`, but
  failed correctness (`2/10` network prompts passed, early EOS/garbled output),
  showing that Gemma4 needs split graph segmentation for dense MLP, router,
  expert, merge, and `layer_scalar` work before replay is safe. The Gemma4
  graph guard is restored with that explicit reason, and unused FP8-window
  sidecar symbols/loaders were removed. Restored-path validation
  `./dev test tests/gemma-4-4-k6v6-a16.conf` passed benchmark plus network:
  `5035.9 tok/s` best prefill, `38.72 tok/s` best internal decode,
  `67.57 tok/s` best HTTP, `14/14` network prompts, HCS `3840/3840`, and
  `11170 MB` minimum free VRAM.
- Implemented Gemma4 CUDA graph decode safely by making the captured graph path
  execute Gemma4's dense-MLP + routed-MoE semantics instead of the generic MoE
  merge. The graph segment now handles Gemma4 pre/post attention RMSNorm,
  dense branch, pre-FFN2 expert input, router input scaling,
  per-expert router scaling, post-expert/post-FFN norms, residual merge, and
  `layer_scalar`. Graph replay now uses per-layer device sequence-length
  scalars, so Gemma sliding-attention layers use their bounded attention length
  while full-attention layers still see full context. Validation:
  `./dev build` passed; `./dev run tests/gemma-4-4-k6v6-a16.conf --test-endpoints`
  plus `./dev network 18013` passed `14/14`; timing-off
  `./dev benchmark tests/gemma-4-4-k6v6-a16.conf` reached `5230.3 tok/s`
  best prefill, `63.92 tok/s` best internal decode, `116.20 tok/s` best HTTP,
  HCS `3840/3840`, `0` copy failures, and `11156 MB` minimum free VRAM.
  Gemma graph decode is deliberately limited to the validated non-ring `k6v6`
  path; k4v4 and ring-window Gemma remain on their existing non-graph paths
  until separately validated.
  Guard run `./dev speed-test` on Qwen3-Coder-Next HQQ4/k4v4 completed with
  `6664.9 tok/s` prefill, `88.42 tok/s` internal decode, `138.06 tok/s` HTTP,
  HCS `15957/24576`, `0` copy failures, and `896 MB` minimum free VRAM.
- Extended Gemma4 CUDA graph decode to the validated non-ring `k4v4` path.
  The first broad k4 graph gate was rejected because pre-HCS calibration
  attempted graph replay and logged `materialized 0 routed experts`; the
  accepted gate now requires HCS residency before enabling k4 graph replay.
  Clean timing-off validation `./dev test tests/gemma-4-4-k4v4-a16.conf`
  improved k4v4 decode from `38.79` to `63.69 tok/s` and HTTP from `66.92`
  to `115.47 tok/s`, with `5192.4 tok/s` best prefill, `14/14` network
  prompts, HCS `3840/3840`, `0` copy failures, and `11030 MB` minimum free
  VRAM. The QCN `./dev speed-test` guard completed afterward at `6856.8`
  prefill, `88.67` decode, `149.12` HTTP, HCS `15957/24576`, `0` copy
  failures, and `896 MB` minimum free VRAM.
- QCN `./dev speed-test` guard after the Gemma4 HQQ change completed at
  `6902.6 tok/s` prefill, `89.37 tok/s` decode, `149.29 tok/s` HTTP, HCS
  `15957/24576`, `0` copy failures, and `928 MB` minimum free VRAM.
- Improved Gemma4 HQQ4/k4v4 prefill by skipping stage-exact temporary KV only
  when runtime scratch budgeting proves the prompt fits as a single chunk. The
  accepted gate is deliberately narrow: Gemma4, fixed HQQ4 GQA, non-ring k4v4
  decode KV, uniform per-layer KV capacity, and measured single-chunk fit.
  Other Gemma HQQ modes and all non-Gemma paths keep the existing stage-exact
  path until separately validated. Timing diagnostics showed HQQ4/k4v4 prefill
  is still dominated by GQA/KV append (`11824` tokens: `13091.5 ms` KV append
  inside `13521.6 ms` GQA), not HQQ projection. Rejected
  `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1` because it regressed prefill to
  `1702.7 tok/s`. The accepted final `./dev test
  tests/gemma-4-4-hqq4-k4v4-a16.conf` improved best prefill from `2150.7` to
  `2303.8 tok/s`, with `65.10 tok/s` decode, `117.49 tok/s` HTTP, `14/14`
  network prompts, HCS `3840/3840`, `0` copy failures, and `11474 MB` minimum
  free VRAM. QCN `./dev speed-test` guard passed afterward at `6356.9`
  prefill, `90.14` decode, `148.29` HTTP, HCS `15957/24576`, `0` copy
  failures, and `928 MB` minimum free VRAM.
- Investigated extending the Gemma4 fast-path prefill and graph-decode
  optimizations beyond the accepted HQQ4/k4v4 gate. The BF16/k4v4 baseline
  `./dev test tests/gemma-4-4-k4v4-a16.conf` measured `5196.5 tok/s`
  prefill, `64.36 tok/s` decode, `115.43 tok/s` HTTP, network `14/14`, HCS
  `3840/3840`, and `11030 MB` minimum free VRAM. Rejected
  `KRASIS_KV_STAGE_EXACT=0` for BF16/k4v4: it reduced long-prefill transient
  memory from `5586 MB` to `4306 MB`, but regressed speed to `5059.5` prefill,
  `63.68` decode, and `114.45` HTTP. Decode graph timing confirmed the
  post-HCS BF16/k4v4 path is already in graph mode; a representative 79-token
  block was `15.59 ms/tok`, with graph segment CUDA time dominated by GQA
  route segments (`9.31 ms/tok`) and final (`1.05 ms/tok`) with no cold DMA.
  Rejected `KRASIS_GPU_ROUTE_SYNC=1` as a graph-decode improvement because it
  passed network `14/14` but regressed to `5051.3` prefill, `62.64` decode,
  and `113.71` HTTP. No new BF16 fast-path runtime change was accepted from
  these probes.
- Ran a focused Gemma4 BF16/k4v4 KV-staging optimization pass with
  prefill/GQA timing enabled. Baseline instrumentation confirmed the long
  calibration row is dominated by stage-exact temporary FP8 KV append:
  `11824` tokens spent `12829.3 ms` in `kv_append` inside `13007.6 ms` GQA,
  with `1270.2 MB` temporary FP8 K/V for 30 active layers. Rejected a 2D
  token-by-KV-segment launch for the generic FP8 append kernels because it was
  unchanged/slower (`12836.7 ms` `kv_append`, `13219.5 ms` total long row).
  Rejected a CUDA `__nv_fp8x2_e4m3` vectorized append candidate: timing
  instrumentation improved the long row to `12505.5 ms` `kv_append`, but the
  timing-off validation regressed BF16/k4v4 best prefill to `5033.4 tok/s`
  with `63.23 tok/s` decode and `115.50 tok/s` HTTP. Network still passed
  `14/14`, HCS stayed `3840/3840`, copy failures were `0`, and min free VRAM
  was `11030 MB`. The vectorized candidate was reverted, and no new
  KV-staging runtime change was accepted.
- Validation: `./dev build` passed. Short reference probes against the legacy
  Gemma4 HF BF16 artifact matched turn 1 exactly and turn 2 first token exactly
  without setting `KRASIS_NO_GRAPH`. The full legacy reference sweep passed
  turns 1-6 and diverged on longer generations with coherent output; Gemma4
  INT4 is not yet llama-witness validated because no local Gemma witness/GGUF
  artifact exists. `./dev benchmark tests/gemma-4-4-a16.conf` completed with
  a clean health scan: `5051.6 tok/s` best prefill, `39.07 tok/s` internal
  decode, `71.33 tok/s` HTTP, `3840/3840` HCS, and `11084 MB` minimum free
  VRAM. `./dev network 18013` on the gated non-ring k6v6 config passed
  `14/14` with `10624` context, and `./dev network 18014` on the gated
  non-ring k4v4 config passed `14/14` with `14880` context; both had clean
  health scans. k6v6 ring-window reached `106784` context and passed the
  formerly failing 25K large-prompt row after the ring fix (`421.6 tok/s`
  prefill, `2222 MB` min free), then the prefill speed follow-up passed the
  same 25K row at `480.1 tok/s`; the 100K network row timed out while the
  server remained GPU-bound, so 100K is still not a validated practical path.
  k4v4 ring-window is explicitly rejected after
  reproducing invalid 25K output. INT8 Gemma4 remains unsupported after diagnostics produced invalid
  token/logprob output.

## 1.0.15 - 2026-06-05

- Hardened Typhon/16GB startup and CUDA-fault safety. HCS startup now keeps two
  measured soft-tier chunks of headroom above the calibrated decode idle floor,
  and server startup gates `KRASIS SERVER READY` on bounded measured
  drain-and-resample checks after VRAM monitor warnings are enabled. If the
  configured safety floor cannot be restored, startup fails visibly instead of
  publishing an unsafe server. CUDA `ILLEGAL_ADDRESS` in prefill/scratch
  release or source-mode HCS boundary sync now exits the process immediately,
  because the CUDA context is poisoned and cannot be recovered by further HCS
  evictions. Validation: `./dev build` passed.
- Hardened request-time VRAM pressure recovery for external VRAM consumers on
  Typhon. Calibration extrapolation now continues the measured slope for prompts
  beyond the long probe instead of clamping at 1.5x, HCS prefill eviction uses
  the actual request token count instead of a 50K cap, minimum-chunk scratch
  allocation fails visibly if the measured post-allocation floor still cannot
  be met, and chat request entry drains pending HCS pressure before tokenization
  or prefill work. The VRAM monitor now records below-critical-floor pressure
  as an immediate drain event rather than exiting solely because an external
  process temporarily consumed VRAM; CUDA illegal-address errors remain fatal.
  Validation: `./dev build` passed with CPython 3.12 packaging for Typhon.
  Typhon stress with a separate CUDA process holding `768 MB` during a
  `64,125`-token prompt completed with HTTP 200, prefill `3610.4 tok/s`, decode
  `29.7 tok/s`, prefill min free `1902 MB`, decode min free `2646 MB`, and
  `copy_failures=0` on an `800 MB` safety margin. A post-pressure small prompt
  completed with decode `54.5 tok/s`, decode min free `918 MB`, and
  `copy_failures=0`.
- Fixed post-scratch reserve planning to use the measured runtime transient for
  the actual prompt length. This preserves the safety-floor check while avoiding
  a false release-test failure where a tiny QCN warmup request inherited the
  long-prompt post-scratch reserve and refused to run despite enough measured
  headroom for that request size.
- Fixed HCS pressure-drain targets to recover to `safety + measured_deficit`
  after a below-safety low, and to keep two physical soft-HCS chunks of
  headroom during forced readiness/decode drains. This was exposed by the QCN
  INT8 multi-GPU release-test path, where GPU0 published ready with only
  `744 MB` free against a `600 MB` margin and later dipped below the floor
  before the next prefill.
- Added a configured-safety-margin entry guard to HCS prefill eviction. Runtime
  prefill starts from a decode-resident HCS state rather than the no-HCS
  calibration state, so the eviction floor now preserves the measured prefill
  requirement plus HQQ stage growth plus two additional configured safety
  margins before scratch allocation.
- Tightened optional prefill pinning so it cannot consume the last measured
  post-scratch safety band. Pinning now only uses surplus above the calibrated
  post-scratch runtime floor plus one configured safety margin, preserving the
  optimization while preventing short HCS-loaded prefill prompts from launching
  with only a tiny margin above the floor.
- Release validation: `./dev release-test QCN` passed all three Qwen3-Coder-Next
  release configs with llama-witness `4/4` validation on each config. Final
  release-test health scan was clean: no CUDA errors, VRAM monitor warnings,
  hard-floor exits, or nonzero HCS copy failures. Release-test speeds were
  INT4 k4v4 HQQ4 `6433.2 tok/s` prefill, `90.36 tok/s` decode,
  `158.41 tok/s` HTTP; INT4 k6v6 HQQ6+8 `6377.3 tok/s` prefill,
  `88.96 tok/s` decode, `144.13 tok/s` HTTP; INT8 k6v6 HQQ6+8 multi-GPU
  `5000.9 tok/s` prefill, `40.30 tok/s` decode, `68.01 tok/s` HTTP.
- Reduced Q122B prefill scratch for fused MoE by sizing the routed/shared
  accumulator as `[tokens, hidden]` FP32 instead of routing-width scratch.
  The fused Marlin reduction scratch is already provided by `d_fp32_scratch`,
  so `d_moe_accum` only needs to hold the final per-token accumulator. The
  exact estimator, coarse startup budget, and actual scratch allocator were
  updated together.
- Validation: `./dev build` passed. Instrumented Q122B HQQ6/k4v4 diagnostic
  completed with a clean health scan, raised the measured-safe long calibration
  probe to `19268` tokens, and made the `20K` diagnostic row one chunk at
  `4737 tok/s`. Clean timing-off
  `./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf` produced
  `3900.5 tok/s` internal prefill, `27.78 tok/s` internal decode, and
  `45.98 tok/s` HTTP with `4050/12288` HCS, `806 MB` minimum free VRAM, and no
  CUDA errors, VRAM monitor warnings, hard-floor exits, or HCS copy failures.
- Added env-gated request/runtime-stage prefill breakdown diagnostics. The
  diagnostic showed `20K` Q122B core prefill was already close to the old fast
  path (`4156 ms` body time), while the broader request window still paid
  about `1.2 s` in HQQ prefill/decode stage copies. Tested and rejected two
  HQQ-stage optimizations: dual-stage residency removed the copy but consumed
  about `4.9 GB` extra VRAM, reduced HCS to `2862/12288`, and regressed
  decode/HTTP; async copy-stream scheduling did not improve stage-copy time.
  After removing those rejected paths, a clean timing-off Q122B benchmark
  produced `3796.8 tok/s` internal prefill, `28.98 tok/s` internal decode, and
  `46.71 tok/s` HTTP with `4050/12288` HCS, `772 MB` minimum free VRAM, and a
  clean CUDA/VRAM/HCS health scan.
- Improved measured-safe Q122B prefill chunk sizing. After startup calibration
  measures the post-scratch runtime low-water delta, prefill scratch planning
  now uses that measured reserve directly instead of adding a second full
  cold-staging reserve on top of it. Before calibration has measured runtime
  overhead, Krasis still falls back to the full cold-staging reserve. This keeps
  the `600 MB` safety margin and automatic HCS pressure behavior intact while
  allowing larger measured-safe chunks.
- Validation: `./dev build` passed. Instrumented Q122B HQQ6/k4v4 diagnostic
  produced `3401.7 tok/s` internal prefill, `28.31 tok/s` internal decode, and
  `48.47 tok/s` HTTP with `776 MB` minimum free VRAM and a clean health scan.
  Clean timing-off `./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf`
  produced `3608.7 tok/s` internal prefill, `27.76 tok/s` internal decode, and
  `47.99 tok/s` HTTP with `4050/12288` HCS, `806 MB` minimum free VRAM, and no
  CUDA errors, VRAM monitor warnings, hard-floor exits, or HCS copy failures.
- Improved Q122B HQQ6+k4v4 prefill planning without weakening the VRAM safety
  margin. Runtime prefill now builds an explicit chunk plan, preserves the
  fewest measured-safe passes, and smooths pathological tiny tails across the
  same pass count. Gate pre-scan and dense pointer-table prefetch now consume
  actual chunk boundaries.
- Added dense-active optional-pinning policy for prefill. When pre-scan shows a
  chunk is near-all-experts, Krasis skips the temporary optional pinning pool
  and uses the dense pointer-table/cold path instead of paying pinning overhead
  for experts that are effectively all active anyway.
- Tightened post-scratch prefill allocation acceptance: scratch is accepted
  only when actual post-allocation free VRAM still covers the measured runtime
  transient floor plus cold-staging reserve, not just the raw safety margin.
- Fixed HCS soft-reload safety after prefill. Reload now sizes chunks by the
  physical soft-tier chunk boundary and leaves one measured soft-HCS chunk of
  headroom above the idle floor, preventing reload from grazing below the
  `600 MB` safety margin before pressure eviction reacts.
- Fixed a test-only Polar4 kernel smoke compile error exposed by
  `./dev test-kernels`, where the append-kernel parameter list referenced an
  undeclared `norm_correction_i32`.
- Validation: `./dev build` passed. Clean timing-off Q122B HQQ6/k4v4 benchmark
  produced `3238.3 tok/s` internal prefill, `27.21 tok/s` internal decode, and
  `47.04 tok/s` HTTP round trip with `4050/12288` HCS, `804 MB` minimum free
  VRAM, and clean CUDA/VRAM/HCS health scan. `PATH="$HOME/.cargo/bin:$PATH"
  ./dev test-kernels polar4_kv_roundtrip_smoke` passed.
- Restored the prefill-pass planner fixes from the shelved Q122B investigation:
  Krasis now evicts additional soft HCS when measured VRAM says that can avoid
  an extra prefill pass, uses post-scratch HCS eviction at chunk boundaries,
  chunks as max-safe plus tail instead of equalizing chunks, and removes a
  double-counted HCS reserve from optional prefill pinning budget calculation.
- Validation: `./dev build` passed, and `./dev speed-test` on QCN HQQ4/k4v4
  produced `5054.2 tok/s` internal prefill, `85.34 tok/s` internal decode, and
  `117.52 tok/s` HTTP round trip with `16443/24576` HCS, `672 MB` minimum free
  VRAM, and clean CUDA/VRAM/HCS health scan.

## 1.0.14 - 2026-06-04

- Fixed VRAM pressure handling after a Typhon Qwen3.6-35B dynamic-HCS failure
  where cleanup-time lows reached `322 MB` free against a `600 MB` safety
  margin before CUDA later reported `ILLEGAL_ADDRESS`. Pending pressure now
  retains the worst observed low until HCS drain reacts, chat requests drain
  HCS pressure at cleanup end, and startup force-drains again immediately
  before publishing server-ready.
- Fixed `./dev release-test` model alias resolution so documented aliases such
  as `QCN` resolve to the real model directory before invoking the guarded
  release-test runner.

- Scoped `./dev benchmark` GPU cleanup to `CFG_SELECTED_GPUS` so benchmarks on one GPU do not stop unrelated Krasis services on other GPUs.

- Deprecated and disabled direct FP8 KV cache modes (`fp8`, `fp8_e4m3`).
  Configs or CLI args that request them now fail explicitly; use `k6v6`,
  `k4v4`, or `bf16` instead.
- Added `--ssh-key-path` / `CFG_SSH_KEY_PATH` for reverse SSH tunnel sharing,
  so managed tunnels can force a specific identity file with `IdentitiesOnly`.

## 1.0.13 - 2026-05-30

- Stable release of the Witsy/Qwen vision compatibility and dynamic HCS safety
  fixes from the `1.0.13` prerelease line.
- When a loaded model supports Qwen image inputs, `/models` now advertises only
  the Witsy/OpenAI-compatible `-vision` model id instead of listing separate
  text and vision entries. Use the same `-vision` id for both text and image
  requests; text-only requests still follow the normal text path.
- Chat completions now accept OpenAI's `max_completion_tokens` field as an
  alias for `max_tokens`, matching current OpenAI SDK clients such as Witsy.
- The Hugging Face downloader now includes `preprocessor_config.json` and
  `processor_config.json`, which are required for Qwen image preprocessing.
- Pillow is now declared as a package dependency because the Qwen image
  preprocessing path requires `PIL` in clean installs and containers.
- Quieted transient socket read timeouts / `EAGAIN` (`os error 11`) during HTTP
  request parsing; incomplete probe connections are ignored without printing a
  scary error, while malformed requests still return `400`.
- Fixed source-mode dynamic HCS VRAM-pressure eviction synchronization. When
  Krasis trims soft HCS chunks after a below-safety VRAM event, it now
  synchronizes the CUDA context before freeing source-backed dynamic HCS chunks,
  matching the decode/prefill boundary safety used by normal prefill eviction.
- Fixed dynamic HCS promotion ordering during graph decode. Promoted expert
  slot contents and the GPU expert pointer table are now updated on the same
  CUDA replay stream, preventing later graph/prefill work from seeing stale or
  incompletely ordered promoted pointers after image-heavy requests on tight
  VRAM systems.

## 1.0.13-rc.2 - 2026-05-29

- When a loaded model supports Qwen image inputs, `/models` now advertises only
  the Witsy/OpenAI-compatible `-vision` model id instead of listing separate
  text and vision entries. Use the same `-vision` id for both text and image
  requests; text-only requests still follow the normal text path.
- Chat completions now accept OpenAI's `max_completion_tokens` field as an
  alias for `max_tokens`, matching current OpenAI SDK clients such as Witsy.
- The Hugging Face downloader now includes `preprocessor_config.json` and
  `processor_config.json`, which are required for Qwen image preprocessing.
- Pillow is now declared as a package dependency because the Qwen image
  preprocessing path requires `PIL` in clean installs and containers.
- Quieted transient socket read timeouts / `EAGAIN` (`os error 11`) during HTTP
  request parsing; incomplete probe connections are ignored without printing a
  scary error, while malformed requests still return `400`.
- Fixed dynamic HCS promotion ordering during graph decode. Promoted expert
  slot contents and the GPU expert pointer table are now updated on the same
  CUDA replay stream, preventing later graph/prefill work from seeing stale or
  incompletely ordered promoted pointers after image-heavy requests on tight
  VRAM systems.

## 1.0.13-rc.1 - 2026-05-29

- Fixed source-mode dynamic HCS VRAM-pressure eviction synchronization. When
  Krasis trims soft HCS chunks after a below-safety VRAM event, it now
  synchronizes the CUDA context before freeing source-backed dynamic HCS chunks,
  matching the decode/prefill boundary safety used by normal prefill eviction.
  This prevents latent async illegal-address failures from surfacing on the
  next request after pressure eviction.

## 1.0.12 - 2026-05-29

- Improved OpenAI-compatible model discovery for clients such as Witsy. The
  model-list endpoint now accepts `/models` as well as `/v1/models`, tolerates
  trailing slashes/query strings, and includes the standard `created` field in
  the returned model object.
- Also accepts root-base OpenAI chat paths (`/chat/completions`) in addition to
  `/v1/chat/completions`, so clients work whether their base URL is configured
  as `http://host:port` or `http://host:port/v1`.
- The server-ready banner now prints client setup details for OpenAI-compatible
  apps: base URL, chat endpoint, models endpoint, API key value, and model name.

## 1.0.11 - 2026-05-29

- Added experimental Qwen image support. Image chat requests now lazily run the
  Qwen3VL BF16 vision tower for image-prefill setup, scatter visual embeddings
  into `<|image_pad|>` positions, use MRoPE rows during prefill, and preserve
  the existing text-only Rust/CUDA path for normal requests.
- Vision staging is transient: the BF16 tower is moved to GPU for image setup
  and released back to CPU before decode. Constrained image requests now return
  an explicit insufficient-VRAM response instead of a generic server failure.
- Added image request validation for OpenAI-style content parts. Video remains
  unsupported, and local image file paths are disabled unless explicitly enabled
  for local testing.
- Recorded final QCN speed-test gates and 35B/122B image smoke validation in
  the benchmark archive.
- Updated the QCN release-test reference path to use the llama-witness artifact
  instead of the archived HF reference.
- Fixed reference validation for first-token witness artifacts so release tests
  validate the captured prefix contract instead of requiring uncaptured tokens.

## 1.0.5 - 2026-05-23

- Added request prefill progress output alongside the existing decode progress
  lines. Chat requests now print real prefill token count, elapsed time, tok/s,
  current VRAM, and min-free VRAM during prefill before decode starts.

## 1.0.4 - 2026-05-23

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
