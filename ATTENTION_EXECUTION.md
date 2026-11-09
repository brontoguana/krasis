# Sparse attention execution

The exact execution profile passes complete relative request validation on an RTX A6000. It remains opt-in; tensor arithmetic remains experimental. Independent absolute model qualification is not established.

## Contract and ownership

The K6 packed cache and selected token indices are unchanged. Sparse attention unpacks existing BF16 scales and signed six-bit codes into FP32 once per selected row, then shares that data across heads. On Ampere and newer devices, the tensor candidate decomposes both operands into TF32 high and residual parts and accumulates all four products in FP32. Scores use 128-feature accumulation bands; values use 128-selection splits. Softmax masks the actual live indices, including sentinel tails.

`KRASIS_MLA_K6_SPLIT=5` opts into qualified tensor geometry: 512 latent dimensions, zero rotary dimensions, head count divisible by 16, and at most 4,096 selections. Other geometries retain existing paths. Modes 1, 2 and 3 retain scalar split, exact score, and exact score plus unpacked-value experiments. Default 0 retains ordinary execution. Compressed pooled-index eager execution conservatively retains the existing kernel; CUDA graph execution uses the qualified candidate. Direct eager and graph operator replay are both checked.

Decode workspace is owned by the graph, allocated before calibration and capture, and reused only by serial main-stream operations. The tested 64-head/2,051-selection geometry uses 7,512,064 additional bytes. Captured pointers remain owned until graph teardown.

`KRASIS_PREFILL_MLA_ABSORB=1` opts into BF16 tensor-core latent-query absorption with FP32 output and full FP32 reductions, for the qualified 64-head/256-query/512-latent geometry. Bands of up to 1,024 rows and a 16 MiB cuBLAS workspace fit inside the existing selection allocation. A stream-ordered copy restores the existing row-major output layout. Small tails and unsupported geometry retain the original operation. No additional persistent prefill allocation is required.

The gathered-attention and value-projection cuBLAS callers also select their stream before binding owned workspace. Selecting a stream resets the workspace binding; the previous order discarded it. Math mode and workspace cleanup remain explicit. See [cuBLAS stream semantics](https://docs.nvidia.com/cuda/cublas/index.html#cublassetstream).

## Exact execution profile

The current candidate uses `KRASIS_MLA_K6_SPLIT=3`, `KRASIS_PREFILL_MLA_ABSORB=2` and `KRASIS_DECODE_V4_INDEX_RADIX_TOPK=1`. The measured control uses 1,024 attention threads; dispatch and cache/chunk settings are retained in immutable receipts. Other paths retain existing execution unless an experiment is explicitly enabled.

Mode 3 produces scores in the original per-score FP32 operation order. It unpacks selected K6 values once, then retains the original weighted accumulation and softmax reduction. For 64 heads and 2,051 selections, owned decode workspace is 4,725,504 bytes.

Prefill mode 2 reuses one head's weights across 32 query rows. Each output retains the original ascending FP32 FMA chain. A CTA owns its 16 KiB shared query tile and writes distinct output elements; no additional global allocation is required. The tested geometry is 64 heads, 256 query/non-positional dimensions and 512 latent dimensions. Rows below 128 and unsupported geometries retain the original kernel.

The exact radix/linear selector reuses existing native buffers and kernels when geometry fits their capacity. It retains score/index tie ordering and falls back to bitonic selection for larger geometry. Actual pooled scores and complete eager/graph selections are checked against an independent stable ordering, including signed zeros, ties, tails and changing live counts.

Tensor attention mode 5 and prefill mode 1 remain separate experiments. Prefill mode 1 changed the short trajectory before decode; disabling it restored prefill and eager-decode logits exactly. The tensor decode trajectory then first differed at the first graph forward. Diagnostic mode 6 retains paired outputs and operands from each eligible graph layer, substitutes the original outputs for continuation, and persists snapshots only for admitted complete-logit trials. A three-token trial isolates the first graph step. All three substituted 64-output trajectories reproduce complete baseline logits and registered state bit for bit. The independent replay covers 44 actual graph-layer cases (1,441,792 outputs per implementation): both paths pass the unchanged 2e-4 absolute threshold; maximum baseline/candidate errors are 1.48e-5/1.03e-5. At the first graph step, layer 3 value projection changes 40 of 16,384 BF16 values. Its first changed FP32 pair, 2.1762400e-5 versus 2.1717193e-5, straddles the BF16 midpoint 2.1755695e-5. Native projection replay and independent round-to-nearest-even checks reproduce that boundary. This is arithmetic reassociation propagating through a typed boundary, without a demonstrated packing or dispatch defect. Preserving the original reduction chain in mode 3 retains a useful gain and complete relative equality; the faster tensor paths remain unqualified. Their failed receipts and original thresholds are retained.

## Operator evidence

RTX A6000, SM86; unchanged complete-output absolute tolerance 2e-4:

- One actual 32K-history row, 64 heads and 2,051 selections: staged tensor attention about 0.072 ms versus 0.809 ms; maximum error against independent float64 packed equations 3.90e-6. Eager and graph outputs agree.
- Twelve derived selection and masking cases cover zero history, tails, repeated indices, invalid indices and capacities through 4,096. All complete outputs pass. Stress timings overlapped compilation and are excluded.
- A complete 18,000-row operand bank checks 589,824,000 outputs. All pass; maximum error 1.30e-5. The batched prefill-attention prototype is not promoted because its useful speed gain is modest and its largest tile exceeds current workspace.
- Native latent-query absorption reconstruction matches all 589,824,000 saved values bitwise. The tensor candidate takes 13.443 ms versus 158.763 ms, maximum difference 5.72e-6.
- The exact score/unpacked-value variant is bit-identical in all 13 tested complete eager/graph cases, including actual 65K operands, and about 2.11 times faster on the original captured row. A shared-head tile and a third TF32 query part were slower and remain excluded.
- Cooperative HQQ packed-word loading produces identical complete outputs but is 20–30% slower on the tested real projections. It remains excluded.

The ordered prefill tile matches all 589,824,000 large-bank outputs and tested row tails bitwise in eager and graph execution. Its 32-row tile takes 63.167 ms versus 161.759 ms. A head-coherent scalar traversal is also exact but slower than the tiled variant.

These measurements are operator evidence, not complete-request speed claims or complete-model quality qualification.

## Complete-request measurements

Frozen native inputs contain 3,084, 16,397 and 65,549 tokens. Each request generates 64 outputs. GLM uses K6V6 cache, a 69,632-row GPU window, 131,072-token allocation, HQQ6 dense attention, routed INT4 and shared/head INT8. The 65K chunk plan is `[32768,32768,13]`; attention workspace remains 172,212,989 bytes. The short labels below describe the retained workload classes, rather than exact token counts.

Two repetitions per ordinary text workload, medians in seconds:

| Workload | Baseline complete | Candidate complete | Baseline first content | Candidate first content |
|---|---:|---:|---:|---:|
| 3K chat | 20.206 | 20.482 | 10.649 | 10.809 |
| 3K code | 19.820 | 20.242 | 10.635 | 10.865 |
| 16K chat | 39.478 | 39.042 | 30.014 | 29.464 |
| 16K code | 39.268 | 38.823 | 30.067 | 29.454 |
| 65K chat | 123.509 | 119.638 | 113.938 | 110.137 |
| 65K code | 123.571 | 119.964 | 113.971 | 110.284 |

The long complete requests improve about 3%; intermediate requests improve about 1%. Short requests regress across these separate process runs. Ordinary decode remains mixed at roughly 146–154 ms/token; these data do not establish an end-to-end decode speedup.

Separate profiling, steady 61 forced forwards at 65K history:

| Scope | Baseline ms/forward | Candidate ms/forward |
|---|---:|---:|
| Attention kernels | 8.478 | 4.284 |
| Index selection kernels | 2.406 | 0.531 |
| Index score kernel | 2.029 | 2.029 |
| Kernel-only intervals | 52.202 | 46.091 |
| Copy-only intervals | 91.346 | 96.660 |
| Kernel/copy overlap | 6.870 | 6.924 |
| Uncovered intervals | 3.145 | 3.089 |
| Reconciled wall | 153.563 | 152.764 |

Named kernel sums are diagnostic scopes and must not be added to the disjoint rows. H2D payload drops from 2.038 to 2.005 decimal GB/forward, while its measured duration rises from 91.689 to 97.184 ms. Different transfer rates offset compute savings across processes. Same-process ABBA controls isolate decode attention: original 154.132/153.998 ms per forward, exact 150.230/149.617 ms, medians 154.065 versus 149.924 ms (2.69% less time). Complete-request medians are 119.396 versus 119.285 seconds because both arms retain the faster prefill/index profile. Each receipt verifies the actual graph execution key; mode changes force graph recapture. Expert placement remains adaptive. A separate same-process profile gives 155.158 versus 151.913 ms per steady forward: kernel-only 50.092→46.044, copy-only 94.928→95.813, overlap 6.879→6.842 and uncovered 3.260→3.214 ms. Attention kernels fall from 8.350 to 4.264 ms. H2D bandwidth is effectively identical at 21.1846 versus 21.1823 decimal GB/s; candidate payload is 0.82% larger because placement remains adaptive. Thus this control confirms a decode gain while preserving the cross-process transfer-rate limitation. No new residency policy is included.

Ordered prefill absorption falls from 7.397 to 2.858 seconds in the 65K trace. Through first content, kernel-only intervals fall from 95.605 to 90.762 seconds; copy-only time rises from 13.067 to 14.118 seconds. H2D payload through first content is effectively unchanged, 341.579 versus 341.567 GiB. This supports a compute improvement rather than a cache-representation or transfer-volume intervention.

Dense projections now account for about 19.4 ms of named compute scopes, routed experts about 13.35 ms, attention 4.28 ms and index scoring 2.03 ms. Transfers dominate the request critical path. The earlier 0.40 ms attention objective has not been achieved by the exact path. Cooperative and aligned packed-weight loading variants pass complete output comparisons but regress real projection speed and are excluded. Further attention layout sweeps retain the selected exact variant; sharing head groups is slower even with concurrent transfers.

## Correctness and limits

All 29,736,960 logits from three complete forced trajectories match the baseline bitwise. Three final registered states and two prefix snapshots total 2,495,217,216 bytes, all bit-identical. All 12 ordinary response texts match. Continuation, RAM/session restoration, cancellation and reset pass. Sampled device-wide free memory reaches a minimum of 926 MiB for the baseline and 928 MiB for the candidate, above the 600 MiB floor.

The unchanged independent witness still disagrees at the first decision in both arms, with the same log-probability difference. Relative equality does not clear that inherited absolute model-quality limitation. Only SM86 A6000 hardware was exercised. Capability and geometry dispatch retain fallbacks; other Ampere and newer cards have not been measured in this campaign.

The active GLM sparse path has zero rotary dimensions. Review did not identify a required half-split rotary ownership import for these changes; released unrelated model changes were not merged.

## Request protocol

Immutable controllers retain source and binary identities, frozen native input IDs, 64-token forced continuations, configuration, cache state, chunk plan and memory samples. Short, intermediate and 65K histories have repeated ordinary requests, followed separately by profiling and complete-logit/state diagnostics. The logical cache is cold; expert placement remains warm and adaptive. Both arms disable the CPU expert tier, so these results must not be mixed with earlier CPU-assisted measurements.

The device-wide reserve remains 600 MiB. CUDA kernel elapsed sums are investigation scopes; disjoint kernel-only, copy-only, overlap and uncovered intervals reconcile against wall time. Full logits and registered states are compared without declaring bit differences equivalent or relaxing existing gates. Inherited absolute model-quality failures remain separate.

A startup stall in inherited CPU-tier construction and an initial diagnostic-fixture admission failure produced no scored request. Failed receipts are retained. Protected refs, installed binaries and prior evidence remain preserved.
