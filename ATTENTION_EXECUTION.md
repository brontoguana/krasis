# Sparse attention execution

The supported exact execution profile is now the cumulative default. Explicit controls remain available for diagnosis; tensor arithmetic is under separate quality qualification. Historical measurements below retain their original control settings. Independent absolute model qualification is not established. Follow the Krasis Optimisation Guide under Performance Qualification in Lore.

## Contract and ownership

The K6 packed cache and selected token indices are unchanged. Sparse attention unpacks existing BF16 scales and signed six-bit codes into FP32 once per selected row, then shares that data across heads. On Ampere and newer devices, the tensor candidate decomposes both operands into TF32 high and residual parts and accumulates all four products in FP32. Scores use 128-feature accumulation bands; values use 128-selection splits. Softmax masks the actual live indices, including sentinel tails.

`src/optimisations.rs` is the compile-time registry. `EXACT_ATTENTION` selects the supported exact score/unpacked-value path; disabling it restores ordinary execution. `TENSOR_DECODE_ATTENTION` selects tensor geometry (512 latent dimensions, zero rotary dimensions, head count divisible by 16, at most 4,096 selections) and remains off pending fast-tier qualification. Capability and geometry fallbacks remain explicit. Build with `./dev build-variant EXACT_ATTENTION` to disable an entry. Runtime environment selectors for adopted paths are retired; legacy diagnostic treatment overrides are rejected; captures and oracles remain available.

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

## Bounded FP32 selected-row prefill

`KRASIS_PREFILL_MLA_F32=1` opts into the next qualified relative improvement on the retained exact profile. Full request qualification is specific to RTX A6000 SM86, K6 values, 64 heads, 256 non-positional query dimensions, zero rotary dimensions, 512 latent dimensions, 2,051 selections and pedantic FP32 GEMMs. Other geometry retains the existing path. The measured profile also uses split mode 3, absorption mode 2, radix indexing, 32-row attention tiles, a 69,632-token GPU KV window and no CPU expert tier. Packed cache bytes, indices, causal masking and logical chunks remain unchanged.

Gather copies queries alongside selected K6 values and leaves score masking to softmax. Score GEMM uses beta zero; softmax applies the original valid-index/position predicate and warp reduction order. This removes score-mask initialization and its GEMM read/modify/write. Score and value contractions use owned FP32 Lt plans on full 32-row tiles; partial tiles retain classic GEMMs. Chunks below 32 rows retain the prior pipeline. There are still four stages and 22,539 launches per stage in the 65K trace.

Startup tests at most four discovered pedantic FP32 configurations per role on actual operands. It requires complete bitwise output equality to the classic contraction and a five-percent timing margin. Selected library version/configuration identities are bound into the heatmap. Plans are sealed after calibration and at every engine handoff; requests cannot start cold tuning. Real allocation borrows check context, extent, alignment and aliases. Engine teardown completes owned streams before releasing prepared descriptors and handles. Hot execution stays in Rust/CUDA.

The selected-row tile uses 172,212,224 bytes of its existing 172,212,989-byte arena. Library workspace is capped at 16 MiB borrowed from existing serial MoE scratch; both selected plans need zero workspace on the measured GPU. No additional explicit persistent GPU data allocation is introduced. Minimum externally sampled free VRAM across the complete run is **908 MiB**, above the **600 MiB** reserve; native request reports reach 925 MiB. These measurements include library initialization and lifecycle work.

### Complete requests

One loaded process, frozen native IDs, 64 outputs, logical KV cold and expert placement warm/adaptive. Timing is separate from tensor capture and profiling. The 65K chunk plan remains `[32768,32768,13]`.

| Nominal history (native input IDs) | Control complete seconds | Candidate complete seconds |
|---|---:|---:|
| 3,072 (3,084) | 21.594 / 20.354 | 19.752 / 19.646 |
| 16,384 (16,397) | 48.166 / 46.853 | 48.236 / 43.435 |
| 65,536 (65,549) | 119.152 / 119.168 | 112.981 / 112.990 |

65K medians are **119.160 → 112.985 seconds**, a **5.18%** reduction. Every timing is retained. Short prefill itself is 10.08/10.08 versus 9.76/9.79 seconds, with decode near 9.40 seconds in both arms. The first short control has extra time outside those printed prefill/decode intervals; its cause is not assigned. Consequently the 6.08% short HTTP median difference should not all be attributed to the operator. The prior separate-process short regression remains recorded above; the new paired short sequence does not reproduce a slowdown. Intermediate request variation remains substantial. Its separate profiles contain about 6.6 seconds of uncovered time in both arms, without establishing a causal explanation for ordinary-run variation. The feature remains opt-in.

Unforced streaming requests repeat in forward and reverse history order. All six satisfy the existing validity and native-ID checks, and all response texts equal the separate accepted baseline. Candidate medians:

| Nominal history | Complete seconds | First content seconds |
|---|---:|---:|
| 3,072 | 19.827 | 10.332 |
| 16,384 | 43.261 | 33.916 |
| 65,536 | 112.841 | 103.536 |

The separate accepted streaming baseline at 65K has medians 118.857 seconds complete and 109.534 seconds to first content. That process comparison supports the result but does not replace the paired control.

### Exposed pipeline work

Separate 65K trace, through the first native decode embedding rather than HTTP first content:

| Attention operation | Control seconds | Candidate seconds |
|---|---:|---:|
| Selected K6 gather and query staging | 7.271 | 6.038 |
| Scaled absorbed-query/selected-value score contraction | 10.639 | 7.732 |
| Mask and softmax | 2.096 | 2.579 |
| Weighted selected-value contraction | 11.562 | 9.163 |
| Pipeline interval union | 31.568 | 25.512 |

Actual same-stream sequences establish the operation mapping, including legacy small-chunk tails. Pipeline intervals have no overlap with other traced GPU work. Moving masking makes softmax slower, while gather and contractions provide a net **19.18%** pipeline improvement. These elapsed scopes are measurements, not additive promised savings.

Disjoint predecode seconds reconcile: kernel-only **90.885 → 84.953**, copy-only **13.494 → 13.492**, overlap **3.597 → 3.601**, uncovered **1.915 → 1.925**. H2D payload is 366,145,070,632 versus 366,145,070,828 bytes; D2D is 121,356,894,208 bytes in both arms. This supports a compute gain with effectively unchanged transfers. Profiles are excluded from ordinary latency scoring. The added 16K pair uses already-admitted trials in the same loaded profiling session, after ordinary paired timing, with independent reserve monitoring and no request overlap.

### Numerical, quality and lifecycle evidence

The saved 18,000-row bank checks all 589,824,000 outputs bitwise. Independent float64 packed equations cover capacities 1/17/129/2051/4096, masked/duplicate/future/sentinel selections, empty history, and 1/17/31-row query tails. Baseline/candidate and eager/graph fixture outputs agree bitwise across 24,248,320 comparisons. Runtime dispatch is narrower than these stress fixtures: only 2,051 selections is request-qualified. Six actual native captures check 196,608 values, maximum error 7.85e-6 at the unchanged 2e-4 absolute threshold. Native prefill is eager; decode graph behavior is retained.

All **192 complete logit vectors (29,736,960 values)** and **1,417,804,736 registered-state bytes** match both the same-process accepted path and prior accepted captures bitwise. Fixed-token NLL and KL changes are zero at every tested history. All four frozen free-generation response texts also equal the accepted baseline: JSON passes, source-only coding still **fails** through repeated self-correction/truncation, 18K retrieval passes and 18K coding passes. The inherited coding failure is retained. Independent absolute witness admission remains unpassed.

Long-prefill disconnect, concurrent-client isolation, subsequent cancellation/reset continuation and live session/RAM restoration pass their existing gates. Before/after cancellation witness tokens match. Restored-versus-fresh log-probability drift remains recorded; these lifecycle/token checks do not prove numerical restoration equivalence. A subsequent short request completes about **52.6 seconds after a long-prefill disconnect**, preserving the existing logical-chunk responsiveness limitation. Model shutdown exits normally and leaves no GPU owner.

A separate BF16 high/residual tensor candidate passed independent operator equations and improved request time, but introduced a frozen JSON-format failure where the accepted path passes. It is rejected and is absent from native serving dispatch. Its source patch, tools and failed receipts are preserved. Its changed floating-point values are not classified as an implementation defect merely for differing from the exact path.

After the measured cohort, final review normalizes boolean aliases, seals every engine handoff, narrows unsupported selection-count fallback and formats the new Rust module. Five pure dispatch tests and eight heatmap/identity tests pass; the final native imports successfully. The final rebuild has identical CUDA/PTX. The measured literal `1`/2,051-selection execution is unchanged; the complete model cohort was not repeated for these startup/fallback-only guards. Measured/final source and binary hashes plus the reversible guard delta are retained in the evidence files.

### Remaining costs and reproduction

Prefill still contains 25.51 seconds in this pipeline, about 17.85 seconds of routed-expert kernels, 12.78 seconds in a 22-call BF16 family, another dense family near 6.19 seconds, KDA near 4.05 seconds and index-score accumulation near 3.62 seconds. These are investigation scopes, not additive savings. Source/launch-order mapping places the 12.78-second family at indexer query projection between pooled-key construction and query RoPE. Its row-independent strided-batch path broadcasts weights for one token per batch. A bounded replay that shares weights across token rows is the next concrete lead; reassociated arithmetic requires independent equations and the same fixed-token/held-out quality gates. The other dense family is not assigned a specific operation without further evidence. Decode receives no new intervention here and remains transfer-dominated. The overall performance objective remains unfinished.

`tools/attention/prepare_prefill_pipeline.py` extracts the retained native gather equations and derives independent fixtures; `prepare_prefill_tails.py` adds query-tile boundaries. `qualify_prefill_f32.cu` checks full eager/graph outputs. `replay_prefill_lt.cu ... exact` reproduces the bounded FP32 algorithm replay, and `attribute_prefill.py` maps profiled launch sequences. `request_prefill.py` replays a checksum-admitted trial against an already loaded server; `check_prefill_likelihood.py`, `check_prefill_run.py` and `check_prefill_quality.py` reproduce distribution, state and frozen-oracle checks. The quality harness uses the cached Node image with network disabled. Controls and thresholds are preserved; no weights were downloaded.

See [measurements](tools/attention/evidence/prefill-f32.json), [source/plan identities](tools/attention/evidence/prefill-f32-source.json), [final guard delta](tools/attention/evidence/prefill-f32-final-guards.patch) and [rejected tensor source](tools/attention/evidence/rejected-bf16-native.patch). Protected commits, installed binaries and prior evidence remain intact. Nothing is released or pushed by this work.


## Fixed-row indexer query projection

`KRASIS_INDEXER_QUERY_TILE=1` opts into the next exact relative improvement on the accepted FP32 attention profile. The active operation is the indexer `wq_b` projection, not main MLA query absorption. At 65,549 native input IDs there are 22 large calls at layers 3,7,…,43 across two 32,768-row chunks, plus 11 calls for the 13-row tail. Each large call consumes BF16 `[32768,1536]` normalized Q-LoRA inputs and existing BF16 `[4096,1536]` weights, with BF16 output and `CUBLAS_COMPUTE_32F`. Native dispatch receipts confirm the control uses per-token strided-batch cuBLAS; the ordered projection bank is absent in this profile and retains priority when present.

The candidate submits fixed 64-row GEMMs, sharing weight tiles across token rows. It pads an internal partial tile in engine-owned storage and copies only live output rows back, all on the existing serial compute stream. Standalone chunks below 64 rows retain the original operation. This is important: the original library changes its reduction family for some small batch counts at real dimensions. Forcing 64-row arithmetic onto standalone 13/31-row captures changed 11/28 BF16 outputs. The narrowed candidate preserves all tested original batch behaviors; it does not claim to repair general rechunking invariance. Unsupported geometry or math mode retains the old path; devices below Ampere do not create the candidate. Only SM86 hardware is measured here.

The existing weight allocation is reused. Tail storage adds **720,896 bytes (704 KiB)**, allocated before calibration; hot execution stays in Rust/CUDA. The existing engine teardown completes streams before freeing storage. The flag is bound into heatmap treatment identity and cannot be ignored by compatibility policy. No cache precision, selection policy, HCS policy or default execution is changed.

### Measured complete requests

One loaded process, accepted FP32 attention enabled in both arms, frozen IDs and 64 forced outputs, logical KV cold, expert placement warm/adaptive, CPU tier off. Retained K6V6 settings use a 69,632-row GPU window, 131,072-token allocation and the same `[32768,32768,13]` long chunk plan. Ordinary timing precedes trace collection. Every request asserts that no operand capture was emitted. Real diagnostic targets remain admitted at a separate 65,536-token observation height and are exercised after timing.

| Native input IDs | Control seconds | Candidate seconds | Median reduction |
|---|---|---|---:|
| 3,084 | 20.101 / 19.604 | 19.100 / 19.028 | 3.97% |
| 16,397 | 36.936 / 36.962 | 33.840 / 33.831 | 8.43% |
| 65,549 | 112.813 / 112.715 | 100.241 / 100.230 | 11.11% |

The 65K medians are **112.764→100.236 seconds**. Decode remains near 150 ms/forward; it receives no new intervention. Short control variation is retained rather than discarded or wholly attributed to the projection. Earlier 16K timings matched operand observer targets; their extra time is not assigned to this operator, and those receipts remain unchanged. The accepted 65K timing receipts did not match such observers.

Six unforced streaming repetitions retain identical native IDs and response texts. Candidate median complete/first-content times are **19.130/9.713**, **33.770/24.446**, and **100.198/90.924 seconds** at the three histories. These secondary measurements occur after trace collection with adaptive expert placement; paired pre-profile timings establish the primary speed claim.

### Operator and trace evidence

Clean actual-operand replay: **580.686→12.085 ms (48.05×)** for a 32,768-row projection. Six saved layer/history banks plus nine row-boundary cases compare **606,683,136 outputs bitwise**. Eager and graph arithmetic replays agree. Native prefill remains eager; existing decode graph validation is retained. Independent float64 equations cover 2,740,224 projection values, including direct BF16 rounding diagnostics; baseline and candidate retain identical typed-boundary results. The expected differences from an ideal float64 contraction are not treated as new implementation errors.

All nine observed indexing boundaries match raw queries, rotated queries, head weights, complete causal scores and all selected indices. Independent score equations have maximum error **5.16e-5**, below the unchanged **2e-4** absolute threshold. Stable ordering, near-tied scores, chunk positions and masked sentinels are checked. The minimum observed cutoff margin is about **0.00120**; changed selections: **zero**. This profile has zero rotary dimensions. Three additional post-timing 65,536-token observations match the accepted complete normalized-input banks, weights, raw/rotated queries, causal scores and selections.

Separate 65K profiling maps source boundaries, actual dispatch receipts and library kernels:

| Scope | Control seconds | Candidate seconds |
|---|---:|---:|
| Indexer query kernels | 12.779 | 0.283 |
| Indexer query spans, including intervening gaps | 12.779 | 0.294 |
| Selected-row attention pipeline | 25.470 | 25.486 |

Disjoint predecode intervals reconcile independently: kernel-only **84.887→72.490**, copy-only **13.435→13.423**, overlap **3.576→3.570**, uncovered **1.932→1.893 seconds**. These total **103.830→91.377 seconds** through first native decode entry, rather than HTTP first content. Named operator scopes overlap the disjoint categories and are not additional savings. H2D payload is 366.249 versus 365.860 decimal GB; expert placement remains adaptive. D2D payload is identical at 121,356,894,208 bytes. No transfer policy change is claimed.

### Qualification and remaining work

All **192 complete logit vectors (29,736,960 values)** and **1,417,804,736 registered-state bytes** match both paired control and accepted captures bitwise. Fixed-token NLL/KL changes are zero. All four held-out texts match: constrained JSON, long retrieval and long coding pass; the inherited source-only coding failure remains identical. Independent witness first-token disagreement and selected-token logprob delta **2.8826026334** remain identical. These relative checks do not grant absolute model admission.

Concurrent-client isolation, continuation after cancellation/reset, active-session hits and RAM restoration pass existing gates. Session probability equality is not established; score drift remains recorded. Following a long-prefill disconnect, the next isolated short request completes after **46.25 seconds**, retaining the active-chunk responsiveness limit. Normal shutdown exits cleanly and leaves no GPU owner. Minimum free VRAM is **926 MiB externally sampled / 925 MiB natively reported**, above the **600 MiB** reserve.

The remaining attention pipeline costs **25.49 seconds**: value contraction 9.15, score contraction 7.72, gather 6.04 and softmax 2.58. Routed expert kernels cost **17.83 seconds** and are now the largest individual family. Another BF16 dense family costs **6.18 seconds** but remains semantically unmapped; KDA preparation/split costs 4.06 and index-score accumulation 3.65 seconds. These scopes are not additive promised savings. A bounded actual-operand routed-expert replay is the next concrete lead; the overall performance objective remains unfinished.

Grouped strided-64 arithmetic changed outputs and remains unselected. Those differences are not classified as an implementation defect; its optimistic incremental gain beyond the exact path is under 0.15 seconds per long request from isolated operator timings, which does not justify a separate arithmetic/quality campaign here. A full variable-row matrix call was fast on the large bank but is not promoted without chunk-size arithmetic qualification; the fixed64 candidate already removes almost all projection cost. The earlier rejected BF16 attention candidate remains rejected and untouched. Failed startup/preflight receipts and all prototype outputs remain retained.

`src/indexer_projection.rs` owns dispatch and tail storage. `replay_query_projection.cu` reproduces default exact replay; `--diagnostics`, `--force-tiny` and `--repeat-row` reproduce the separate probes. `check_query_equations.py` reconstructs BF16 equations; `check_query_index.py` checks causal scores/selections; `check_query_run.py` compares complete logits/state, chunk plans and observer-free paired timings. `attribute_query_projection.py` maps actual query work, and the existing attention attribution tool verifies unchanged pipeline work. Fifteen original-source teacher tests and nine heatmap identity tests pass.

See [complete measurements and identities](tools/attention/evidence/indexer-query.json). Protected refs, accepted evidence and installed binaries are preserved. No weights were downloaded; nothing was released or pushed.


## Experimental FP32 value-plan reassociation

`KRASIS_PREFILL_MLA_VALUE_REASSOC=1` is an **off-by-default experiment**, outside the accepted exact profile. It uses the existing FP32 attention path and 32-row tiles; the campaign keeps `KRASIS_PREFILL_MLA_F32=1` and `KRASIS_INDEXER_QUERY_TILE=1` enabled in both arms. Separate exact/candidate pedantic-FP32 plans are prepared before sealing. Complete finite cold outputs must satisfy the established 2e-4 operator gate, with a timing margin and preference for exact arithmetic within 5%. Tails, unsupported geometry and unavailable plans retain existing execution. No hot tuning, representation change or additional explicit GPU workspace is introduced. Exact-only heatmap identities keep their previous schema; the new treatment is explicit and cannot be ignored.

Same-process capture-free ABBA, frozen native IDs and 64 forced outputs:

| Nominal history | Exact seconds | Experimental seconds | Reduction |
|---|---:|---:|---:|
| 3,072 | 19.005 | 18.894 | 0.58% |
| 16,384 | 33.782 | 33.382 | 1.18% |
| 65,536 | 100.439 | 98.726 | 1.71% |

Native value-kernel time falls **9.113→7.286 seconds**, and the attention pipeline **25.431→23.808 seconds**. Copy-only time remains about 13.47 seconds. Independent 18K operator replay compares 589,824,000 outputs, maximum error 1.18e-5 under the unchanged 2e-4 gate, with eager/graph and tail checks. Against the active exact plan the full operator pipeline improves 6.2–6.6%; earlier classic-control figures are excluded from the incremental claim.

The exact control preserves all 192 logit vectors and 1.42 GB of registered state against accepted captures bitwise. The candidate changes states, logits and likelihoods. At the first modified layer, queries/cache/selections/weights match; a maximum 1.31e-6 output difference changes seven BF16 rounding decisions and 49 downstream WVC values. Both operators satisfy independent equations. This is arithmetic reassociation crossing a typed boundary, not a demonstrated packing or ownership defect. The first registered-state change is at layer 4. No full-continuation substitution proof or absolute model admission is claimed.

All four held-out pass/fail outcomes are preserved, including the source-only coding failure. Mean selected-token NLL changes are +0.05095/+0.00342/−0.03687 at the three histories. Witness first-decision failure remains, with selected-token discrepancy 2.88260→1.86038. The candidate stays experimental; the accepted profile retains exact value arithmetic. No narrow repair preserving the faster reduction and original typed-boundary outputs was demonstrated.

Minimum free VRAM is 926 MiB sampled/925 MiB native; added explicit workspace and host-weight copies are zero. Lifecycle/session checks pass; restoration probability equality remains unestablished. Long-prefill cancellation followed by an isolated short request takes 45.56 seconds, within the recorded 46.25 seconds. Shutdown exits normally and releases the GPU. Only SM86 hardware is measured. Fifteen teacher and eleven heatmap-policy tests pass. A final identity compatibility/safety correction is rebuilt; measured arithmetic/dispatch are unchanged and the full GPU cohort is not repeated for that metadata-only change.

Routed attribution maps 11.03 seconds gate/up and 6.80 seconds down. Actual full-prefix counts cover 22,020,096 assignments: only 0.86% have occupancy≤32, while 66.74% exceed 256. The exact-output block32 replay regresses representative larger occupancies and remains unselected; the whole tiny-tail family is only 73.7 ms. The vector-gather prototype is exact but offers only ~5% gather gain and remains outside serving dispatch. No further indexer micro-optimisation is included.

An initial added 2e-6 startup screen rejected 4.17e-6 error and selected exact arithmetic. That rejected startup remains preserved and receives no candidate speed claim; the subsequent screening-policy correction uses the established 2e-4 operator gate. Initial classic-control overstatement, parser/build/test failures and invalid attribution-analysis receipts remain retained. Ordinary streaming first-content data are secondary comparisons against a prior process; primary complete-request results use the same-process pair. The overall performance objective remains unfinished. See [measurements and limits](tools/attention/evidence/value-plan.json).


## Natural routed gate/up screening

The bounded N128 alternative is **rejected**. One actual layer 3 band contains 8,192 distinct token rows, 288 expert counts and original sorted routing. The retained eight selected experts have 119–462 assigned slots each: 1,879 slots total. All their natural operands and exact packed INT4/BF16 group-128 weights are used without repeating rows. M64, K64, full-K reduction ownership, expert/slot order and the 84-block grid stay fixed. This selected-expert replay is not a full-band/request benchmark and does not exhaust larger occupancies or other routed opportunities.

Repeated paired offline gate/up medians are **0.844 ms active N256**, **0.928 ms separately compiled N256**, and **1.319 ms N128**. N128 is 56.3% slower than active execution and 42.1% slower than the same-build N256 control. All 7,696,384 compared projection values match bitwise; all 32,768 values of the captured native producer row match both replay arms. No tolerance changes or full-model admission follows from this relative projection result.

Both variants consume 255 registers/thread and have 72-byte local frames. Shared memory drops 71,680→54,272 bytes but occupancy remains one block/SM. N128 halves resident warps and doubles N tiles; logical A tile-load volume increases 264→528 MiB for this selected workload, while packed-weight tile volume remains 264 MiB before caching. These are source-derived requested volumes, not measured DRAM bytes. Counter collection was blocked by GPU counter permissions. No reduction partition change is attempted; prior full-K ownership remains.

No slower candidate advances to activation/down/scatter qualification, independent packed-equation admission, model trajectories, likelihoods, held-out quality or lifecycle reruns. Serving code/binaries and controls are unchanged; value reassociation remains disabled. The accepted approximately 100.24-second 65K request baseline and its inherited coding/witness and restoration-score limits remain in force. Added serving workspace is zero; offline replay retains over 45 GiB free. Cancellation latency is not remeasured; prior evidence remains 45.56–46.25 seconds.

The retained exact-profile dense family is now mapped completely: 334 launches / 6.156 seconds. KDA Q/K/V contribute 3.556 seconds, KDA output 1.081 seconds, MLA query expansion/output 1.035 seconds, and dense MLP gate/up/down 0.485 seconds. HQQ6 projections materialize the same BF16 weights before `cublasGemmEx`: BF16 inputs/outputs with FP32 compute and default algorithm. Separate nonlinear kernels preserve activation boundaries; the generic compiled kernel's relu name does not imply a fused model activation. The subsequent headroom assessment below supersedes the proposed KDA dispatch experiment. See `tools/attention/replay_routed_n.{cu,py}` and `attribute_dense.py`.


## Register and bandwidth headroom

The mapped Q/K/V work sustains 126.15 TFLOP/s; an independent same-shape resident BF16 probe sustains 126.13 TFLOP/s. Duration alone supplies no useful dispatch headroom, so no KDA GEMM tuning was started.

A 512-thread M64/N256/K64 template is unsupported: its one B-fetch iteration cannot advance the existing even-length pipeline. Preserve its failure without treating its timing as a valid performance result. The supported 192-register cap retains 256 threads, all tile dimensions, complete-K ownership, pipeline and warp mapping. Local frame grows 72→760 bytes; shared memory remains 71,680 bytes and occupancy remains one block/SM. With the original eight natural experts, active / same-build / capped medians are 0.843/0.926/3.901 ms. Adding a naturally hotter691-token expert gives 1.105/1.202/5.236 ms with all 10,526,720 outputs bit-identical. The cap is rejected before downstream/model qualification. Spill reports are compiler/static-SASS evidence, not measured DRAM traffic.

A separate cooperative half-warp K6 loader preserves scalar FP32 reconstruction and output order. All 19,491,840,000 large-bank values match; independent masked/tail fixtures and complete small-graph outputs pass. Eager/graph times remain 138.05/137.48 ms for both arms. Reject it as flat; no model recalibration or request rerun follows. Gather emits approximately 3.123 TB of logical FP32 outputs over the 65K profile; measured write-only throughput gives an optimistic 4.40-second floor versus 6.03 seconds observed, before reads/instructions. Do not treat that difference as recoverable request time.

New demonstrated request savings: zero. The next structural research priority is score/value intermediate materialization and actual-shape pedantic FP32 throughput, followed by routed accumulator/shared-memory ownership. Gather-only tweaks and already near-ceiling BF16 GEMM dispatch are lower priority. No new serving workspace or arithmetic change; value reassociation stays disabled. Inherited model-quality and cancellation limits remain. Tools are `replay_routed_threads.{cu,py}`, `attribute_spills.py` and `replay_gather_halfwarp.cu`; retained evidence is under `krasis-internal/routed-registers-237355/`.


## Intermediate materialization screen

Preserve `29baba8` and earlier failed evidence. The accepted full-batch score/value plan identities are independently reproduced from the actual device UUID, library version, algorithm bytes and geometry. At batch32, 64 heads, 512 features and 2,051 selections, resident pedantic FP32 scores/values measure **0.325/0.362 ms** (13.23/11.87 TFLOP/s). Gather and softmax measure **0.257/0.118 ms**. Full/resident pipeline medians are **1.068/0.824 ms**; their 0.244 ms difference removes all gather work, including query copying, and is an intervention bound rather than achievable fused execution. No BF16 ceiling is used.

The materialized selection tensor is **134,414,336 bytes per 32 queries**. Its single logical write plus one logical read by each contraction totals403,243,008 bytes. Actual DRAM transactions and exposed request time are not inferred from this source accounting. Both full pipelines have four launches. The earlier65K gather scope writes approximately 3.123 TB logically; not all of that materialization can disappear while only one contraction changes.

Screen one structural intervention: a CTA owns32 heads by 64 output features, unpacks32 selected K6 rows directly into12 KiB shared storage, shares probabilities and unpacked features across its outputs, and retains an ascending FP32 FMA chain. This bypasses the value contraction's global dense reload. Scores still require the dense materialization, so its allocation and gather remain. No softmax, mask, selection, precision or output boundary changes. The kernel uses256 threads,48 registers, no spills and five active CTAs/SM on the measured device. SM80 compilation passes; only SM86 hardware is executed.

The direct value contraction takes **0.832 ms**. Same-process ABBA full-pipeline medians regress **1.069→1.498 ms**; graph medians regress **1.047→1.421 ms**. Reject before request startup or calibration. Lower global reload traffic has not compensated for contraction-local unpacking, shared operations and synchronization. No attribution assigns the regression exclusively to one of those costs.

All 17 independent fixtures pass complete scores, probabilities and outputs at the unchanged 2e-4 absolute threshold: 3,997,696 output values, distinct natural32/33-row query batches, actual layers 3/23/43 at 3K/16K/65K histories, capacities1/17/129/2051/4096, masked/invalid/empty selections, ties and near ties. Natural32 candidate/control outputs are bit-identical. Genuine batch1 fallbacks use another library reduction; maximum candidate/control difference across the suite is6.92e-6. Maximum error against independent FP64 packed equations is9.28e-5. Eager/graph outputs agree bitwise at matching geometry. Initial checker and graph-tail harness failures remain preserved with their repairs; they do not establish a serving defect.

There is zero added serving workspace and no new global candidate workspace; shared storage is12 KiB/CTA. The existing172,212,989-byte attention allocation remains unchanged. This rejected offline candidate adds no request, likelihood, held-out, continuation or lifecycle admission. The approximately 100.24-second accepted65K baseline,600 MiB reserve policy and recorded 45.56–46.25-second cancellation limits remain unchanged. Independent absolute-quality limitations remain unresolved.

Next structural work would need higher-throughput contraction-local reuse or output ownership before it could profitably remove more materialization. Both contractions would have to consume local packed-cache tiles to remove the common dense buffer entirely. Current evidence demonstrates zero recoverable request time from this intervention; the 7.70/9.11-second score/value and 11.03/6.79-second routed gate/up/down scopes remain research leads, not additive savings. See `krasis-internal/attention-materialization-237377/RESULTS.md` and reproducible tools in `tools/attention/`.


## Contraction-local tensor value screen

A bounded 32-head by 64-feature shared tile unpacks 128 K6 selections per band and uses the retained four-product TF32 high/residual arithmetic. All correction products remain. Shared workspace is 48 KiB/CTA, 56 registers/thread, no spills and no global partial-output storage. SM80 compilation passes; only SM86 is measured. Accepted scores, softmax and indices remain unchanged.

Reject: actual accepted value timing 0.364 ms versus 1.321 ms; paired full pipeline 1.066→1.976 ms, graph 1.026→1.850 ms. An intrusive CTA-cycle probe observes 66.11% operand preparation, 33.77% fragment conversion/shared-load/contraction and about 0.12% explicit barrier waits combined; these are observer fractions, not wall-time or hardware stall-counter attribution. Preparation dominates this ownership scheme despite tensor arithmetic.

All 17 independent packed-equation fixtures and 3,997,696 outputs pass unchanged 2e-4, including actual layer/history operands, natural 32/33-row batches, masks, duplicates, tails and near ties. Candidate eager/graph outputs agree bitwise; accepted scores/probabilities/value outputs match previous exact controls bitwise. Candidate reassociation is separately tolerance-checked and has no whole-model quality qualification. No model startup or request gain follows. The accepted approximately 100.24-second 65K baseline, exact attention/indexer profile, disabled value reassociation, 600 MiB reserve and inherited quality/cancellation limits remain unchanged. Added serving/global candidate workspace is zero. See `krasis-internal/attention-local-tensor-237394/RESULTS.md` and `tools/attention/replay_local_tensor{,_probe}.cu`.


## Scale-factored K6 value screen

One offline tile owns all64 heads and a16-feature BF16 scale group, with128-selection bands. Codes -32..31 are exact TF32; splitting FP32 probability-times-scale needs two MMA products. This reassociates arithmetic without changing K6 bytes. Shared workspace is41,472 bytes/CTA,54 registers/thread, no spills; SM80 compilation passes, SM86 execution only. Compared with the rejected four-product tile, logical unpacking and MMA work halve, while probability staging increases fourfold and total staged words increase about69%. No bank, barrier or loader sweep.

Reject: accepted/candidate values0.364/1.442ms; paired full pipeline1.050/2.036ms; graph1.028/1.943ms. All required scaling, unpacking, fragment conversion, synchronization and accumulation are timed. Score gather remains common. All17 fixtures and3,997,696 outputs pass the unchanged2e-4 independent packed equations; eager/graph outputs agree. Accepted arrays remain bit-identical to retained controls. Operator tolerance does not establish model quality.

Exhaustive host checks show all4,118,172 finite BF16-scale/code products are exact in FP32;59,748 overflow combinations demonstrate that arbitrary finite scale is not enough. Factoring can also underflow probability-times-scale. Actual retained scales/probabilities stay in normal finite range; zero probabilities remain zero. Exactness of integer codes does not make a two-part probability or reassociated reduction exact. No model startup or new request gain; accepted approximately100.24-second65K profile, disabled value reassociation,600MiB reserve and inherited quality/cancellation limits are unchanged. Details: `krasis-internal/attention-scale-factor-237409/RESULTS.md`, `tools/attention/replay_scale_factor.cu`, `check_scale_factor.py`.


## Globally prepared four-product TF32 screen

Selected K6 rows and queries are split once per physical tile into globally owned high/residual components, shared across64 heads and both contractions. Four FP32-accumulated cuBLAS products retain HH,LH,HL,LL. Masked softmax runs in place; probabilities split once.16 physical queries fit159,604,736 bytes including the complete32-row output, within172,212,989 bytes;4096 selections use8 physical rows. Existing16MiB serial library workspace remains separate. No source/licence notices are removed or imported. This differs from earlier CTA-local TF32 conversion and the rejected three-product BF16 approximation.

Reject: accepted/candidate complete32-query pipeline1.038/2.789ms; graphs1.030/2.750ms. All preparation, eight GEMMs per physical tile, beta accumulation, writes and launches are charged. Shared conversion removes duplicate preparation but doubled component traffic and four products lose materially. Separate profiling confirms tensor-core dispatch. No parameter/loader sweep or model startup follows.

All17 independent fixtures and3,997,696 outputs pass unchanged2e-4, including complete intermediates, masks, near ties, tails and complete eager/graph output checks. Accepted controls remain bit-identical. Retained ranges are safe; extreme finite FP32/subnormal witnesses limit universal split equivalence. No model quality or new request gain. Accepted exact attention/indexer profile, disabled value reassociation, approximately100.24s65K baseline,600MiB reserve and inherited failures/cancellation limits stay unchanged. Source: tools/attention/replay_prepared.cu,check_prepared.py,check_prepared_range.py. Evidence: krasis-internal/attention-prepared-237423/RESULTS.md. Remaining measured exposure ranks routed ownership17.82s, exact contractions16.81s, gather6.03s; no recovery is demonstrated, and gather's optimistic stage bound is under1.64s before reads/instructions.

## Routed row-ownership feasibility

No specialization built: the pre-build headroom gate does not pass. On the natural eight-expert band and the same band plus expert252 (691 slots), active packed gate/up takes0.853/1.122ms versus resident BF16 useful-row0.884/1.139ms and padded0.901/1.167ms. The second BF16 weight layout also loses. Active down0.545/0.710ms closely matches resident useful0.538/0.712ms. Rebuilt unchanged controls are separately0.932/1.211ms gate/up and0.593/0.763ms down. Complete active/rebuilt outputs and complete eager/graph outputs match across36,446,208 BF16 values; down uses explicitly regenerated identical activation operands, not a full native activation capture.

Four N groups/two M groups would halve accumulators128→64 and A fragments32→16 registers/thread, requiring separate physical-copy/compute iterations, replicated packed-weight preparation, changed A/B shared addressing and disjoint row epilogues. Weight preparation doubles;71,680 shared bytes keep occupancy at one CTA/SM. Removing128KiB logical K-reduction traffic/tile is plausible, but no substantial diagnostic throughput gap pays for the extra preparation. No compiler resources or candidate numerical result is claimed.

The retained65,536-ID prefix computes738.872/845.441TFLOPs useful/padded gate/up and369.436/422.720TFLOPs down, at76.99/62.44 padded TFLOP/s. Resident BF16 is a diagnostic, not a universal bound or deployable speed claim; CPU reconstruction/upload are excluded and separately recorded. No serving changes, startup or request gain. Accepted exact profile, disabled value reassociation, approximately100.24s65K baseline and inherited quality/restoration/cancellation limits remain. Details and pre-compilation model: `krasis-internal/routed-row-ownership-237444/RESULTS.md`, `PLAN.md`; tools: `measure_routed_headroom.py`, `replay_routed_controls.py`.

## FP32 index-score row bands

An experimental `KRASIS_PREFILL_INDEX_SCORE_ROW_BANDS=1` splits physical score rows inside the existing logical tile so eight heads fit each main temporary batch. Retain ascending FP32 head accumulation, BF16 query/key/head-weight operands, FP32 score temporaries, causal masks and logical chunks. Original scheduling remains below 64 rows, for all-head batches and for BF16/causal-band score modes. No new native CUDA kernel or serving allocation is introduced; default is off.

The 65K score-to-index GPU work, including common query staging and raw-index expansion, falls 7.426→5.749 seconds (22.6%). Accumulation falls 3.646→1.479 seconds while smaller GEMMs rise 2.409→2.931 seconds. Logical final-score read/write traffic falls 1.489→0.166 TB; temporary traffic remains. Same-process ordinary ABBA requests observe 99.853→98.225 seconds (1.63%); transfers are identical and disjoint kernel-only time falls by 1.629 seconds. These observations are separate from strict request qualification.

Complete independent equations, graph/eager score/indices, 192 full primary logit vectors and 1,417,804,736 registered bytes pass unchanged gates and match preserved controls bitwise. Static-policy long numerical controls also pass. Held-out results retain the inherited coding failure; witness and restoration-probability limitations remain. Active/RAM cache token checks, concurrent clients, reset and continuation pass. Cancellation follow-up is 45.44 seconds; minimum observed free is 926 MiB externally / 925 MiB natively, above 600 MiB.

No strict same-placement request gain is promoted. The residency observer re-enters the diagnostic endpoint and refills 2,715 experts versus 2,700 after ordinary long chat; full-history probes retain that mismatch. Failed receipts remain. This observer is not passive, and its allocation/admission discrepancy is not isolated enough to waive the gate. Keep the candidate opt-in and value reassociation disabled. Before further request-speed qualification, collect passive host-only placement receipts on the normal request path. No additional calibration campaign was started to force acceptance. See `tools/attention/evidence/index-score-bands.json` and `krasis-internal/index-score-bands-237463/RESULTS.md`.
