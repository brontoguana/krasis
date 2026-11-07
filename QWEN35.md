# Bounded Qwen3.5-27B text and images

The normal 27B development profile uses **INT4 language weights**, a single pinned
host owner, bounded GPU streaming and the native BF16 vision encoder. Model
execution and scheduling run in Rust/CUDA. Python handles startup conversion,
image processing and request bookkeeping. Original local safetensors are required;
startup does not download model weights.

## Current profile

The launcher and server select INT4 attention, dense MLP and output head, with
BF16 KV and vision. The supported default geometry is hidden size 5120, 64 layers
and intermediate size 17408. Other model families retain their precision policy.

- 4,096 tokens including images and output; 256-token prefill bands; full causal
  attention and recurrent chronology; 256 MiB BF16 KV.
- One image up to 512×512, at most 1,024 encoder patches / 256 image tokens.
- Ampere or newer NVIDIA GPU. Measured with independent device-wide allocations
  on an A6000 at 4, 6 and 8 GiB, including loading, warmup, vision and generation.
  These are constrained-A6000 results; actual small-card hardware remains untested.
- The measured minimum free VRAM is above the **600 MiB reserve**. Context overflow
  returns HTTP 413. Multi-GPU and speculative/batched verification are unsupported.

The INT4 representation uses symmetric group-128 quantisation, Marlin layout and
BF16 scales. Conversion reads and releases checkpoint sources incrementally.
Versioned, checkpoint-bound, checksummed artifacts publish atomically; incomplete
or incompatible entries are rebuilt. Native private file mappings are pinned once;
there is no retained complete BF16 language model or second pinned INT4 model.
Clean OS file-cache pages are separate, reclaimable memory, not a second serving
weight owner. Shape-only Python tensors contain no large weight storage.

Physical pinned text backing is **15,741,952,000 bytes (14.66 GiB)**, including
higher-precision exceptions. Measured cold conversion peaks at **4.36 GiB RSS**.
Warm core serving uses roughly **16 GiB**, rising to approximately **19–20 GiB**
with native vision and the exercised session cache. Process RSS/PSS, anonymous/file
pages, system memory and GPU sampling are retained in the results record. Allow
**32 GiB system RAM**; startup checks available RAM and cgroup v2 limits. Session
cache growth has its own budget. A physical 24/32 GiB host limit was not tested.

Higher-precision exceptions are explicit:

| Owner | Representation and footprint |
|---|---|
| Embedding access | BF16, 2,542,796,800 host bytes; fetch requested rows from mapped host memory rather than streaming the entire table |
| Small recurrent B/A projections | BF16, about 45 MiB GPU; their 96-row shape is outside the tensor-core packing contract |
| Norms, convolution and state parameters | Existing prescribed precision; approximately 6 MiB additional resident parameters |
| Native vision weights | BF16, 921,460,192 host bytes; one block at a time in an 89,677,312-byte GPU slab |

All large MLP, full-attention/recurrent projections and the output head stream
packed bytes and scales directly. There are no full BF16 weight expansions on the
normal execution path. Separate QKV/Z and B/A contractions, repaired norms,
rotary boundaries, softmax and recurrence remain intact. Packed head results are
explicitly converted to FP32 logit storage before sampling.

The joint device planner protects resident weights before sizing the cold slab.
With the head protected, a **197,345,280-byte** slab suffices instead of funding a
head-sized slab separately. Freed space funds additional resident weights while
preserving the combined allocation. Defaults protect about 2.43 GiB at 6 GiB and
4.43 GiB at 8 GiB. Residency automatically scales down for smaller allowances;
the 4 GiB test uses no resident pool. Explicit resident budgets are respected.
Pointers and copy ownership remain fixed across eager and graph execution.

Eligible BF16-output INT4 decode contractions use the existing Marlin tensor-core
program with fixed reduction/lock workspace. FP32-producing projections keep
their separate dispatch. Same-input operator comparisons and complete-request
measurements determine promotion; no full-model benchmark depends on proof writes
or per-tile synchronization.

On identical arithmetic and coding workloads, the preserved BF16 control measures
about **0.44 tok/s**. The current INT4 path measures about **2.0 tok/s at 6 GiB** and
**2.5 tok/s at 8 GiB** (about 1.7 tok/s at 4 GiB). These remain PCIe-bound streaming rates, not resident-model
throughput. First-content and whole-request timings, cold vision costs, actual
copy bytes and timing overhead controls are recorded in
`krasis-internal/qwen27b-int4-235939/RESULTS.md`.

## Run from the development checkout

Use complete local tokenizer, processor and safetensors assets:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 KRASIS_QWEN_GPU_LIMIT_MIB=8192 \
./dev python -m krasis.server \
  --model-path /path/to/local/Qwen3.5-27B --num-gpus 1 \
  --kv-cache-mb 256 --max-context-tokens 4096 --no-enable-thinking
```

The normal launcher applies the same INT4 defaults and joint budget. Precision
can be stated explicitly with `--attention-quant int4 --dense-mlp-quant int4
--lm-head-quant int4 --kv-dtype bf16 --vision-quant bf16`. The native vision tower
retains BF16 regardless of the language quantisation.

`KRASIS_QWEN_GPU_LIMIT_MIB` controls admission, not physical enforcement.
`KRASIS_QWEN_RESIDENT_MIB` sets an explicit resident-weight budget;
`KRASIS_QWEN_INT4_CACHE` selects the cache root. Set
`KRASIS_QWEN_INT4_TENSORCORE_DECODE=0` to retain the scalar Marlin GEMV control.
`KRASIS_QWEN_DENSE_STREAM=0` exits this bounded profile and requires the resident
path's own supported precision and substantially larger VRAM budget.

BF16 remains an explicit numerical reference: pass `--attention-quant bf16
--dense-mlp-quant bf16 --lm-head-quant bf16 --kv-dtype bf16`. That legacy reference
retains source and pinned BF16 ownership and needs roughly 110–113 GiB host RAM;
it is not the default INT4 implementation.

## INT4 evidence and scope

Complete independent inverse-layout GEMV and 65-row batched GEMM checks pass at
unchanged `atol=.03, rtol=.01`, including the complete vocabulary head and both
recurrent projection output types. Earlier arithmetic/is_even smoke, square/rectangular images,
image-dependent follow-up, reset, cancellation, cached attention and RAM session
restoration pass. Changing residency preserves all 128 prefix-state arrays and
all logits bitwise. Graph execution is active, with stable weight pointers.

Numerical evidence is kept distinct: the preserved BF16 reference passes its
complete fixed 64-layer gate. The original INT4 reference uses different
projection reductions, an FP32 head and BF16 cache storage for recurrent states.
A bounded investigation checks all 496 real projection contractions (400 INT4
contractions and 96 BF16 exceptions) against independent equations: all
109,312,000 output values pass unchanged
`atol=.03, rtol=.01`. Substituting native projection outputs into the reference
makes every later projection input and all 128 outgoing state arrays bit-identical;
complete logits pass. This localises accumulated differences to contraction
rounding, rather than a demonstrated packing or recurrence defect.

The unconditioned whole-model backend comparison still fails, including after
matching the packed equation and precision boundaries. The conditioned replay is
a causal diagnostic, not an independent whole-model admission pass. All original
failures remain preserved. Bounded task tests and operator agreement do not
establish canonical ADQ/release admission or general quantised quality. A bounded
global-dedup coding task fails identically in INT4 and preserved BF16; filtering,
arithmetic and image-shape tasks pass in both profiles. See
`krasis-internal/qwen27b-int4-correctness-236009/RESULTS.md` for contracts, causal
controls, practical coding limitations and the repaired diagnostic stream wait.

## Arête compatibility and qualification limits

[Arête's published config](https://huggingface.co/NousResearch/Arete-27B/blob/main/config.json)
uses the supported dense language geometry. Its `output_gate_type: swish`
selects the **GDN recurrent output norm gate**, where Swish/SiLU is
`z * sigmoid(z)`. The independent
[vLLM implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)
normalises `swish` to `silu`. Krasis already implements this equation. The
separate full-attention output gate remains sigmoid. Unsupported recurrent gate
variants now fail explicitly.

Arête and the local Qwen checkpoint list the same 333 vision tensor keys and
compatible processor/encoder geometry. Only public configuration, processor and
index metadata were inspected. **No Arête weights were downloaded or executed.**
This establishes configuration/equation compatibility, not an Arête model verdict.

Functional serving and memory fit are separate from numerical qualification.
The preserved BF16 causal repair follow-up clears the fixed 28-token, complete 64-layer text
comparison: zero failing logits and zero failing arrays among all 128 sequence
state arrays at unchanged `atol=.03, rtol=.01`. Original, square and rectangular
complete encoders match their frozen reference features bitwise. The original
failed identities remain preserved; see the causal report for substitutions,
reference repeatability, operator checks and the exact measured scope.
These fixed-fixture passes do not establish general model quality, canonical
ADQ/release admission, Arête execution or other-GPU qualification.

Evidence and immutable controllers:
`krasis-internal/qwen27b-8gb-235682/` and
`krasis-internal/qwen27b-repair-235737/`. Separate performance experiments remain
on the `performance` branch.

Integrated contraction follow-up:
`krasis-internal/qwen27b-integrated-235797/RESULTS.md`. This includes the actual
Rust attention replay, complete block and encoder checks, and serving evidence.
Qwen-specific sigmoid/SiLU materialization and BF16 GDN norm-parameter boundaries
apply in prefill and decode. Numerical admission remains governed by the complete
arrays in that report, separately from memory fit and functional requests.


Numerical isolation follow-up (`krasis-internal/qwen27b-numerical-235763/`):
actual layer0 normalized projection inputs match the independent implementation
bitwise. Combining QKV with Z, or B with A, changes BF16 contraction results.
The native Qwen BF16 path now restores the separate contractions using temporary
canonical-layout storage; it checks those physical bytes against the 600 MiB
reserve and waits for readers before release. The largest tested temporary
weight allocation is 160 MiB, plus the bounded projection output. BF16 L2
preparation and an FP32 token recurrence avoid the captured FLA final-state
rounding failure. Other model families retain their existing dispatch. Qwen
rotary uses pair ownership and the reference BF16 table/product boundaries.
These are accuracy repairs, not a speed claim.

The actual independent text backend is mixed: installed fused `causal-conv1d`
with SiLU, plus the Torch chunked delta-rule fallback. Its global fast-path
warning does not mean that convolution is unfused. The rejected unfused-BF16
convolution counterfactual is retained only in the investigation.

Vision claims distinguish individual operators, complete residual blocks and
complete encoders. The earlier numerical-isolation baseline failed same-input
block26. The integrated-235797 baseline passed all 81 complete same-input blocks and
three mergers; its three accumulated complete-feature comparisons failed. Independent FP64 projection/attention equations
and reduced-BF16-reduction controls demonstrate backend rounding sensitivity;
they do not replace the original reference or waive its unchanged acceptance
checks. Functional image answers and constrained-memory fit remain separate
from numerical admission. Arête execution and actual small-card qualification
are still not claimed.


Causal repair follow-up: `krasis-internal/qwen27b-causal-235834/RESULTS.md`.
The bounded native GDN path now uses 64-token FP32 chunk contractions, matching
reference decay scans and reduction order without changing chronological state
semantics. Its largest live workspace is covered by a separate 32 MiB admission
allowance and a physical 600 MiB reserve check. Gated, outer and attention-head RMSNorm use the
reference reduction/materialization contract. Diagnostic rollback controls are
`KRASIS_QWEN_GDN_RECURRENCE=serial`, `KRASIS_QWEN_GDN_NORM=legacy`, and
`KRASIS_QWEN_OUTER_NORM=legacy`, `KRASIS_QWEN_HEAD_NORM=legacy`, and
`KRASIS_QWEN_SOFTMAX=legacy`; these isolate arithmetic changes independently.
Vision SDPA uses the frozen PyTorch `UNFUSE_FMA` compiler contract with precise
math. Complete original, square and rectangular encoders now match the frozen
reference bitwise. The report records full-text numerical, functional and memory
verdicts separately; historical failed identities and independent FP64 evidence
remain preserved. No Arête execution or actual small-card qualification is implied.
