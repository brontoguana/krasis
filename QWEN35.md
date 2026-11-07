# Bounded Qwen3.5-27B text and images

This development path streams the original BF16 language weights from host RAM
through one reusable GPU slab. Prefill, recurrent state updates, decode scheduling
and the Qwen3.5 vision network execute in Rust/CUDA. Python handles startup,
image processing and request bookkeeping.

## Demonstrated profile

- One RTX A6000, physically constrained to a **6,144 MiB device-wide ceiling**
  using an independent allocation. This is constrained-A6000 evidence, **not an
  actual 6GB or 8GB card test**. An 8,192 MiB control also passed. Normal-server peak effective device use
  was 5,210 MiB, leaving 934 MiB free in the integrated follow-up, sampled every 25 ms.
- Original local Qwen/Qwen3.5-27B BF16 safetensors; no model weights downloaded.
- 4,096-token capacity, including image tokens and requested output; 256 MiB BF16
  KV, complete recurrent state, 256-token language prefill chunks, 600 MiB reserve.
- One image per request, at most 1,024 encoder patches / 256 image tokens.
  A 512×512 image was tested, alongside 256×256 red/blue images and text follow-up.
- Text, image-dependent answers, a 4,049-token prompt, eager/graph state identity,
  reset, cancellation and active-GPU/pageable-RAM cache restoration were exercised.
  Over-capacity requests return HTTP 413.

Allow **128 GiB system RAM**. Source tensors and immutable pinned native copies
coexist during preparation; measured peak RSS was approximately 113 GiB. Startup
checks available host RAM, including Linux cgroup v2 limits. The launcher budget
includes setup headroom. Native pinned text weights occupy 53,786,705,920 bytes;
the reusable GPU weight slab is 2,542,796,800 bytes. Vision streams one block at a
time through an additional 89,677,312-byte slab, never a complete GPU tower.

This is a memory-fit implementation. Sustained BF16 decode measured about
**0.44 tokens/sec** on the A6000 because nearly 50 GiB of weights cross PCIe per
forward. A smaller card's throughput has not been measured. No quantisation is
applied by this profile, so packed-weight quantisation metadata is zero. INT8/HQQ
settings are rejected before loading rather than silently creating full GPU copies.

## Run from the development checkout

Use complete local safetensors, tokenizer and processor assets. This command
keeps all Hugging Face access offline and performs normal server warmup and
calibration:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 KRASIS_QWEN_GPU_LIMIT_MIB=8192 \
./dev python -m krasis.server \
  --model-path /path/to/local/Qwen3.5-27B --num-gpus 1 \
  --attention-quant bf16 --dense-mlp-quant bf16 --lm-head-quant bf16 \
  --shared-expert-quant bf16 --kv-dtype bf16 --kv-cache-mb 256 \
  --max-context-tokens 4096 --no-enable-thinking
```

Streaming defaults on for the 64-layer, hidden-5120, intermediate-17408 dense
Qwen3.5-27B geometry. Other model families and dense sizes retain their existing
policy. `KRASIS_QWEN_DENSE_STREAM=0` restores the old resident path, which requires
substantially more VRAM. `KRASIS_QWEN_GPU_LIMIT_MIB` controls joint admission
planning; it is not a replacement for physical GPU limits. Qualification used
an independent physical allocation plus device-wide NVML sampling.

Graphs capture ordered H2D copies with stable pointers into the same slab used
by eager execution. Group ownership remains immutable for the loaded model.
Multi-GPU, speculation/batched verification and non-BF16 bounded profiles are
rejected. CPU Torch forward/stress modes are not bounded execution paths.

Vision uses the Qwen3.5 Conv3d patch contract, learned position interpolation,
axial RoPE, 27 blocks and pre-shuffle merger. Head-width-72 attention uses a
native split-KV SDPA contraction, with bounded FP32 partial-output storage and
an occupancy-based split count. BF16 Linear uses a bias epilogue and a 1 MiB
cuBLASLt workspace. Inverse rotary frequencies are computed once on the CPU in
FP32, matching the reference's construction order. Checkpoints without local
vision assets remain text-only.

`KRASIS_QWEN_VISION_ATTENTION=fast` retains the padded FlashAttention path;
`portable` retains the CUDA equation fallback. `KRASIS_QWEN_VISION_LINEAR=fp32`
retains the FP32-output GEMM and separate bias path. These alternatives retain
their independent numerical verdicts.

Within the 4K BF16 profile, text attention defaults to batched QK/PV contractions
with BF16 scores, scaling and probabilities, and FP32 accumulation. Query bands
bound the score buffer to 16 MiB without splitting the key reduction. Grouped
heads and cached-prefix causal masks use the same contract. Physical allocations
are charged in joint admission and checked against the 600 MiB reserve.
`KRASIS_QWEN_TEXT_ATTENTION=fast` retains the previous attention implementation;
larger contexts and unsupported KV formats retain their existing dispatch.

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
The causal repair follow-up clears the fixed 28-token, complete 64-layer text
comparison: zero failing logits and zero failing arrays among all 128 sequence
state arrays at unchanged `atol=.03, rtol=.01`. Original, square and rectangular
complete encoders match their frozen reference features bitwise. The original
failed identities remain preserved; see the causal report for substitutions,
reference repeatability, operator checks and the exact measured scope.
These fixed-fixture passes do not establish general model quality, canonical
ADQ/release admission, Arête execution or other-GPU qualification.

Evidence and immutable controllers:
`krasis-internal/qwen27b-8gb-235682/` and
`krasis-internal/qwen27b-repair-235737/`. The Maya performance work is independently
preserved on `perf/maya-parity-20261107`; it was not merged into this implementation.

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
