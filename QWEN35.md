# Bounded Qwen3.5-27B text and images

This development path streams the original BF16 language weights from host RAM
through one reusable GPU slab. Prefill, recurrent state updates, decode scheduling
and the Qwen3.5 vision network execute in Rust/CUDA. Python handles startup,
image processing and request bookkeeping.

## Demonstrated profile

- One RTX A6000, physically constrained to a **6,144 MiB device-wide ceiling**
  using an independent allocation. This is constrained-A6000 evidence, **not an
  actual 6GB or 8GB card test**. An 8,192 MiB control also passed. Normal-server peak effective device use
  was 5,149 MiB, leaving 995 MiB free, sampled every 25 ms.
- Original local Qwen/Qwen3.5-27B BF16 safetensors; no model weights downloaded.
- 4,096-token capacity, including image tokens and requested output; 256 MiB BF16
  KV, complete recurrent state, 256-token language prefill chunks, 600 MiB reserve.
- One image per request, at most 1,024 encoder patches / 256 image tokens.
  A 512×512 image was tested, alongside 256×256 red/blue images and text follow-up.
- Text, image-dependent answers, a 4,049-token prompt, eager/graph state identity,
  reset, cancellation and active-GPU/pageable-RAM cache restoration were exercised.
  Over-capacity requests return HTTP 413.

Allow **128 GiB system RAM**. Source tensors and immutable pinned native copies
coexist during preparation; measured peak RSS was approximately 110 GiB. Startup
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
axial RoPE, 27 blocks and pre-shuffle merger. The existing FlashAttention
sidecar uses zero padding from head width 72 to 96 with the original scale;
`KRASIS_QWEN_VISION_ATTENTION=portable` selects the native CUDA fallback. Both
were compared against an independent Transformers encoder. Checkpoints without
local vision assets remain text-only.

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
Complete 27B native/reference logits and all 128 sequence-state arrays were
compared. Follow-up repairs retain FP32 delta-norm scales and the GDN BF16 norm/scale
boundaries, and round decode residuals before normalization. Real four-layer
resident and streamed paths agree exactly, including all state arrays. The
remaining independent-reference discrepancy begins in the first GDN recurrent
state, before streaming can distinguish the paths. Numerical qualification
remains incomplete: the strict full-array
comparison does not pass, despite agreeing first tokens and successful requests.
The complete vision encoder also retains failures at unchanged `rtol=.01,
atol=.03`; operator-level agreement is not whole-model admission. No absolute
quality, ADQ, release or other-GPU qualification is claimed.

Evidence and immutable controllers:
`krasis-internal/qwen27b-8gb-235682/` and
`krasis-internal/qwen27b-repair-235737/`. The Maya performance work is independently
preserved on `perf/maya-parity-20261107`; it was not merged into this implementation.
