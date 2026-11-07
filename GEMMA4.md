# Gemma 4 INT4 on an 8 GiB GPU

The bounded profile supports the Gemma-4-26B-A4B-it architecture with native Rust/CUDA language and vision execution. Python loads and converts weights at startup and preprocesses images. Measurements use an RTX A6000 externally constrained to an 8 GiB device-wide allowance; physical 8GB-card performance is untested. This feature branch does not establish general release or independent whole-model numerical qualification.

Select the checkpoint in the launcher on a single GPU with at most 8 GiB, or explicitly select the profile:

```sh
./dev python -m krasis.server \
  --model-path /path/to/gemma-4-26b-a4b-it \
  --gemma-8gb --num-gpus 1 \
  --max-context-tokens 32768 --kv-cache-mb 512 \
  --no-enable-thinking
```

Authentication, binding and model acquisition follow the normal server workflow. The profile does not download model weights. On a larger GPU `--gemma-8gb` forces the profile; it does not impose a physical VRAM allowance. `KRASIS_GEMMA_8GB=0` explicitly selects the existing larger-memory path. The 32K default includes text, image tokens and generated output. Requests must also fit measured live memory while leaving at least 600 MiB free. The bounded server plans with at least 664 MiB free to cover transient driver/library allocations. Larger contexts require a sufficient explicit KV budget and remain subject to admission.

The profile uses INT4 routed experts (Marlin, group 128), HQQ4 attention (default group 128), INT4 dense MLPs and output head, and native streamed INT4 vision projections. Dense down projections use group 64 because their input width is 2,112. BF16 activations, normalization, small routing matrices, learned image position embeddings and the host token embedding retain their original semantics. The tied output projection has a separate INT4 representation; there is no retained full BF16 language-model source copy.

The authoritative BF16 token embedding is host-mapped. Packed dense/head matrices occupy about 629 MiB resident GPU memory. Routed experts stream from their existing source-backed packed cache. Vision has one pinned packed backing (about 316 MiB) and one device staging slab (about 45.4 MiB), rather than keeping its complete encoder on the GPU. The two-axis learned vision position table accounts for 45 MiB of the higher-precision host backing. Conversion is incremental, not a simultaneous full BF16 and packed model load.

INT4 weights are independent of KV precision. This profile uses k6v6 KV. The existing k4v4 ring guard remains in place. Its 25 sliding-attention layers retain a 1,024-token mathematical window and bounded physical storage. The exact prefill stage keeps 32 additional rows for short chat-template rewinds; deeper branches require a suitable saved boundary or full prefill. The five full-attention layers retain full history. Export reconstructs complete canonical sliding windows before decode, with physical wrap and stage-specific conversion argument order handled explicitly.

One image per request is supported, resized by the checkpoint image processor to at most 280 soft tokens / 2,520 valid patches. Image token blocks remain intact across prefill chunks, and sliding image attention preserves its bidirectional mask. Square and rectangular images are tested. Image requests remain explicitly uncacheable, including follow-ups that resend the image. Text session restoration supports active GPU and pageable RAM boundaries. CUDA graph decode remains guarded for this ring profile; execution uses the native eager path.

Numerical checks distinguish packed-operator correctness, backend compatibility and quantisation loss. Native vision's complete unconditioned feature trajectory does not meet the independent backend tolerance, despite passing local operator replays. A final-encoder substitution clears the independent pool/projector comparison. INT4 versus BF16 vision features have cosine similarity around 0.94 on the retained images. Simple image answers work, but this is not a broad visual-quality qualification. The coding check for removing all duplicates also fails in the preserved BF16 dense/head control.

Detailed memory, timing, context and remaining quality verdicts are recorded in the companion internal evidence directory `gemma4-int4-8gb-236082/`. Its final results and immutable controllers are the authority for measured limits; allocation alone is not a context-quality pass.

Measured final default-profile evidence uses 32K context and a 512 MiB KV budget (a ceiling, not a fixed allocation). Three short 128-token text completions sustain 21.78 tokens/sec and finish in 6.92–6.93 seconds. Image requests near 4K finish in 2.819–2.829 seconds; near 32K in 17.796–17.900 seconds, with 16.85–16.95 seconds spent in prefill. These are warmed, uncached-prompt requests after byte callbacks were disabled. Minimum free memory across final default checks was 702.1 MiB; host RSS peaked at 18.55 GiB including text-session snapshots. Fresh calibration and heatmap startup takes about three minutes. These are constrained-A6000 measurements, not physical 8GB-card results.

A no-HCS run completes three 64K image/retrieval repeats and a changed-code/image variant correctly in 52.1–52.8 seconds. The 96K variant runs but fails changed-code retrieval; 128K image input is rejected before scratch allocation. A final 32K changed-code repeat also returns a wrong digit despite earlier passes. The 64K result is a bounded demonstration, not a generally reliable retrieval guarantee. Overall output stability and quality remain unqualified; this profile is experimental.

Higher-precision weight exceptions measured from the selected checkpoint are 1,476,395,008 bytes of host embedding, 21,803,520 bytes of routers/scales, 1,224,252 bytes of language norms/scalars and 47,447,136 bytes of vision positions/norms/std data. Runtime BF16 activations, required per-layer RoPE tables, k6v6 KV and transient scratch are additional. INT4 vision stages one projection at a time into BF16 contraction scratch; it does not keep a complete BF16 encoder resident. New dense/head conversion reached 16.67 GiB RSS while reusing existing expert/HQQ artifacts; a fully cold rebuild of all prior artifacts was not remeasured.

The image limit is 280 processed soft tokens; public requests up to 270 were exercised. The 64K tests use 256/260 image tokens. Runtime code identities cover the new Gemma execution files; older heatmaps must be rebuilt. The companion evidence directory retains failed quality/reserve/profiling receipts and the final audit's failed quality verdict.
