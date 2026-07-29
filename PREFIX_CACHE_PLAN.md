# Hybrid Prefix-State Cache Plan

## Goal

Avoid re-prefilling an unchanged conversation prefix on every compatible chat
request while preserving the model's exact existing results. Hybrid models must
reuse both kinds of sequence state:

- paged KV state for full/GQA attention layers;
- recurrent and convolution state for linear-attention layers (and the
  equivalent recurrent state for other supported hybrid layer types).

This is an experimental, opt-in feature. It is **disabled by default**.
With it disabled, requests must follow the current full-prefill path without
additional state mutation, cache lookup, or changed sampling behavior.

## User-facing switch

Add one persistent launcher setting and one CLI override:

- `CFG_PREFIX_CACHE="0"` by default;
- `--prefix-cache` enables it;
- `--no-prefix-cache` explicitly disables it.

Pass the resolved boolean explicitly from the Python server to `RustServer`;
do not use a hidden always-on environment variable. Log the resolved state once
at startup. A later PR may promote the default only after correctness and
stability data justify it.

## Independent SSE compatibility switch

Keep the existing trailing-timing-chunk compatibility fix separate from prefix
caching and opt-in:

- `CFG_SSE_TIMING_COMPAT="0"` by default;
- `--sse-timing-compat` enables it;
- `--no-sse-timing-compat` explicitly disables it.

With the switch off, retain upstream's empty `choices` array in the custom
timing SSE chunk. With it on, emit one neutral OpenAI-compatible choice with an
empty delta. No normal content, tool-call, finish, or `[DONE]` chunk may change.
Unit tests must cover both exact JSON shapes.

## First implementation scope

Implement a single-lineage, exact-token continuation cache. Krasis currently
serializes model requests, so one retained live state is enough to accelerate
the common agent workflow without introducing a radix tree or a VRAM-resident
multi-session cache.

The cache records:

- the exact token IDs represented by the retained device state;
- the represented sequence position;
- model/runtime compatibility metadata needed for safe reuse;
- hit, miss, invalidation, reused-token, and suffix-token counters.

The retained device state consists of the already-live state on every pipeline
GPU. No second full KV allocation is introduced in this phase.

## Eligibility and lookup

A request is a cache hit only when all of the following are true:

1. the feature is enabled;
2. the request has no image or other external prefill embeddings;
3. a valid retained state exists from a completed request;
4. the new rendered prompt's token IDs start with the complete retained token
   sequence, compared token-for-token;
5. the retained position and every device store agree;
6. the suffix is non-empty and the resulting context fits the configured KV
   capacity;
7. no intervening endpoint or failure invalidated sequence state.

Sampling parameters do not invalidate a deterministic prefix state. Changes to
messages, tools, chat template output, tokenizer output, or model identity
naturally miss because the token sequence differs.

Two additional eligibility rules hold in the implementation:

- prompts that overflow the KV capacity stay on the fully validated
  fresh-prefill path so their existing error contract is unchanged;
- requests that can use a speculative draft model never promote an entry,
  because batched draft verification and its recurrent-state rollback break
  per-token consumed accounting.

On any uncertainty, log a miss reason, invalidate the entry, and use the
existing full-prefill path.

## State-position accounting

Correct token accounting is the central invariant.

Prefill produces the first sampled token while device state initially
represents the prompt. Each decode step consumes one token and advances device
state before selecting the next token. The implementation must record only
tokens that have actually been consumed by the model. An emitted token that has
not yet been consumed must not be included in the cached prefix.

Before enabling reuse, add tests around a small pure-Rust state tracker that
models:

- normal EOS completion;
- max-token completion;
- stop-string completion;
- client disconnect/write failure;
- tool-call output;
- zero/one-token completion.

Promotion of a new cache entry happens only after all GPU pipeline segments
have completed the same decode position and the exact consumed-token list is
known.

## Continuation prefill

Add a separate continuation-prefill entry point rather than weakening the
fresh-prefill contract:

- fresh prefill keeps its current state resets and position zero;
- continuation prefill receives `prefix_len` and only the suffix token IDs;
- it does not zero KV, recurrent, convolution, or Mamba state;
- RoPE/GQA positions and ring/cache indices start at `prefix_len`;
- chunk positions are absolute even though scratch buffers contain only the
  suffix;
- continuation bypasses the stage-exact temporary KV cache and appends suffix
  positions directly to the registered decode cache, preserving the retained
  prefix;
- updated recurrent state is left in the same decode-store buffers;
- auxiliary pipeline stores are advanced consistently, without overwriting a
  valid retained prefix with reset state.

If continuation setup or execution fails before any state mutation, fall back
to full prefill. If it fails after mutation may have begun, invalidate and run
the existing full reset/prefill path. Never continue decoding from partially
advanced state.

## Invalidation

Invalidate the live entry before or after these operations as appropriate:

- any full prefill;
- `/v1/internal/prefill_logits` or reference/benchmark endpoints that reuse the
  same device state;
- multimodal prefill;
- model/runtime reconfiguration;
- sequence overflow or ring-window wrap not explicitly validated;
- CUDA, prefill, decode, aux-copy, or cleanup error;
- cancellation where consumed-token accounting is uncertain;
- exact-prefix mismatch.

The first version deliberately supports one conversation lineage. An unrelated
request replaces it via full prefill; returning to an older branch is a miss.

`RustServer::benchmark_request` mutates device sequence state from a Python
thread without access to the server request loop. It bumps a global
sequence-state epoch on entry and exit; cache entries record the epoch captured
before their request's prefill and any epoch change forces a miss and
invalidation.

## Observability

Emit one concise per-request cache line:

`prefix_cache=hit|miss|disabled reason=... cached=N reused=N suffix=N`

Expose aggregate counters in existing timing/benchmark JSON where practical:

- lookups, hits, misses, invalidations;
- reused prompt tokens and suffix-prefilled tokens;
- continuation-prefill milliseconds;
- full-prefill milliseconds.

Do not call HCS expert residency a prefix-cache hit rate; it is a separate
weight-residency mechanism.

## Test and validation sequence

1. Pure Rust unit tests for exact-prefix matching, state tracking, invalidation,
   and disabled-mode behavior.
2. Python config/CLI tests proving the default is off and both CLI overrides
   work.
3. Existing non-GPU test suite with the feature disabled.
4. Small-model equivalence test:
   - run the same multi-turn prompts with cache off and on;
   - greedy decode must produce identical token IDs;
   - compare selected logits at the continuation boundary within the existing
     backend's established tolerance.
5. Hybrid-model equivalence test covering both KV and recurrent layers.
6. Multi-GPU test proving every pipeline segment reaches the same position.
7. Fault tests for mismatch, disconnect, forced prefill error, and aux-copy
   error, followed by a clean request that must match cache-off output.
8. Agent benchmark comparison:
   - same patch quality/resolution result;
   - materially lower repeated-prefix prefill time;
   - report hit rate and reused tokens;
   - no new tool-protocol or streaming failures.

## Acceptance criteria

- Feature remains disabled unless explicitly enabled.
- Cache-off path has no behavioral or performance regression beyond trivial
  startup/config plumbing.
- Cache-on greedy output is token-identical to cache-off for validated
  single- and multi-GPU hybrid tests.
- Any mismatch or uncertain state safely falls back to full prefill.
- Long agent conversations show prefix hits and reduced prefill work.
- SSE timing compatibility is independently opt-in and its disabled path
  preserves the upstream timing-chunk shape.

## Deferred work

- Multi-entry radix/paged prefix cache.
- Host snapshots or device snapshots for conversation branching.
- Cross-process persistence.
- Multimodal prefix reuse.
- Automatic default enablement.
