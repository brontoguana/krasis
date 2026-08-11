# Krasis Quality Stats

Quality comparisons use BF16 llama-witness references where available, and
Krasis BF16 reference runs for models that do not yet have llama-witness
artifacts. Perplexity deltas use a Krasis BF16 runtime baseline on WikiText-2
where that metric is valid for the model, and a chat-continuation BF16
baseline where raw-token corpus PPL is not valid for the model.

Detailed quality logs and artifacts are indexed in [benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md).

| Model | Params | Active params | Quant | BF16 reference | Prompts | PPL | PPL delta vs BF16 | BF16 top-k drift | Prefill argmax | Prefill top-10 | First token | Result |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [DeepSeek-V4-Flash-0731](benchmarks/BENCHMARKS.md#deepseek-v4-flash-0731-learned-index-gemm-attempt-1-full-ppl--2026-08-03) | 304.2B checkpoint / 284B main | 13B main | INT4/BF16/BF16 KV | llama-witness native source + frozen Krasis scalar anchor | 4 | 4.8214 | +0.114% vs 4.8159 scalar anchor | diagnostic deltas in accepted range | 4/4 | 4/4 | 4/4 | PASS |
| [DeepSeek-V4-Flash-0731](benchmarks/BENCHMARKS.md#deepseek-v4-hqq8-attention-and-native-cache-acceptance--2026-08-10) | 304.2B checkpoint / 284B main | 13B main | INT4/HQQ8/BF16 cache | accepted BF16/BF16 runtime + llama-witness native source | 4 | 4.8160 | -0.111% vs accepted BF16/BF16 | accepted witness range | 4/4 | 4/4 | 4/4 | PASS |
| [DeepSeek-V4-Flash-0731 Native](benchmarks/BENCHMARKS.md#deepseek-v4-hqq8-attention-and-native-cache-acceptance--2026-08-10) | 304.2B checkpoint / 284B main | 13B main | INT4/HQQ8/Native cache | HQQ8/BF16 cache + llama-witness native source | 4 | 4.8161 | +0.000775% vs HQQ8/BF16 cache | exact cache reconstruction | 4/4 | 4/4 | 4/4 | PASS |
| DeepSeek-V4-Flash-0731 launcher default | 304.2B checkpoint / 284B main | 13B main | INT4/HQQ6/Native cache | HQQ8/Native + llama-witness native source | 4 | 4.8385 | +0.466% vs HQQ8/Native | accepted optional lower-bit mode | 4/4 | 4/4 | 4/4 | PASS |
| [DeepSeek-V4-Flash-0731 mixed](benchmarks/BENCHMARKS.md#deepseek-v4-and-gemma4-mixed-hqq-acceptance--2026-08-11) | 304.2B checkpoint / 284B main | 13B main | INT4/HQQ4+10%/Native cache | HQQ8/Native + llama-witness native source | 4 | 4.8573 | +0.856% vs HQQ8/Native | accepted witness range | 4/4 | 4/4 | 4/4 | PASS |
| [DeepSeek-V4-Flash-0731 mixed](benchmarks/BENCHMARKS.md#deepseek-v4-and-gemma4-mixed-hqq-acceptance--2026-08-11) | 304.2B checkpoint / 284B main | 13B main | INT4/HQQ6+10%/Native cache | HQQ8/Native + llama-witness native source | 4 | 4.8259 | +0.203% vs HQQ8/Native | accepted witness range | 4/4 | 4/4 | 4/4 | PASS |
| Qwen3.6-35B-A3B | 35.5B text | 3.0B | INT4/HQQ4/k4v4 | llama-witness BF16 | 8 | 5.8175 | +3.28% | avg 0.254%, max 0.783% | 8/8 | 8/8 | 8/8 | PASS |
| Qwen3.6-35B-A3B | 35.5B text | 3.0B | INT4/HQQ6/k6v6 | llama-witness BF16 | 8 | 5.6895 | +1.00% | avg 0.200%, max 0.695% | 8/8 | 8/8 | 8/8 | PASS |
| Ornith-1.0-35B | 35B class | 3B class | INT4/HQQ4/k4v4 | Krasis BF16 | n/a | 5.9170 | +3.67% | n/a | n/a | n/a | n/a | PASS |
| Ornith-1.0-35B | 35B class | 3B class | INT4/HQQ6/k6v6 | Krasis BF16 | n/a | 5.7069 | -0.01% | n/a | n/a | n/a | n/a | PASS |
| Qwen3-Coder-Next | 80B class | n/a | INT4/HQQ4/k4v4 | llama-witness BF16 | 8 | 5.6149 | +5.33% | avg 0.382%, max 1.564% | 8/8 | 8/8 | 8/8 | PASS |
| Qwen3-Coder-Next | 80B class | n/a | INT4/HQQ6/k6v6 | llama-witness BF16 | 8 | 5.3326 | +0.04% | avg 0.297%, max 1.260% | 8/8 | 8/8 | 8/8 | PASS |
| Qwen3.5-35B-A3B | 35B class | 3B class | INT4/HQQ4/k4v4 | llama-witness BF16 | 10 | 6.0853 | +4.85% | avg 0.545%, max 3.579% | 10/10 | 10/10 | 10/10 | PASS |
| Qwen3.5-35B-A3B | 35B class | 3B class | INT4/HQQ6/k6v6 | llama-witness BF16 | 10 | 5.8505 | +0.80% | avg 0.314%, max 2.096% | 10/10 | 10/10 | 10/10 | PASS |
| Qwen3.5-122B-A10B | 122B class | 10B class | INT4/HQQ4/k4v4 | llama-witness BF16 | 14 | 4.5805 | +3.25% | avg 1.195%, max 4.089% | 14/14 | 14/14 | 14/14 | PASS |
| Qwen3.5-122B-A10B | 122B class | 10B class | INT4/HQQ6/k6v6 | llama-witness BF16 | 14 | 4.4037 | -0.73% | avg 0.617%, max 3.057% | 14/14 | 14/14 | 14/14 | PASS |
| Qwen3-235B-A22B | 235B class | 22B class | INT4/HQQ4/k4v4 | llama-witness BF16 | 14 | 4.2236 | +4.74% | avg 2.746%, max 15.866% | 14/14 | 14/14 | 14/14 | PASS |
| Qwen3-235B-A22B | 235B class | 22B class | INT4/HQQ6/k6v6 | llama-witness BF16 | 14 | 4.0252 | -0.18% | avg 2.225%, max 18.889% | 13/14 | 14/14 | 13/14 | PASS |
| Qwen3.5-397B-A17B | 397B class | 17B class | INT4/HQQ4/k4v4 | Krasis BF16 | n/a | 3.0189 | +4.66% | n/a | n/a | n/a | n/a | PASS |
| Qwen3.5-397B-A17B | 397B class | 17B class | INT4/HQQ6/k6v6 | Krasis BF16 | n/a | 2.8806 | -0.14% | n/a | n/a | n/a | n/a | PASS |
| Ornith-1.0-397B | 397B class | 17B class | INT4/HQQ4/k4v4 | Krasis BF16 | n/a | 3.2146 | +3.58% | n/a | n/a | n/a | n/a | PASS |
| Ornith-1.0-397B | 397B class | 17B class | INT4/HQQ6/k6v6 | Krasis BF16 | n/a | 3.1049 | +0.05% | n/a | n/a | n/a | n/a | PASS |
| Step-3.7-Flash | 201.4B total / 199.4B text | 13.9B | INT4/HQQ4/k4v4 | llama-witness BF16 | 8 | 1.7671 | +3.80% | avg 3.655%, max 7.582% | 7/8 | 8/8 | 8/8 | PASS |
| Step-3.7-Flash | 201.4B total / 199.4B text | 13.9B | INT4/HQQ6/k6v6 | llama-witness BF16 | 8 | 1.7016 | -0.04% | avg 2.206%, max 3.997% | 8/8 | 8/8 | 8/8 | PASS |
| Gemma-4-26B-A4B-it | 26B class | 4B class | INT4/HQQ4/k4v4 | Krasis BF16 | 14 | 1.1736 | +9.14% chat | avg 1.969%, max 39.239% | 187/197 | 197/197 | 14/14 | PASS |
| Gemma-4-26B-A4B-it | 26B class | 4B class | INT4/HQQ4+10%/k6v6 | Krasis BF16 | 14 | 1.1234 | +4.47% chat | avg 1.205%, max 53.929% | 191/197 | 197/197 | 14/14 | PASS |
| Gemma-4-26B-A4B-it | 26B class | 4B class | INT4/HQQ6/k6v6 | Krasis BF16 | 14 | 1.0792 | +0.36% chat | avg 0.279%, max 10.515% | 193/197 | 197/197 | 14/14 | PASS |
| Gemma-4-26B-A4B-it | 26B class | 4B class | INT4/HQQ6+10%/k6v6 | Krasis BF16 | 14 | 1.0784 | +0.29% chat | avg 0.165%, max 4.657% | 195/197 | 197/197 | 14/14 | PASS |
| Nemotron-3-Super-120B-A12B | 123.6B total | 12.4B | INT4/HQQ4/k4v4 | llama-witness BF16 | 6 | n/a | n/a | avg 18.267%, max 54.409% | 4/6 | 6/6 | 4/6 | PASS |
| Nemotron-3-Super-120B-A12B | 123.6B total | 12.4B | INT4/HQQ6/k6v6 | llama-witness BF16 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | BLOCKED |
| Nemotron-3-Nano-30B-A3B | 31.6B total | 3.2B | INT4/HQQ4/k4v4 | llama-witness BF16 | 6 | n/a | n/a | avg 14.185%, max 44.584% | 4/6 | 6/6 | 4/6 | PASS |
| Nemotron-3-Nano-30B-A3B | 31.6B total | 3.2B | INT4/HQQ6/k6v6 | llama-witness BF16 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | BLOCKED |

Column notes:

- DeepSeek-V4-Flash-0731's PPL delta is intentionally reported against the
  frozen accepted scalar INT4 anchor (`4.8159`), not against the native-source
  witness quantization tier. The final 281-window result is
  `4.82137538882331` (`+0.11369%`), while the independent llama-witness gate
  passed all four prompts for prefill argmax, top-10 containment, and first
  token. The full prefill campaign stayed inside its permanent quality ledger.
  Phase-one HQQ8/BF16 cache measured `4.816034369291275`, or `-0.1108%`
  versus that accepted BF16/BF16 run. Native measured `4.816071679021491`,
  only `+0.000775%` versus HQQ8/BF16 cache, and reconstructs the prior
  post-QAT BF16 cache values exactly. Both HQQ8 modes passed the four-prompt
  witness; Native is kept explicit for performance, not quality, reasons.
  The mixed 10% profiles each promoted 87 of 172 eligible projection tensors
  within a measured 72.5/75.2 MiB budget. HQQ4+10%/Native measured
  `4.857317863`, between fixed HQQ6 and fixed HQQ4 and 0.731% better than the
  fixed-HQQ4 endpoint. HQQ6+10%/Native measured `4.825864933`, between HQQ8
  and fixed HQQ6 and 0.262% better than fixed HQQ6. Both retained 4/4 witness
  agreement and the unchanged 600 MiB runtime safety contract.

- `PPL delta vs BF16` uses the per-model Krasis BF16 runtime baseline, measured
  through the Rust prefill `/v1/internal/prefill_logits` test endpoint with
  target-token logprobs.
- Step-3.7's Krasis BF16 PPL baseline is diagnostic only because that runtime
  path is marked unvalidated for correctness. The pass/fail reference remains
  BF16 llama-witness.
- Qwen3-235B BF16 startup builds a quick route heatmap because no BF16-approved
  heatmap is published. The completed rerun reached server ready after that
  startup path and produced the BF16 PPL baseline used for the Q235 deltas.
- Qwen3.5-397B rows use Krasis BF16 as the current accepted reference because
  no BF16 llama-witness artifact exists yet. Krasis BF16 PPL was sane on
  WikiText-2 (`2.8845`), measured HQQ deltas were acceptable, and sampled
  output was sensible, so these rows are marked `PASS` for now.
- Ornith-1.0-35B rows use Krasis BF16 as the current accepted reference because
  no BF16 llama-witness artifact exists yet. Krasis BF16 WikiText-2 PPL was
  sane (`5.7077`), HQQ4 measured `+3.67%`, and HQQ6 measured `-0.01%` against
  that baseline.
- Ornith-1.0-397B rows use Krasis BF16 as the current accepted reference
  because no BF16 llama-witness artifact exists yet. Krasis BF16 WikiText-2 PPL
  was sane (`3.1034`), HQQ4 measured `+3.58%`, and HQQ6 measured `+0.05%`
  against that baseline.
- Gemma rows use chat-continuation PPL because raw WikiText-2 PPL is
  pathological for this instruction/vision model. Krasis BF16 generated 14 chat
  continuations, and BF16/HQQ runs scored the same 197 continuation tokens
  through `/v1/internal/prefill_logits`. The BF16 baseline was PPL `1.0753`.
  HQQ4, HQQ4+10%, HQQ6, and HQQ6+10% kept the BF16 continuation token in the
  top 10 for `197/197` scored tokens. The mixed rows use the same accepted BF16
  continuations and 10% planner budget as their launcher presets.
- Nemotron Nano and Super now have BF16 llama-witness first-token references.
  Their HQQ4/k4v4 rows passed all six prompt verdicts with full prefill top-10
  containment. The first-token drift percentages are reported even though they
  are materially larger than the mature Qwen rows, because the witness verdict
  is based on containment and the values should remain visible. HQQ6/k6v6
  remains `BLOCKED` until those exact configurations are compared against the
  new witnesses.
- Nemotron PPL is `n/a`: the prior Krasis BF16 raw-corpus diagnostic produced
  pathological absolute values and is not a valid quality metric for these
  rows. Those raw results remain archived as diagnostics but are not used to
  justify the witness `PASS`.
- `BF16 top-k drift` is normalized Jensen-Shannon divergence over the BF16/HQQ
  top-10 union. For `llama-witness BF16` rows it is measured on first-token
  witness diagnostics; for `Krasis BF16` chat-continuation rows it is measured
  over the scored continuation-token positions from `/v1/internal/prefill_logits`.
  Lower is closer to BF16. This is a top-k diagnostic, not full-vocabulary
  forced-token drift.
- `Prefill argmax` and `Prefill top-10` are BF16 top-1/top-10 containment counts
  over the measured comparison positions. For llama-witness rows those positions
  are one first-token diagnostic per prompt; for chat-continuation rows they are
  scored continuation tokens.
