# Krasis Quality Stats

Quality comparisons use BF16 llama-witness references and `./dev
witness-compare` for pass/fail checks. Perplexity deltas use a Krasis BF16
runtime baseline on WikiText-2 so HQQ profiles are compared against the same
runtime path.

Qwen3.6 witness reference: profile `llama_witness_qwen36_expanded`, `8`
prompts, `16` generated tokens per prompt. Krasis BF16 WikiText-2 baseline:
PPL `5.6329`, BPC `2.4939`, `19,999` scored tokens, window `2048`, stride
`1024`
([PPL log](benchmarks/20260714_qwen36_quality_rust_ppl_bf16.log)).

Step-3.7 witness reference: profile `greedy_chat_thinking_off`, `8` prompts,
`16` generated tokens per prompt. Krasis BF16 WikiText-2 diagnostic baseline:
PPL `1.7023`, BPC `0.7675`, `19,999` scored tokens, window `2048`, stride
`1024`
([PPL log](benchmarks/20260714_step37_quality_rust_ppl_bf16_bf16kv.log)).

Qwen3-Coder-Next witness reference: profile
`llama_witness_stage3_qcn_expanded`, `8` prompts, `1` generated token per prompt.
Krasis BF16 WikiText-2 baseline: PPL `5.3307`, BPC `2.4143`, `19,999` scored
tokens, window `2048`, stride `1024`
([PPL log](benchmarks/20260715_qcn_bf16_bf16kv_quality_rust_ppl.log)).

Qwen3.5-35B-A3B witness reference: profile
`llama_witness_qwen35_expanded_thinking_off`, `10` prompts, `1` generated token
per prompt. Krasis BF16 WikiText-2 baseline: PPL `5.8038`, BPC `2.5370`,
`19,999` scored tokens, window `2048`, stride `1024`
([PPL log](benchmarks/20260715_q35b_bf16_bf16kv_quality_rust_ppl.log)).

Qwen3.5-122B-A10B witness reference: profile `llama_witness_q122b_expanded`,
`14` prompts, `1` generated token per prompt. Krasis BF16 WikiText-2 baseline:
PPL `4.4362`, BPC `2.1493`, `19,999` scored tokens, window `2048`, stride
`1024`
([PPL log](benchmarks/20260715_q122b_bf16_bf16kv_quality_rust_ppl.log)).

Qwen3-235B-A22B witness reference: profile `llama_witness_q235_thinking_off`,
`14` prompts, `1` generated token per prompt. Krasis BF16 WikiText-2 baseline:
PPL `4.0326`, BPC `2.0117`, `19,999` scored tokens, window `2048`, stride
`1024`
([PPL log](benchmarks/20260715_q235_bf16_bf16kv_quality_rerun_ppl.log)).

Gemma-4-26B-A4B-it does not yet have a BF16 llama-witness reference. Krasis BF16
diagnostic PPL was captured, but the absolute value is pathological
(`366.7270`), so Gemma quality rows are diagnostic-only and are not marked PASS.

Nemotron Super/Nano do not yet have BF16 llama-witness references. Krasis BF16
diagnostic PPL was captured, but the absolute values are pathological
(`326,150.3391` for Super and `36,988.1497` for Nano), so Nemotron quality rows
are diagnostic-only and are not marked PASS.

| Model | Params | Active params | Quant | BF16 reference | Prompts | PPL | PPL delta vs BF16 | BF16 top-k drift | Prefill argmax | Prefill top-10 | First token | 16-token match run | Decode top-k containment | First-token top-10 overlap | Selected-token logprob delta | Common top-k logprob delta | Result | Logs |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Qwen3.6-35B-A3B | 35.5B text | 3.0B excl. LM head / 3.5B incl. LM head | INT4/HQQ4/k4v4 | BF16 llama-witness, 16-token sequence | 8 | 5.8175 | +3.28% | avg 0.254%, max 0.783% | 8/8 | 8/8 | 8/8 | avg 9.0, min 4 | avg 83.6%; weighted 95/116 (81.9%) | avg 8.625/10, min 7/10 | mean abs 0.0128, max abs 0.0383 | mean abs 0.6692, max abs 2.2443 | PASS | [quality log](benchmarks/20260713_qwen36_quality_hqq4_hqq6.log), [PPL log](benchmarks/20260714_qwen36_quality_rust_ppl_hqq4_k4v4.log) |
| Qwen3.6-35B-A3B | 35.5B text | 3.0B excl. LM head / 3.5B incl. LM head | INT4/HQQ6/k6v6 | BF16 llama-witness, 16-token sequence | 8 | 5.6895 | +1.00% | avg 0.200%, max 0.695% | 8/8 | 8/8 | 8/8 | avg 10.0, min 4 | avg 85.2%; weighted 97/116 (83.6%) | avg 9.0/10, min 8/10 | mean abs 0.0118, max abs 0.0325 | mean abs 0.5771, max abs 2.7713 | PASS | [quality log](benchmarks/20260713_qwen36_quality_hqq4_hqq6.log), [PPL log](benchmarks/20260714_qwen36_quality_rust_ppl_hqq6_k6v6.log) |
| Qwen3-Coder-Next | 80B class | n/a | INT4/HQQ4/k4v4 | BF16 llama-witness, 1-token sequence | 8 | 5.6149 | +5.33% | avg 0.382%, max 1.564% | 8/8 | 8/8 | 8/8 | avg 1.0, min 1 | avg 100.0%; weighted 8/8 (100.0%) | avg 8.750/10, min 8/10 | mean abs 0.0113, max abs 0.0578 | mean abs 0.6784, max abs 2.6466 | PASS | [witness log](benchmarks/20260715_qcn_hqq4_k4v4_witness_compare.log), [summary](benchmarks/20260715_qcn_hqq4_k4v4_reference_test_summary.json), [PPL log](benchmarks/20260715_qcn_hqq4_k4v4_quality_rust_ppl.log) |
| Qwen3-Coder-Next | 80B class | n/a | INT4/HQQ6/k6v6 | BF16 llama-witness, 1-token sequence | 8 | 5.3326 | +0.04% | avg 0.297%, max 1.260% | 8/8 | 8/8 | 8/8 | avg 1.0, min 1 | avg 100.0%; weighted 8/8 (100.0%) | avg 9.125/10, min 8/10 | mean abs 0.0124, max abs 0.0346 | mean abs 0.6205, max abs 2.1678 | PASS | [witness log](benchmarks/20260715_qcn_hqq6_k6v6_witness_compare.log), [summary](benchmarks/20260715_qcn_hqq6_k6v6_reference_test_summary.json), [PPL log](benchmarks/20260715_qcn_hqq6_k6v6_quality_rust_ppl.log) |
| Qwen3.5-35B-A3B | 35B class | 3B class | INT4/HQQ4/k4v4 | BF16 llama-witness, 1-token sequence | 10 | 6.0853 | +4.85% | avg 0.545%, max 3.579% | 10/10 | 10/10 | 10/10 | avg 1.0, min 1 | avg 100.0%; weighted 10/10 (100.0%) | avg 8.600/10, min 6/10 | mean abs 0.0284, max abs 0.1105 | mean abs 0.8735, max abs 3.7604 | PASS | [witness log](benchmarks/20260715_q35b_hqq4_k4v4_witness_compare.log), [summary](benchmarks/20260715_q35b_hqq4_k4v4_reference_test_summary.json), [PPL log](benchmarks/20260715_q35b_hqq4_k4v4_quality_rust_ppl.log) |
| Qwen3.5-35B-A3B | 35B class | 3B class | INT4/HQQ6/k6v6 | BF16 llama-witness, 1-token sequence | 10 | 5.8505 | +0.80% | avg 0.314%, max 2.096% | 10/10 | 10/10 | 10/10 | avg 1.0, min 1 | avg 100.0%; weighted 10/10 (100.0%) | avg 9.100/10, min 8/10 | mean abs 0.0296, max abs 0.2056 | mean abs 0.6624, max abs 3.9677 | PASS | [witness log](benchmarks/20260715_q35b_hqq6_k6v6_witness_compare.log), [summary](benchmarks/20260715_q35b_hqq6_k6v6_reference_test_summary.json), [PPL log](benchmarks/20260715_q35b_hqq6_k6v6_quality_rust_ppl.log) |
| Qwen3.5-122B-A10B | 122B class | 10B class | INT4/HQQ4/k4v4 | BF16 llama-witness, 1-token sequence | 14 | 4.5805 | +3.25% | avg 1.195%, max 4.089% | 14/14 | 14/14 | 14/14 | avg 1.0, min 1 | avg 100.0%; weighted 14/14 (100.0%) | avg 8.714/10, min 7/10 | mean abs 0.0866, max abs 0.4776 | mean abs 0.7004, max abs 3.4680 | PASS | [witness log](benchmarks/20260715_q122b_hqq4_k4v4_witness_compare.log), [summary](benchmarks/20260715_q122b_hqq4_k4v4_reference_test_summary.json), [PPL log](benchmarks/20260715_q122b_hqq4_k4v4_quality_rust_ppl.log) |
| Qwen3.5-122B-A10B | 122B class | 10B class | INT4/HQQ6/k6v6 | BF16 llama-witness, 1-token sequence | 14 | 4.4037 | -0.73% | avg 0.617%, max 3.057% | 14/14 | 14/14 | 14/14 | avg 1.0, min 1 | avg 100.0%; weighted 14/14 (100.0%) | avg 9.143/10, min 7/10 | mean abs 0.0666, max abs 0.4021 | mean abs 0.5264, max abs 2.0327 | PASS | [witness log](benchmarks/20260715_q122b_hqq6_k6v6_witness_compare.log), [summary](benchmarks/20260715_q122b_hqq6_k6v6_reference_test_summary.json), [PPL log](benchmarks/20260715_q122b_hqq6_k6v6_quality_rust_ppl.log) |
| Qwen3-235B-A22B | 235B class | 22B class | INT4/HQQ4/k4v4 | BF16 llama-witness, 1-token sequence | 14 | 4.2236 | +4.74% | avg 2.746%, max 15.866% | 14/14 | 14/14 | 14/14 | avg 1.0, min 1 | avg 100.0%; weighted 14/14 (100.0%) | avg 8.643/10, min 7/10 | mean abs 0.0978, max abs 0.5380 | mean abs 0.7719, max abs 2.9509 | PASS | [witness log](benchmarks/20260715_q235_hqq4_k4v4_witness_compare.log), [summary](benchmarks/20260715_q235_hqq4_k4v4_reference_test_summary.json), [PPL log](benchmarks/20260715_q235_hqq4_k4v4_quality_rust_ppl.log) |
| Qwen3-235B-A22B | 235B class | 22B class | INT4/HQQ6/k6v6 | BF16 llama-witness, 1-token sequence | 14 | 4.0252 | -0.18% | avg 2.225%, max 18.889% | 13/14 | 14/14 | 13/14 | avg 0.9, min 0 | avg 100.0%; weighted 14/14 (100.0%) | avg 8.714/10, min 7/10 | mean abs 0.0653, max abs 0.3680 | mean abs 0.5855, max abs 2.7923 | PASS | [witness log](benchmarks/20260715_q235_hqq6_k6v6_witness_compare.log), [summary](benchmarks/20260715_q235_hqq6_k6v6_reference_test_summary.json), [PPL log](benchmarks/20260715_q235_hqq6_k6v6_quality_rust_ppl.log) |
| Step-3.7-Flash | 201.4B total / 199.4B text | 13.9B excl. LM head / 14.4B incl. LM head | INT4/HQQ4/k4v4 | BF16 llama-witness, 16-token sequence | 8 | 1.7671 | +3.80% | avg 3.655%, max 7.582% | 7/8 | 8/8 | 8/8 | avg 7.2, min 1 | avg 68.0%; weighted 71/112 (63.4%) | avg 7.875/10, min 7/10 | mean abs 0.1733, max abs 0.4808 | mean abs 0.4781, max abs 2.1365 | PASS | [quality log](benchmarks/20260714_step37_quality_witness_hqq4_hqq6.log), [PPL log](benchmarks/20260714_step37_quality_rust_ppl_hqq4_k4v4.log) |
| Step-3.7-Flash | 201.4B total / 199.4B text | 13.9B excl. LM head / 14.4B incl. LM head | INT4/HQQ6/k6v6 | BF16 llama-witness, 16-token sequence | 8 | 1.7016 | -0.04% | avg 2.206%, max 3.997% | 8/8 | 8/8 | 8/8 | avg 10.1, min 3 | avg 80.5%; weighted 87/112 (77.7%) | avg 8.0/10, min 7/10 | mean abs 0.1161, max abs 0.3234 | mean abs 0.3612, max abs 1.5521 | PASS | [quality log](benchmarks/20260714_step37_quality_witness_hqq4_hqq6.log), [PPL log](benchmarks/20260714_step37_quality_rust_ppl_hqq6_k6v6.log) |
| Gemma-4-26B-A4B-it | 26B class | 4B class | INT4/HQQ4/k4v4 | Krasis BF16 diagnostic only; no witness | n/a | 395.0428 | +7.72% diagnostic | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | BLOCKED | [PPL log](benchmarks/20260715_gemma4_hqq4_k4v4_quality_rust_ppl.log), [server](benchmarks/20260715_gemma4_hqq4_k4v4_quality_server.log) |
| Gemma-4-26B-A4B-it | 26B class | 4B class | INT4/HQQ6/k6v6 | Krasis BF16 diagnostic only; no witness | n/a | 293.7475 | -19.90% diagnostic | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | BLOCKED | [PPL log](benchmarks/20260715_gemma4_hqq6_k6v6_quality_rust_ppl.log), [server](benchmarks/20260715_gemma4_hqq6_k6v6_quality_server.log) |
| Nemotron-3-Super-120B-A12B | 123.6B total | 12.4B excl. LM head / 12.9B incl. LM head | INT4/HQQ4/k4v4 | Krasis BF16 diagnostic only; no witness | n/a | 379,305.2043 | +16.30% diagnostic | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | BLOCKED | [PPL log](benchmarks/20260714_nemotron_super_quality_rust_ppl_hqq4_k4v4.log), [stdout](benchmarks/20260714_nemotron_super_hqq4_k4v4_quality_rust_ppl_stdout.log), [server](benchmarks/20260714_nemotron_super_hqq4_k4v4_quality_server.log) |
| Nemotron-3-Super-120B-A12B | 123.6B total | 12.4B excl. LM head / 12.9B incl. LM head | INT4/HQQ6/k6v6 | Krasis BF16 diagnostic only; no witness | n/a | 337,774.3820 | +3.56% diagnostic | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | BLOCKED | [PPL log](benchmarks/20260714_nemotron_super_quality_rust_ppl_hqq6_k6v6.log), [stdout](benchmarks/20260714_nemotron_super_hqq6_k6v6_quality_rust_ppl_stdout.log), [server](benchmarks/20260714_nemotron_super_hqq6_k6v6_quality_server.log) |
| Nemotron-3-Nano-30B-A3B | 31.6B total | 3.2B excl. LM head / 3.6B incl. LM head | INT4/HQQ4/k4v4 | Krasis BF16 diagnostic only; no witness | n/a | 35,918.9461 | -2.89% diagnostic | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | BLOCKED | [PPL log](benchmarks/20260714_nemotron_nano_quality_rust_ppl_hqq4_k4v4.log), [stdout](benchmarks/20260714_nemotron_nano_hqq4_k4v4_quality_rust_ppl_stdout.log), [server](benchmarks/20260714_nemotron_nano_hqq4_k4v4_quality_server.log) |
| Nemotron-3-Nano-30B-A3B | 31.6B total | 3.2B excl. LM head / 3.6B incl. LM head | INT4/HQQ6/k6v6 | Krasis BF16 diagnostic only; no witness | n/a | 36,753.1349 | -0.64% diagnostic | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | BLOCKED | [PPL log](benchmarks/20260714_nemotron_nano_quality_rust_ppl_hqq6_k6v6.log), [stdout](benchmarks/20260714_nemotron_nano_hqq6_k6v6_quality_rust_ppl_stdout.log), [server](benchmarks/20260714_nemotron_nano_hqq6_k6v6_quality_server.log) |

Column notes:

- `16-token match run` is the length of the exact generated-token prefix that
  matched BF16 before divergence, averaged across prompts with the minimum
  shown as a worst case.
- `Decode top-k containment` counts generated-token positions where the BF16
  token remained in Krasis's top-k set.
- `Selected-token logprob delta` compares the BF16-selected first token's
  logprob between BF16 and Krasis.
- `Common top-k logprob delta` compares logprobs for first-token candidates
  common to both BF16 and Krasis top-k lists.
- `PPL delta vs BF16` uses the Krasis BF16 runtime baseline above, measured
  through the Rust prefill `/v1/internal/prefill_logits` test endpoint with
  target-token logprobs.
- Some witness profiles generate only one token per prompt. For those rows,
  `16-token match run` reports the available generated-token prefix length.
- Step-3.7's Krasis BF16 PPL baseline is diagnostic only because that runtime
  path is marked unvalidated for correctness. The pass/fail reference remains
  BF16 llama-witness.
- Qwen3-235B BF16 startup builds a quick route heatmap because no BF16-approved
  heatmap is published. The completed rerun reached server ready after that
  startup path and produced the BF16 PPL baseline above.
- Gemma rows are `BLOCKED` because there is no Gemma BF16 llama-witness
  reference yet, and the Krasis BF16 diagnostic PPL path produced a pathological
  absolute PPL. The measured PPL deltas are retained only as raw diagnostics.
- Nemotron rows are `BLOCKED` because there is no Nemotron BF16 llama-witness
  reference yet, and the Krasis BF16 diagnostic PPL path produced pathological
  absolute PPL values. The measured PPL deltas are retained only as raw
  diagnostics from the run.
- `BF16 top-k drift` is normalized Jensen-Shannon divergence over the
  first-token BF16/Krasis top-10 union from llama-witness diagnostics. Lower is
  closer to BF16. This is a top-k diagnostic, not full-vocabulary forced-token
  drift.
