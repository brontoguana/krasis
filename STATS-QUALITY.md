# Krasis Quality Stats

Quality comparisons use BF16 llama-witness references and `./dev
witness-compare` for pass/fail checks. Perplexity deltas use a Krasis BF16
runtime baseline on WikiText-2 so HQQ profiles are compared against the same
runtime path.

| Model | Params | Active params | Quant | BF16 reference | Prompts | PPL | PPL delta vs BF16 | BF16 top-k drift | Prefill argmax | Prefill top-10 | First token | Result | Logs |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Qwen3.6-35B-A3B | 35.5B text | 3.0B | INT4/HQQ4/k4v4 | llama-witness BF16 | 8 | 5.8175 | +3.28% | avg 0.254%, max 0.783% | 8/8 | 8/8 | 8/8 | PASS | [quality log](benchmarks/20260713_qwen36_quality_hqq4_hqq6.log), [PPL log](benchmarks/20260714_qwen36_quality_rust_ppl_hqq4_k4v4.log) |
| Qwen3.6-35B-A3B | 35.5B text | 3.0B | INT4/HQQ6/k6v6 | llama-witness BF16 | 8 | 5.6895 | +1.00% | avg 0.200%, max 0.695% | 8/8 | 8/8 | 8/8 | PASS | [quality log](benchmarks/20260713_qwen36_quality_hqq4_hqq6.log), [PPL log](benchmarks/20260714_qwen36_quality_rust_ppl_hqq6_k6v6.log) |
| Qwen3-Coder-Next | 80B class | n/a | INT4/HQQ4/k4v4 | llama-witness BF16 | 8 | 5.6149 | +5.33% | avg 0.382%, max 1.564% | 8/8 | 8/8 | 8/8 | PASS | [witness log](benchmarks/20260715_qcn_hqq4_k4v4_witness_compare.log), [summary](benchmarks/20260715_qcn_hqq4_k4v4_reference_test_summary.json), [PPL log](benchmarks/20260715_qcn_hqq4_k4v4_quality_rust_ppl.log) |
| Qwen3-Coder-Next | 80B class | n/a | INT4/HQQ6/k6v6 | llama-witness BF16 | 8 | 5.3326 | +0.04% | avg 0.297%, max 1.260% | 8/8 | 8/8 | 8/8 | PASS | [witness log](benchmarks/20260715_qcn_hqq6_k6v6_witness_compare.log), [summary](benchmarks/20260715_qcn_hqq6_k6v6_reference_test_summary.json), [PPL log](benchmarks/20260715_qcn_hqq6_k6v6_quality_rust_ppl.log) |
| Qwen3.5-35B-A3B | 35B class | 3B class | INT4/HQQ4/k4v4 | llama-witness BF16 | 10 | 6.0853 | +4.85% | avg 0.545%, max 3.579% | 10/10 | 10/10 | 10/10 | PASS | [witness log](benchmarks/20260715_q35b_hqq4_k4v4_witness_compare.log), [summary](benchmarks/20260715_q35b_hqq4_k4v4_reference_test_summary.json), [PPL log](benchmarks/20260715_q35b_hqq4_k4v4_quality_rust_ppl.log) |
| Qwen3.5-35B-A3B | 35B class | 3B class | INT4/HQQ6/k6v6 | llama-witness BF16 | 10 | 5.8505 | +0.80% | avg 0.314%, max 2.096% | 10/10 | 10/10 | 10/10 | PASS | [witness log](benchmarks/20260715_q35b_hqq6_k6v6_witness_compare.log), [summary](benchmarks/20260715_q35b_hqq6_k6v6_reference_test_summary.json), [PPL log](benchmarks/20260715_q35b_hqq6_k6v6_quality_rust_ppl.log) |
| Qwen3.5-122B-A10B | 122B class | 10B class | INT4/HQQ4/k4v4 | llama-witness BF16 | 14 | 4.5805 | +3.25% | avg 1.195%, max 4.089% | 14/14 | 14/14 | 14/14 | PASS | [witness log](benchmarks/20260715_q122b_hqq4_k4v4_witness_compare.log), [summary](benchmarks/20260715_q122b_hqq4_k4v4_reference_test_summary.json), [PPL log](benchmarks/20260715_q122b_hqq4_k4v4_quality_rust_ppl.log) |
| Qwen3.5-122B-A10B | 122B class | 10B class | INT4/HQQ6/k6v6 | llama-witness BF16 | 14 | 4.4037 | -0.73% | avg 0.617%, max 3.057% | 14/14 | 14/14 | 14/14 | PASS | [witness log](benchmarks/20260715_q122b_hqq6_k6v6_witness_compare.log), [summary](benchmarks/20260715_q122b_hqq6_k6v6_reference_test_summary.json), [PPL log](benchmarks/20260715_q122b_hqq6_k6v6_quality_rust_ppl.log) |
| Qwen3-235B-A22B | 235B class | 22B class | INT4/HQQ4/k4v4 | llama-witness BF16 | 14 | 4.2236 | +4.74% | avg 2.746%, max 15.866% | 14/14 | 14/14 | 14/14 | PASS | [witness log](benchmarks/20260715_q235_hqq4_k4v4_witness_compare.log), [summary](benchmarks/20260715_q235_hqq4_k4v4_reference_test_summary.json), [PPL log](benchmarks/20260715_q235_hqq4_k4v4_quality_rust_ppl.log) |
| Qwen3-235B-A22B | 235B class | 22B class | INT4/HQQ6/k6v6 | llama-witness BF16 | 14 | 4.0252 | -0.18% | avg 2.225%, max 18.889% | 13/14 | 14/14 | 13/14 | PASS | [witness log](benchmarks/20260715_q235_hqq6_k6v6_witness_compare.log), [summary](benchmarks/20260715_q235_hqq6_k6v6_reference_test_summary.json), [PPL log](benchmarks/20260715_q235_hqq6_k6v6_quality_rust_ppl.log) |
| Step-3.7-Flash | 201.4B total / 199.4B text | 13.9B | INT4/HQQ4/k4v4 | llama-witness BF16 | 8 | 1.7671 | +3.80% | avg 3.655%, max 7.582% | 7/8 | 8/8 | 8/8 | PASS | [quality log](benchmarks/20260714_step37_quality_witness_hqq4_hqq6.log), [PPL log](benchmarks/20260714_step37_quality_rust_ppl_hqq4_k4v4.log) |
| Step-3.7-Flash | 201.4B total / 199.4B text | 13.9B | INT4/HQQ6/k6v6 | llama-witness BF16 | 8 | 1.7016 | -0.04% | avg 2.206%, max 3.997% | 8/8 | 8/8 | 8/8 | PASS | [quality log](benchmarks/20260714_step37_quality_witness_hqq4_hqq6.log), [PPL log](benchmarks/20260714_step37_quality_rust_ppl_hqq6_k6v6.log) |
| Gemma-4-26B-A4B-it | 26B class | 4B class | INT4/HQQ4/k4v4 | Krasis BF16 | n/a | 395.0428 | +7.72% diagnostic | n/a | n/a | n/a | n/a | BLOCKED | [PPL log](benchmarks/20260715_gemma4_hqq4_k4v4_quality_rust_ppl.log), [server](benchmarks/20260715_gemma4_hqq4_k4v4_quality_server.log) |
| Gemma-4-26B-A4B-it | 26B class | 4B class | INT4/HQQ6/k6v6 | Krasis BF16 | n/a | 293.7475 | -19.90% diagnostic | n/a | n/a | n/a | n/a | BLOCKED | [PPL log](benchmarks/20260715_gemma4_hqq6_k6v6_quality_rust_ppl.log), [server](benchmarks/20260715_gemma4_hqq6_k6v6_quality_server.log) |
| Nemotron-3-Super-120B-A12B | 123.6B total | 12.4B | INT4/HQQ4/k4v4 | Krasis BF16 | n/a | 379,305.2043 | +16.30% diagnostic | n/a | n/a | n/a | n/a | BLOCKED | [PPL log](benchmarks/20260714_nemotron_super_quality_rust_ppl_hqq4_k4v4.log), [stdout](benchmarks/20260714_nemotron_super_hqq4_k4v4_quality_rust_ppl_stdout.log), [server](benchmarks/20260714_nemotron_super_hqq4_k4v4_quality_server.log) |
| Nemotron-3-Super-120B-A12B | 123.6B total | 12.4B | INT4/HQQ6/k6v6 | Krasis BF16 | n/a | 337,774.3820 | +3.56% diagnostic | n/a | n/a | n/a | n/a | BLOCKED | [PPL log](benchmarks/20260714_nemotron_super_quality_rust_ppl_hqq6_k6v6.log), [stdout](benchmarks/20260714_nemotron_super_hqq6_k6v6_quality_rust_ppl_stdout.log), [server](benchmarks/20260714_nemotron_super_hqq6_k6v6_quality_server.log) |
| Nemotron-3-Nano-30B-A3B | 31.6B total | 3.2B | INT4/HQQ4/k4v4 | Krasis BF16 | n/a | 35,918.9461 | -2.89% diagnostic | n/a | n/a | n/a | n/a | BLOCKED | [PPL log](benchmarks/20260714_nemotron_nano_quality_rust_ppl_hqq4_k4v4.log), [stdout](benchmarks/20260714_nemotron_nano_hqq4_k4v4_quality_rust_ppl_stdout.log), [server](benchmarks/20260714_nemotron_nano_hqq4_k4v4_quality_server.log) |
| Nemotron-3-Nano-30B-A3B | 31.6B total | 3.2B | INT4/HQQ6/k6v6 | Krasis BF16 | n/a | 36,753.1349 | -0.64% diagnostic | n/a | n/a | n/a | n/a | BLOCKED | [PPL log](benchmarks/20260714_nemotron_nano_quality_rust_ppl_hqq6_k6v6.log), [stdout](benchmarks/20260714_nemotron_nano_hqq6_k6v6_quality_rust_ppl_stdout.log), [server](benchmarks/20260714_nemotron_nano_hqq6_k6v6_quality_server.log) |

Column notes:

- `PPL delta vs BF16` uses the per-model Krasis BF16 runtime baseline, measured
  through the Rust prefill `/v1/internal/prefill_logits` test endpoint with
  target-token logprobs.
- Step-3.7's Krasis BF16 PPL baseline is diagnostic only because that runtime
  path is marked unvalidated for correctness. The pass/fail reference remains
  BF16 llama-witness.
- Qwen3-235B BF16 startup builds a quick route heatmap because no BF16-approved
  heatmap is published. The completed rerun reached server ready after that
  startup path and produced the BF16 PPL baseline used for the Q235 deltas.
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
