# Models in Krasis

- GPU Model is always BF16 Native
- GPU Model is always quantized to INT8 or INT4 if the model was trained at Q4
- CPU Model is either quantized to INT8 or is a downloaded GGUF

Each model is benchmarked for:

- Prefill speed (1x RTX Ada 2000)
- Prefill speed (2x RTX Ada 2000)
- Prefill speed (3x RTX Ada 2000)
- Decode speed

Each line in the results for each model shows:

- GPU strategy
- GPU VRAM used (each GPU)
- CPU strategy
- CPU RAM used
- TTFT
- Prefill tokens/sec (large prompt)
- Decode tokens/sec (med-large output)


## Qwen3-Coder-Next (80B / 3B Active)

TODO

## Nemotron Nano (30B / 3B Active)

TODO

## GLM 4.7 Flash (30B / 3B Active)

TODO

## GPT-OSS 120B (117B / 5.1B Active)

TODO
