# Advanced Configuration

## Launcher Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--non-interactive` | — | Skip TUI, use saved config |
| `--model-path PATH` | — | HuggingFace model directory (must contain safetensors + config.json) |
| `--num-gpus N` | all | Number of GPUs to use |
| `--selected-gpus IDX` | all | Comma-separated GPU indices (e.g. `0,2`) |
| `--pp-partition STR` | auto | Layer partition across GPUs (e.g. `24,24`) |
| `--expert-divisor VAL` | auto | Expert loading strategy |
| `--kv-dtype` | fp8_e4m3 | KV cache dtype: `fp8_e4m3` (2x VRAM savings) or `bf16` |
| `--gpu-expert-bits` | 4 | GPU expert quantization: `4` or `8` |
| `--cpu-expert-bits` | 4 | CPU expert quantization: `4` or `8` |
| `--attention-quant` | int8 | Attention weight quantization: `int8` or `bf16` |
| `--shared-expert-quant` | int8 | Shared expert quantization: `int8` or `bf16` |
| `--dense-mlp-quant` | int8 | Dense MLP quantization: `int8` or `bf16` |
| `--lm-head-quant` | int8 | LM head quantization: `int8` or `bf16` |
| `--krasis-threads N` | 48 | CPU threads for expert computation |
| `--gguf-path PATH` | — | Optional GGUF file for CPU experts (instead of building from native) |
| `--gpu-prefill-threshold N` | 300 | Minimum tokens to use GPU prefill |
| `--host ADDR` | 0.0.0.0 | Server bind address |
| `--port PORT` | 8012 | Server port |
| `--benchmark` | — | Run benchmark before launching server |
| `--force-load` | — | Force rebuild of cached weights |

## Per-Component Quantization

Krasis lets you quantize each component independently. The defaults (INT8 attention, INT4 experts, FP8 KV) are a good starting point — increase precision if you need better quality, decrease if you need to fit in less VRAM/RAM.

| Component | Options | Default |
|-----------|---------|---------|
| GPU experts | INT4, INT8 | INT4 |
| CPU experts | INT4, INT8 | INT4 |
| Attention | INT8, BF16 | INT8 |
| Shared expert | INT8, BF16 | INT8 |
| Dense MLP | INT8, BF16 | INT8 |
| LM head | INT8, BF16 | INT8 |
| KV cache | FP8, BF16 | FP8 |

Embeddings, norms, and routing gates are always kept at BF16.

## Direct Server (advanced)

For fine-grained control, run the server directly:

```bash
python -m krasis.server \
    --model-path /path/to/model \
    --num-gpus 2 \
    --gpu-expert-bits 4 \
    --cpu-expert-bits 4 \
    --benchmark
```

The server automatically builds an expert heatmap on first run (calibration with a 10K token prompt), then uses HCS (Hot Cached Static) mode to pin the most frequently used experts on GPU for fastest decode.
