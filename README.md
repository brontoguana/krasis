# Krasis

Rust + PyO3 MoE runtime for large mixture-of-experts LLMs. Runs 350B+ parameter models on commodity hardware with full GPU prefill and efficient CPU decode.

You can [contact me here](https://forms.gle/ue4nvyvNNHtUZ7MQ7) but please don't ask for help getting Krasis working.  If a model doesn't work or a particular hardware config then you can try to narrow it down and then report an issue.


## Krasis runs MoE LLMs fast on consumer level hardware

Krasis can run MoE language models that are much too large to fit in a consumer GPU (multi-hundred gigabyte modesl with 100 - 500+ billion parameters) on consumer or accessible server hardware you can actually buy without a second mortgage and your own personal power station. 

**Crucially, it runs these models at a speed that is usable.**

## Qwen3-Coder-Next / 856 tok/s prefill / 10.5 tok/s decode##

For example, running Qwen3-Coder-Next (80B params, 146GB BF16) on a single-cpu Epyc server (7742) with 2x Ada 2000 16GB, Krasis achieves **856 tokens/sec prefill** and **10.5 tokens/sec decode**

## How LLMs work

LLM model operation consist of two key steps:

1) Prefill (handling potentially large amounts of input coming into the model)
2) Decode (handling the generation of text after processing the input data)

These are essentially the **LLM reading (prefill) and writing (decode)**.

Prefill is best handled by the GPUs (large amounts of very parallel matrix multiplication, but on typical LLM runtimes its not possible to do more than offload a little of the large model onto the GPU.

The result is that you enter a simple chat prompt and it responds in a reasonable time, but **if you hand it a file to read or try to work with it in an IDE, you wait minutes for it to even start generating text.**

Krasis employs a different approach that utilises the GPU and system RAM more heavily which results in much faster prefill times.  In practice this means the model will generate text at a similar speed (faster in some cases due to other optimisations) **but you wait much less time for an answer, and the model can read files much more quickly.**

## Krasis tradeoffs
In order to achieve these speeds, Krasis has a few requirements.

- **Krasis uses more system RAM than other runtimes**, you may need 2x the model weights worth of system ram (so to run a 100GB model you may need 200GB of system ram), but this is almost always **far more achievable than the equivalent VRAM**.
- Krasis must be given the *BF16 safetensors model** downloaded from (HuggingFace)[https://huggingface.co/]
- Krasis can build everything it needs from this model or if you prefer you can give it a second GGUF model (in addition to the BF16 safetensors model) which takes advantage of more advanced quantisation (e.g. unsloth Q4_K models)
- Krasis currently only works with **NVidia GPUs**
- Krasis **may take some time on the first run** as it is doing a lot of pre-run work to optimise everything, major parts of this are cached for later runs though so they are generally much shorter startup times.
- Krasis optimises models and caches them in .krasis, these can be large so you may need the original model **x3 space** or if you provide a GGUF in addition to the BF16 you may need **4x the space**.

## Known Supported Models and Benchmark Speeds

Speeds reported in the following models are benchmarked on the following hardware:

- Epyc 7742
- DDR4 2666 RAM (8x channels)
- 2x RTX Ada 2000

| Model | Params | BF16 Size | Experts | Attention | Prefill | Decode |
|-------|:------:|:---------:|---------|-----------|:-------:|:------:|
| **Qwen3-Coder-Next** | 80B | 148 GB | 512 routed, top-10 | Hybrid (36 linear + 12 GQA) | 812 tok/s | 10.5 tok/s |
| **Qwen3-235B-A22B** | 235B | 438 GB | 128 routed, top-8 | GQA | 198 tok/s | 1.65 tok/s |
| **DeepSeek V2-Lite** | 16B | 29 GB | 64 + 2 shared, top-6 | MLA | 2,400 tok/s | 5.8 tok/s |
| **GLM-4.7** | 358B | 667 GB | 160 + 1 shared, top-8 | GQA (partial RoPE, bias) | untested | untested |


## Quick Start

### Option A: pipx install (recommended)

```bash
# Install pipx if you don't have it
sudo apt install pipx   # Ubuntu/Debian
# or: pip install --user pipx

# Install Krasis (isolated environment, no conflicts)
pipx install krasis
pipx ensurepath        # adds ~/.local/bin to PATH (restart terminal or source ~/.bashrc)

# PyTorch with CUDA is required — inject into the pipx environment
pipx inject krasis torch --index-url https://download.pytorch.org/whl/cu126

# Download a model into ~/.krasis/models/
huggingface-cli download Qwen/Qwen3-Coder-Next \
    --local-dir ~/.krasis/models/Qwen3-Coder-Next

# Launch
krasis
```

> **Alternative:** If you prefer pip, create a venv first: `python3 -m venv ~/.krasis-env && source ~/.krasis-env/bin/activate && pip install krasis torch --index-url https://download.pytorch.org/whl/cu126`

### Option B: from source

```bash
# Prerequisites (Ubuntu/Debian)
sudo apt update && sudo apt install python3.12-venv

# Clone and run — everything else is automatic
git clone https://github.com/brontoguana/krasis.git
cd krasis
./krasis
```

## Building from Source

The `./krasis` launcher handles building automatically on first run. For manual/development setup:

```bash
git clone https://github.com/brontoguana/krasis.git
cd krasis
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# PyTorch must be installed separately
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Usage

### Interactive Launcher (recommended)

```bash
krasis        # pip install
./krasis      # from source
```

The launcher walks you through a TUI with four screens:

1. **Model selection** — scans `~/.krasis/models/` for safetensors models, shows architecture, layer count, expert count, and estimated RAM
2. **CPU expert source** — build INT4 or INT8 from the native model, or select an existing GGUF file
3. **GPU selection** — multi-select your GPUs (Space to toggle, Enter to confirm)
4. **Configuration editor** — tune all quantization and runtime options with a live VRAM budget display showing per-GPU memory usage and estimated context length

All settings are saved to `~/.krasis/config` and reloaded on subsequent launches.

On the final screen you can choose to launch immediately or run a benchmark first.

### Non-Interactive Launch

```bash
# Use saved config from last TUI session
krasis --non-interactive

# Override specific settings
krasis --non-interactive --model-path /path/to/model --num-gpus 2 --benchmark
```

For launcher flags, per-component quantization options, and direct server usage, see [ADVANCED.md](ADVANCED.md).

### Chat Client

```bash
krasis-chat                          # auto-discovers running servers
krasis-chat --port 8012              # connect to specific port
krasis-chat --url http://host:8012   # connect to remote server
krasis-chat --temperature 0.3        # override sampling temperature
```

The chat client auto-discovers running Krasis servers via `~/.krasis/servers/`. Commands: `/new` (clear history), `/system PROMPT` (change system prompt), `/exit`.

### API

The server exposes an OpenAI-compatible API at `http://localhost:8012/v1/chat/completions` with SSE streaming, compatible with Cursor, OpenCode, and any OpenAI SDK client.

Additional endpoints:
- `GET /health` — server status
- `GET /v1/models` — list loaded models
- `POST /v1/timing` — toggle instrumentation at runtime

## License

AGPL-3.0
