"""Standardized benchmark for Krasis model + hardware combinations.

Runs after model load (in server.py), before serving begins.
Results are printed to stdout and appended to benchmark_results.log.

Usage (from server.py):
    from krasis.benchmark import KrasisBenchmark
    bench = KrasisBenchmark(model)
    bench.run()
"""

import json
import logging
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional

import torch

from krasis.timing import TIMING

logger = logging.getLogger("krasis.benchmark")

# ANSI formatting
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"
NC = "\033[0m"


def _section(label: str) -> str:
    """Format a highlighted section header."""
    return f"\n{BOLD}{CYAN}▸ {label}{NC}"


class KrasisBenchmark:
    """Standardized benchmark suite for a loaded KrasisModel."""

    PREFILL_LENGTHS = [20000, 35000, 50000]  # tokens — timed runs at each length
    WARMUP_MAX_CHARS = 125000  # ~25K tokens — warmup prompt truncation

    def __init__(self, model):
        self.model = model

        self.decode_tokens = 64
        self.n_runs = int(os.environ.get("KRASIS_BENCH_RUNS", "3"))

        # Ensure instrumentation is OFF for benchmarking
        TIMING.decode = False
        TIMING.prefill = False
        os.environ.pop("KRASIS_DECODE_TIMING", None)

    # ──────────────────────────────────────────────────────────
    # System info collection
    # ──────────────────────────────────────────────────────────

    def _collect_system_info(self) -> Dict:
        """Collect CPU, RAM, GPU info."""
        info = {
            "cpu_model": "unknown",
            "cpu_cores": 0,
            "ram_total_gb": 0,
            "ram_process_gb": 0.0,
            "gpus": [],
        }

        # CPU model
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        except (OSError, ValueError):
            pass

        # Physical cores
        try:
            result = subprocess.run(
                ["lscpu"], capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                cores = 0
                sockets = 1
                for line in result.stdout.split("\n"):
                    if line.startswith("Core(s) per socket:"):
                        cores = int(line.split(":")[-1].strip())
                    elif line.startswith("Socket(s):"):
                        sockets = int(line.split(":")[-1].strip())
                info["cpu_cores"] = cores * sockets
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        # Total RAM
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        info["ram_total_gb"] = kb // (1024 * 1024)
                        break
        except (OSError, ValueError):
            pass

        # Process RSS
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        info["ram_process_gb"] = round(kb / (1024 * 1024), 1)
                        break
        except (OSError, ValueError):
            pass

        # GPUs via nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        info["gpus"].append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "vram_mb": int(parts[2]),
                        })
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        return info

    def _collect_vram_usage(self) -> List[Dict]:
        """Collect per-GPU VRAM usage after model load."""
        usage = []
        for i in range(torch.cuda.device_count()):
            usage.append({
                "index": i,
                "allocated_mb": round(torch.cuda.memory_allocated(i) / (1024 * 1024)),
                "max_allocated_mb": round(torch.cuda.max_memory_allocated(i) / (1024 * 1024)),
            })
        return usage

    def _collect_model_info(self) -> Dict:
        """Collect model config and strategy info."""
        cfg = self.model.cfg
        qcfg = self.model.quant_cfg

        # Read model_type from config.json (not stored on ModelConfig)
        model_type = "unknown"
        config_path = os.path.join(cfg.model_path, "config.json")
        try:
            with open(config_path) as f:
                raw = json.load(f)
            model_type = raw.get("text_config", raw).get("model_type", "unknown")
        except (OSError, json.JSONDecodeError):
            pass

        # Determine strategy description
        lgs = self.model.layer_group_size
        if lgs == 0:
            expert_mode = "persistent"
        else:
            expert_mode = f"layer_grouped({lgs})"
        # Check if HCS is active
        first_mgr = next(iter(self.model.gpu_prefill_managers.values()), None)
        if first_mgr and getattr(first_mgr, '_hcs_initialized', False):
            expert_mode = f"hcs + {expert_mode}"

        # Streaming attention
        if getattr(self.model, '_stream_attn_enabled', False):
            expert_mode += " + stream_attn"

        # Decode mode
        gpu_threshold = getattr(self.model, 'gpu_prefill_threshold', 300)
        if gpu_threshold <= 1:
            decode_mode = f"gpu_decode ({expert_mode})"
        else:
            decode_mode = f"pure_cpu decode, {expert_mode} prefill"

        # KV dtype
        kv_dtype_str = "FP8 E4M3" if self.model.kv_dtype == torch.float8_e4m3fn else "BF16"

        # Actual GPU count: may differ from PP partition length (e.g. HCS uses
        # extra GPUs for pinned experts without splitting layers across them).
        num_gpus_used = getattr(self.model, '_num_gpus', len(self.model.pp_partition))

        return {
            "model_name": os.path.basename(cfg.model_path),
            "model_type": model_type,
            "num_layers": cfg.num_hidden_layers,
            "n_routed_experts": cfg.n_routed_experts,
            "num_experts_per_tok": cfg.num_experts_per_tok,
            "n_shared_experts": cfg.n_shared_experts,
            "is_hybrid": cfg.is_hybrid,
            "num_full_attention_layers": cfg.num_full_attention_layers if cfg.is_hybrid else cfg.num_hidden_layers,
            "pp_partition": self.model.pp_partition,
            "num_gpus": num_gpus_used,
            "gpu_expert_bits": qcfg.gpu_expert_bits,
            "cpu_expert_bits": qcfg.cpu_expert_bits,
            "attention_quant": qcfg.attention,
            "shared_expert_quant": qcfg.shared_expert,
            "dense_mlp_quant": qcfg.dense_mlp,
            "lm_head_quant": qcfg.lm_head,
            "kv_dtype": kv_dtype_str,
            "layer_group_size": lgs,
            "expert_mode": expert_mode,
            "gpu_prefill_threshold": gpu_threshold,
            "decode_mode": decode_mode,
        }

    # ──────────────────────────────────────────────────────────
    # Prompt building
    # ──────────────────────────────────────────────────────────

    def _load_prompt_file(self, filename: str) -> str:
        """Load a prompt file bundled with the package."""
        prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
        prompt_path = os.path.join(prompts_dir, filename)
        if os.path.isfile(prompt_path):
            with open(prompt_path) as f:
                return f.read().strip()
        raise FileNotFoundError(
            f"Benchmark prompt file '{filename}' not found at {prompt_path}"
        )

    def _make_short_prompt(self) -> List[int]:
        """Build decode prompt from file."""
        content = self._load_prompt_file("decode_prompt")
        messages = [{"role": "user", "content": content}]
        return self.model.tokenizer.apply_chat_template(messages)

    def _kv_cache_max_tokens(self) -> int:
        """Return the maximum number of tokens the KV cache can hold."""
        for cache in self.model.kv_caches:
            if cache is not None:
                return cache.max_pages * cache.page_size
        return 100000  # fallback

    def _discover_prefill_files(self) -> List[str]:
        """Discover available prefill_prompt_N files."""
        files = []
        for i in range(1, 100):
            try:
                self._load_prompt_file(f"prefill_prompt_{i}")
                files.append(f"prefill_prompt_{i}")
            except FileNotFoundError:
                break
        return files

    def _make_prefill_prompts(self, n: int, max_tokens_override: int = 0) -> List[List[int]]:
        """Build n different prefill prompts from separate numbered files.

        Loads prefill_prompt_1 through prefill_prompt_N — each file
        covers a completely different domain to ensure different expert
        activation patterns across runs.  Truncates to KV cache capacity
        (with margin for decode tokens).

        If max_tokens_override > 0, truncate to that instead of KV cache limit.
        Cycles through files if n > number of files available.
        """
        # Leave room for decode tokens after prefill
        kv_limit = self._kv_cache_max_tokens() - 200
        cap = max_tokens_override if max_tokens_override > 0 else kv_limit
        cap = min(cap, kv_limit)  # never exceed KV cache

        files = self._discover_prefill_files()

        if not files:
            raise FileNotFoundError(
                "No prefill prompt files found. Expected prefill_prompt_1, prefill_prompt_2, etc."
            )

        prompts = []
        for i in range(n):
            content = self._load_prompt_file(files[i % len(files)])
            messages = [{"role": "user", "content": content}]
            tokens = self.model.tokenizer.apply_chat_template(messages)
            if len(tokens) > cap:
                tokens = tokens[:cap]
            prompts.append(tokens)
        return prompts

    def _make_prefill_prompts_at_lengths(self, lengths: List[int], file_offset: int = 0) -> List[List[int]]:
        """Build one prefill prompt per target length from different files.

        Each prompt uses a different file and is truncated to exactly the
        target token count.  Falls back gracefully if a file doesn't have
        enough tokens (uses what's available).

        file_offset: start from this index into the discovered files list,
        so timed runs can use different files from warmup runs.
        """
        kv_limit = self._kv_cache_max_tokens() - 200
        files = self._discover_prefill_files()

        if not files:
            raise FileNotFoundError(
                "No prefill prompt files found. Expected prefill_prompt_1, prefill_prompt_2, etc."
            )

        prompts = []
        for i, target in enumerate(lengths):
            content = self._load_prompt_file(files[(i + file_offset) % len(files)])
            # Truncate raw text to ~5 chars/tok as a rough upper bound before tokenizing
            # to avoid tokenizing huge texts when we only need a fraction
            char_limit = target * 6
            if len(content) > char_limit:
                content = content[:char_limit]
            messages = [{"role": "user", "content": content}]
            tokens = self.model.tokenizer.apply_chat_template(messages)
            cap = min(target, kv_limit)
            if len(tokens) > cap:
                tokens = tokens[:cap]
            prompts.append(tokens)
        return prompts

    def _make_decode_prompts(self, n: int) -> List[List[int]]:
        """Build n different decode prompts from separate numbered files.

        Loads decode_prompt_1 through decode_prompt_N — each file has a
        different topic to ensure varied expert activation.

        Cycles through files if n > number of files available.
        """
        # Discover available decode prompt files
        files = []
        for i in range(1, 100):
            try:
                self._load_prompt_file(f"decode_prompt_{i}")
                files.append(f"decode_prompt_{i}")
            except FileNotFoundError:
                break

        if not files:
            # Fall back to legacy single file
            files = ["decode_prompt"]

        prompts = []
        for i in range(n):
            content = self._load_prompt_file(files[i % len(files)])
            messages = [{"role": "user", "content": content}]
            prompts.append(self.model.tokenizer.apply_chat_template(messages))
        return prompts

    # ──────────────────────────────────────────────────────────
    # Benchmark phases
    # ──────────────────────────────────────────────────────────

    def _warmup(self):
        """Run 2 short generations to warm up CUDA graphs and caches."""
        print("  Warmup: 2 short generations...")
        short = self._make_short_prompt()
        for i in range(2):
            self.model.generate(short, max_new_tokens=8, temperature=0.6, top_k=50)
        print("  Warmup complete.")

    def _benchmark_prefill(self, prompts: List[List[int]]) -> Dict:
        """Measure prefill speed — one run per prompt (each may be different length).

        Reports the **best** (maximum) tok/s across runs, along with
        the TTFT from that best run.
        """
        runs = []
        for i, tokens in enumerate(prompts):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.model.generate(tokens, max_new_tokens=1, temperature=0.6)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            ttft = getattr(self.model, '_last_ttft', t1 - t0)
            tok_s = len(tokens) / ttft if ttft > 0 else 0
            runs.append({"ttft": round(ttft, 2), "tok_s": round(tok_s, 1), "num_tokens": len(tokens)})

        best_run = max(runs, key=lambda r: r["tok_s"])
        return {
            "best_tok_s": best_run["tok_s"],
            "best_ttft": best_run["ttft"],
            "best_num_tokens": best_run["num_tokens"],
            "runs": runs,
        }

    def _benchmark_decode(self, prompts: List[List[int]]) -> Dict:
        """Measure decode speed — one run per prompt. Captures output from last run."""
        runs = []
        last_generated_ids = []
        for i, tokens in enumerate(prompts):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            generated = self.model.generate(
                tokens, max_new_tokens=self.decode_tokens,
                temperature=0.6, top_k=50,
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            n_gen = len(generated)
            total_s = t1 - t0
            ttft = getattr(self.model, '_last_ttft', 0)
            decode_s = total_s - ttft
            if decode_s > 0 and n_gen > 0:
                tok_s = n_gen / decode_s
                ms_per_tok = (decode_s / n_gen) * 1000
            else:
                tok_s = 0
                ms_per_tok = 0
            runs.append({
                "tok_s": round(tok_s, 2),
                "ms_per_tok": round(ms_per_tok, 1),
                "n_gen": n_gen,
            })
            last_generated_ids = generated

        avg_tok_s = sum(r["tok_s"] for r in runs) / len(runs)
        avg_ms = sum(r["ms_per_tok"] for r in runs) / len(runs)

        # Decode the generated output text for verification
        output_text = ""
        if last_generated_ids:
            try:
                output_text = self.model.tokenizer.decode(last_generated_ids)
            except Exception:
                output_text = f"[{len(last_generated_ids)} tokens, decode failed]"

        return {
            "num_tokens": self.decode_tokens,
            "avg_tok_s": round(avg_tok_s, 2),
            "avg_ms_per_tok": round(avg_ms, 1),
            "runs": runs,
            "output_text": output_text,
        }

    # ──────────────────────────────────────────────────────────
    # Output formatting
    # ──────────────────────────────────────────────────────────

    def _format_results(self, sys_info, model_info, vram_usage,
                        prefill_result, decode_result,
                        prefill_prompt: str = "", decode_prompt: str = "") -> str:
        """Format results as a human-readable block."""
        lines = []
        sep = "=" * 64
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines.append(sep)
        lines.append(f"Krasis Benchmark — {now}")
        lines.append(sep)

        # Model
        arch_parts = [f"{model_info['model_type']}"]
        arch_parts.append(f"{model_info['num_layers']} layers")
        if model_info['n_routed_experts'] > 0:
            arch_parts.append(f"{model_info['n_routed_experts']} experts")
            arch_parts.append(f"top-{model_info['num_experts_per_tok']}")
        if model_info['is_hybrid']:
            n_full = model_info['num_full_attention_layers']
            n_lin = model_info['num_layers'] - n_full
            arch_parts.append(f"{n_full} GQA + {n_lin} linear")

        lines.append(f"Model:            {model_info['model_name']}")
        lines.append(f"Architecture:     {', '.join(arch_parts)}")
        lines.append(f"PP Partition:     {model_info['pp_partition']} ({model_info['num_gpus']} GPUs)")

        # Hardware
        lines.append("")
        lines.append("Hardware:")
        lines.append(f"  CPU:            {sys_info['cpu_model']} ({sys_info['cpu_cores']} cores)")
        lines.append(f"  RAM:            {sys_info['ram_total_gb']} GB total, {sys_info['ram_process_gb']} GB used by process")
        for gpu in sys_info["gpus"]:
            alloc = "N/A"
            for vu in vram_usage:
                if vu["index"] == gpu["index"]:
                    alloc = f"{vu['allocated_mb']} MB allocated"
                    break
            lines.append(f"  GPU {gpu['index']}:          {gpu['name']} ({gpu['vram_mb']} MB), {alloc}")

        # Quantization
        lines.append("")
        lines.append("Quantization:")
        lines.append(f"  GPU experts:    INT{model_info['gpu_expert_bits']} (Marlin)")
        lines.append(f"  CPU experts:    INT{model_info['cpu_expert_bits']}")
        lines.append(f"  Attention:      {model_info['attention_quant'].upper()}")
        lines.append(f"  Shared expert:  {model_info['shared_expert_quant'].upper()}")
        lines.append(f"  Dense MLP:      {model_info['dense_mlp_quant'].upper()}")
        lines.append(f"  LM head:        {model_info['lm_head_quant'].upper()}")
        lines.append(f"  KV cache:       {model_info['kv_dtype']}")

        # Strategy
        lines.append("")
        lines.append("Strategy:")
        lines.append(f"  Layer group size: {model_info['layer_group_size']} ({model_info['expert_mode']})")
        lines.append(f"  Prefill threshold: {model_info['gpu_prefill_threshold']}")
        lines.append(f"  Mode:           {model_info['decode_mode']}")

        # Prefill results
        n_prefill_runs = len(prefill_result["runs"])
        lines.append("")
        lines.append(f"Prefill ({n_prefill_runs} runs at different lengths):")
        for i, run in enumerate(prefill_result["runs"]):
            lines.append(f"  Run {i+1}:  {run['tok_s']:.1f} tok/s, TTFT={run['ttft']:.2f}s ({run['num_tokens']:,} tokens)")
        lines.append(f"  Best:    {prefill_result['best_tok_s']:.1f} tok/s, TTFT={prefill_result['best_ttft']:.2f}s ({prefill_result['best_num_tokens']:,} tokens)")

        # Decode results
        lines.append("")
        lines.append(f"Decode ({self.decode_tokens} tokens, {self.n_runs} runs, different prompts):")
        for i, run in enumerate(decode_result["runs"]):
            lines.append(f"  Run {i+1}:  {run['tok_s']:.2f} tok/s ({run['ms_per_tok']:.1f}ms/tok)")
        lines.append(f"  Average: {decode_result['avg_tok_s']:.2f} tok/s ({decode_result['avg_ms_per_tok']:.1f}ms/tok)")

        # Verification: prompts and generated output
        lines.append("")
        lines.append("Verification:")
        if prefill_prompt:
            # Show first 200 chars of prefill prompt
            truncated = prefill_prompt[:200]
            if len(prefill_prompt) > 200:
                truncated += f"... [{len(prefill_prompt)} chars total]"
            lines.append(f"  Prefill prompt: {truncated}")
        if decode_prompt:
            lines.append(f"  Decode prompt:  {decode_prompt}")
        output_text = decode_result.get("output_text", "")
        if output_text:
            last_run = decode_result["runs"][-1] if decode_result["runs"] else {}
            n_gen = last_run.get("n_gen", "?")
            lines.append(f"  Generated output ({n_gen} tokens):")
            for out_line in output_text.split("\n"):
                lines.append(f"    {out_line}")

        lines.append(sep)
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────
    # Benchmark archive
    # ──────────────────────────────────────────────────────────

    def _archive_benchmark(self, model_info: Dict, report: str) -> Optional[str]:
        """Write benchmark report to benchmarks/<name>.log for archival.

        Returns the relative path (benchmarks/<filename>) or None on failure.
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        benchmarks_dir = os.path.join(repo_root, "benchmarks")

        # Build filename: <model>_<gguf>_<N>gpu_int<X>gpu_int<Y>cpu.log
        model_name = model_info["model_name"]
        gguf_path = getattr(self.model, "gguf_path", None)
        if gguf_path:
            gguf_name = os.path.splitext(os.path.basename(gguf_path))[0]
        else:
            gguf_name = "native"
        num_gpus = model_info["num_gpus"]
        gpu_quant = f"int{model_info['gpu_expert_bits']}gpu"
        cpu_quant = f"int{model_info['cpu_expert_bits']}cpu"
        # Add stream_attn and layer_group_size to filename for differentiation
        suffix = ""
        if getattr(self.model, '_stream_attn_enabled', False):
            lgs = getattr(self.model, 'layer_group_size', 0)
            suffix = f"_stream_lgs{lgs}"
        filename = f"{model_name}_{gguf_name}_{num_gpus}gpu_{gpu_quant}_{cpu_quant}{suffix}.log"
        rel_path = f"benchmarks/{filename}"

        try:
            os.makedirs(benchmarks_dir, exist_ok=True)
            archive_path = os.path.join(benchmarks_dir, filename)
            with open(archive_path, "w") as f:
                f.write(report + "\n")
            return rel_path
        except OSError as e:
            logger.warning("Could not archive benchmark: %s", e)
            return None

    # ──────────────────────────────────────────────────────────
    # Network benchmark (same prompts through HTTP)
    # ──────────────────────────────────────────────────────────

    def _start_temp_server(self, host: str = "127.0.0.1", port: int = 18199) -> "RustServer":
        """Start a temporary Rust HTTP server for network benchmarking."""
        import threading
        from krasis import RustServer

        tokenizer_path = os.path.join(self.model.cfg.model_path, "tokenizer.json")
        max_ctx = self.model.get_max_context_tokens()
        model_name = os.path.basename(self.model.cfg.model_path)

        server = RustServer(
            self.model, host, port, model_name, tokenizer_path, max_ctx,
        )

        t = threading.Thread(target=server.run, daemon=True)
        t.start()

        # Wait for server to be ready
        import urllib.request
        for _ in range(50):
            time.sleep(0.1)
            try:
                req = urllib.request.Request(f"http://{host}:{port}/health")
                with urllib.request.urlopen(req, timeout=1) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                pass
        else:
            raise RuntimeError("Temp server failed to start")

        return server

    def _network_request(
        self, host: str, port: int, prompt_tokens: List[int],
        max_new_tokens: int, temperature: float = 0.6,
    ) -> Dict:
        """Send a prompt over HTTP and measure TTFT + decode speed.

        Returns dict with ttft, decode_tok_s, total_time, num_generated, num_prompt.
        """
        import http.client
        import json as _json

        # Convert tokens back to text for the HTTP request
        prompt_text = self.model.tokenizer.decode(prompt_tokens)
        body = _json.dumps({
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": True,
        }).encode()

        conn = http.client.HTTPConnection(host, port, timeout=600)
        conn.request("POST", "/v1/chat/completions", body=body,
                      headers={"Content-Type": "application/json"})
        resp = conn.getresponse()

        t_start = time.perf_counter()
        t_first = None
        n_tokens = 0
        buf = b""

        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                line, buf = buf.split(b"\n\n", 1)
                line = line.strip()
                if not line or line == b"data: [DONE]":
                    continue
                if line.startswith(b"data: "):
                    if t_first is None:
                        t_first = time.perf_counter()
                    try:
                        data = _json.loads(line[6:])
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            n_tokens += 1
                    except _json.JSONDecodeError:
                        pass

        conn.close()
        t_end = time.perf_counter()

        ttft = (t_first - t_start) if t_first else (t_end - t_start)
        total = t_end - t_start
        decode_time = total - ttft
        decode_tok_s = (n_tokens - 1) / decode_time if decode_time > 0 and n_tokens > 1 else 0

        return {
            "ttft": round(ttft, 2),
            "tok_s": round(len(prompt_tokens) / ttft, 1) if ttft > 0 else 0,
            "decode_tok_s": round(decode_tok_s, 2),
            "decode_ms_per_tok": round(1000 / decode_tok_s, 1) if decode_tok_s > 0 else 0,
            "num_tokens": len(prompt_tokens),
            "num_generated": n_tokens,
            "total_time": round(total, 2),
        }

    def _benchmark_network_prefill(
        self, prompts: List[List[int]], host: str, port: int,
    ) -> Dict:
        """Measure prefill speed over HTTP (SSE streaming, 1 decode token)."""
        runs = []
        for tokens in prompts:
            r = self._network_request(host, port, tokens, max_new_tokens=1)
            runs.append(r)

        best_run = max(runs, key=lambda r: r["tok_s"])
        return {
            "best_tok_s": best_run["tok_s"],
            "best_ttft": best_run["ttft"],
            "best_num_tokens": best_run["num_tokens"],
            "runs": runs,
        }

    def _benchmark_network_decode(
        self, prompts: List[List[int]], host: str, port: int,
    ) -> Dict:
        """Measure decode speed over HTTP."""
        runs = []
        for tokens in prompts:
            r = self._network_request(host, port, tokens,
                                       max_new_tokens=self.decode_tokens)
            runs.append({
                "tok_s": r["decode_tok_s"],
                "ms_per_tok": r["decode_ms_per_tok"],
                "n_gen": r["num_generated"],
            })

        avg_tok_s = sum(r["tok_s"] for r in runs) / len(runs) if runs else 0
        avg_ms = sum(r["ms_per_tok"] for r in runs) / len(runs) if runs else 0

        return {
            "num_tokens": self.decode_tokens,
            "avg_tok_s": round(avg_tok_s, 2),
            "avg_ms_per_tok": round(avg_ms, 1),
            "runs": runs,
        }

    def _format_comparison(
        self, engine_prefill, engine_decode, network_prefill, network_decode,
    ) -> str:
        """Format engine vs network results side by side."""
        lines = []
        lines.append("")
        lines.append(f"{'─' * 64}")
        lines.append(f"  ENGINE vs NETWORK COMPARISON")
        lines.append(f"{'─' * 64}")
        lines.append(f"  {'':30s} {'Engine':>14s} {'Network':>14s} {'Overhead':>10s}")
        lines.append(f"  {'─'*30} {'─'*14} {'─'*14} {'─'*10}")

        # Prefill comparison
        e_pf = engine_prefill["best_tok_s"]
        n_pf = network_prefill["best_tok_s"]
        pf_pct = ((n_pf / e_pf - 1) * 100) if e_pf > 0 else 0
        lines.append(f"  {'Prefill (best tok/s)':30s} {e_pf:>12.1f}   {n_pf:>12.1f}   {pf_pct:>+8.1f}%")

        e_ttft = engine_prefill["best_ttft"]
        n_ttft = network_prefill["best_ttft"]
        ttft_diff = (n_ttft - e_ttft) * 1000  # ms
        lines.append(f"  {'TTFT (s)':30s} {e_ttft:>12.2f}   {n_ttft:>12.2f}   {ttft_diff:>+7.0f}ms")

        # Per-length prefill
        for i, (er, nr) in enumerate(zip(engine_prefill["runs"], network_prefill["runs"])):
            e_t = er["tok_s"]
            n_t = nr["tok_s"]
            pct = ((n_t / e_t - 1) * 100) if e_t > 0 else 0
            lines.append(f"  {'  ' + f'{er[\"num_tokens\"]:,} tokens':30s} {e_t:>12.1f}   {n_t:>12.1f}   {pct:>+8.1f}%")

        lines.append("")

        # Decode comparison
        e_dec = engine_decode["avg_tok_s"]
        n_dec = network_decode["avg_tok_s"]
        dec_pct = ((n_dec / e_dec - 1) * 100) if e_dec > 0 else 0
        lines.append(f"  {'Decode (avg tok/s)':30s} {e_dec:>12.2f}   {n_dec:>12.2f}   {dec_pct:>+8.1f}%")

        e_ms = engine_decode["avg_ms_per_tok"]
        n_ms = network_decode["avg_ms_per_tok"]
        ms_diff = n_ms - e_ms
        lines.append(f"  {'Decode (ms/tok)':30s} {e_ms:>12.1f}   {n_ms:>12.1f}   {ms_diff:>+7.1f}ms")

        lines.append(f"{'─' * 64}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """Run the full benchmark suite. Returns results dict."""
        print(f"\n{BOLD}{'═' * 48}")
        print(f"  KRASIS BENCHMARK")
        print(f"{'═' * 48}{NC}")

        # 1. Collect info
        print(_section("Collecting system info"))
        sys_info = self._collect_system_info()
        model_info = self._collect_model_info()
        vram_usage = self._collect_vram_usage()

        print(f"  Model:    {BOLD}{model_info['model_name']}{NC}")
        print(f"  GPUs:     {len(sys_info['gpus'])}x {sys_info['gpus'][0]['name'] if sys_info['gpus'] else 'none'}")
        print(f"  Strategy: {model_info['decode_mode']}")

        # 2. Build prompts
        # Warmup: n_runs prefill at ~25K tokens + n_runs decode
        # Timed:  len(PREFILL_LENGTHS) prefill at varying lengths + n_runs decode
        n_timed_prefill = len(self.PREFILL_LENGTHS)
        lengths_str = "/".join(f"{l//1000}K" for l in self.PREFILL_LENGTHS)
        print(_section(f"Loading benchmark prompts (warmup + timed at {lengths_str})"))

        warmup_prefill = self._make_prefill_prompts(self.n_runs, max_tokens_override=25000)
        timed_prefill = self._make_prefill_prompts_at_lengths(self.PREFILL_LENGTHS, file_offset=self.n_runs)
        decode_prompts = self._make_decode_prompts(self.n_runs * 2)

        print(f"  Warmup prefill:  {self.n_runs} prompts, ~{len(warmup_prefill[0]):,} tokens each")
        print(f"  Timed prefill:   {n_timed_prefill} prompts at {lengths_str} tokens")
        for i, p in enumerate(timed_prefill):
            print(f"    [{i+1}] {len(p):,} tokens")
        print(f"  Decode:          {len(decode_prompts)} prompts")

        warmup_decode = decode_prompts[:self.n_runs]
        timed_decode = decode_prompts[self.n_runs:]

        # 3. Warmup — compile kernels + warm caches with DIFFERENT prompts than timed runs
        print(_section(f"Warmup ({self.n_runs} prefill + {self.n_runs} decode, all different prompts)"))
        self._warmup()
        t0 = time.perf_counter()
        print(f"  Prefill warmup ({self.n_runs} runs, ~25K tokens each)...")
        self._benchmark_prefill(warmup_prefill)
        print(f"  Decode warmup ({self.n_runs} runs, different prompts)...")
        self._benchmark_decode(warmup_decode)
        warmup_s = time.perf_counter() - t0
        print(f"  Warmup complete ({warmup_s:.1f}s). All kernels compiled.")

        # 4. Prefill benchmark (timed — different lengths: 20K/35K/50K tokens)
        print(_section(f"Running prefill benchmark ({lengths_str} tokens, {n_timed_prefill} runs)"))
        prefill_result = self._benchmark_prefill(timed_prefill)
        for i, run in enumerate(prefill_result["runs"]):
            print(f"  Run {i+1}: {run['tok_s']:.1f} tok/s, TTFT={run['ttft']:.2f}s ({run['num_tokens']:,} tokens)")
        print(f"  {BOLD}Best: {prefill_result['best_tok_s']:.1f} tok/s, TTFT={prefill_result['best_ttft']:.2f}s ({prefill_result['best_num_tokens']:,} tokens){NC}")

        # 5. Decode benchmark (timed — different prompts from warmup)
        print(_section(f"Running decode benchmark ({self.decode_tokens} tokens, {self.n_runs} runs, different prompts)"))
        decode_result = self._benchmark_decode(timed_decode)
        for i, run in enumerate(decode_result["runs"]):
            print(f"  Run {i+1}: {run['tok_s']:.2f} tok/s ({run['ms_per_tok']:.1f}ms/tok)")
        print(f"  {BOLD}Average: {decode_result['avg_tok_s']:.2f} tok/s ({decode_result['avg_ms_per_tok']:.1f}ms/tok){NC}")

        # 6. Network benchmark — same prompts through HTTP
        net_prefill_result = None
        net_decode_result = None
        comparison_text = ""
        try:
            print(_section("Starting Rust HTTP server for network benchmark"))
            bench_host = "127.0.0.1"
            bench_port = 18199
            temp_server = self._start_temp_server(bench_host, bench_port)
            print(f"  Server running on {bench_host}:{bench_port}")

            # Network prefill (same lengths as engine)
            print(_section(f"Network prefill benchmark ({lengths_str} tokens)"))
            net_prefill_result = self._benchmark_network_prefill(
                timed_prefill, bench_host, bench_port)
            for i, run in enumerate(net_prefill_result["runs"]):
                print(f"  Run {i+1}: {run['tok_s']:.1f} tok/s, TTFT={run['ttft']:.2f}s ({run['num_tokens']:,} tokens)")
            print(f"  {BOLD}Best: {net_prefill_result['best_tok_s']:.1f} tok/s, TTFT={net_prefill_result['best_ttft']:.2f}s{NC}")

            # Network decode (same prompts as engine)
            print(_section(f"Network decode benchmark ({self.decode_tokens} tokens, {self.n_runs} runs)"))
            net_decode_result = self._benchmark_network_decode(
                timed_decode, bench_host, bench_port)
            for i, run in enumerate(net_decode_result["runs"]):
                print(f"  Run {i+1}: {run['tok_s']:.2f} tok/s ({run['ms_per_tok']:.1f}ms/tok)")
            print(f"  {BOLD}Average: {net_decode_result['avg_tok_s']:.2f} tok/s ({net_decode_result['avg_ms_per_tok']:.1f}ms/tok){NC}")

            # Comparison
            comparison_text = self._format_comparison(
                prefill_result, decode_result,
                net_prefill_result, net_decode_result,
            )
            print(comparison_text)

            temp_server.stop()
            time.sleep(0.2)
        except Exception as e:
            logger.warning("Network benchmark failed (non-fatal): %s", e, exc_info=True)
            print(f"  {YELLOW}Network benchmark skipped: {e}{NC}")

        # 7. Decode prompt text for the log (use last timed prompt, matching generated output)
        try:
            prefill_prompt_text = self.model.tokenizer.decode(timed_prefill[0])
        except Exception:
            prefill_prompt_text = f"[{len(timed_prefill[0])} tokens]"
        try:
            decode_prompt_text = self.model.tokenizer.decode(timed_decode[-1])
        except Exception:
            decode_prompt_text = "[decode prompt]"

        # 8. Format and write report
        print(_section("Writing results"))
        report = self._format_results(
            sys_info, model_info, vram_usage,
            prefill_result, decode_result,
            prefill_prompt=prefill_prompt_text,
            decode_prompt=decode_prompt_text,
        )
        if comparison_text:
            report += "\n" + comparison_text

        # 9. Archive to benchmarks/ directory
        archive_path = self._archive_benchmark(model_info, report)

        # 10. Final summary
        print(f"\n{BOLD}{'─' * 48}")
        print(f"  BENCHMARK COMPLETE")
        print(f"{'─' * 48}{NC}")
        print(f"  {BOLD}Engine:{NC}")
        print(f"    Prefill: {GREEN}{BOLD}{prefill_result['best_tok_s']:.1f} tok/s{NC}  TTFT={prefill_result['best_ttft']:.2f}s  (best of {lengths_str})")
        print(f"    Decode:  {GREEN}{BOLD}{decode_result['avg_tok_s']:.2f} tok/s{NC}  ({decode_result['avg_ms_per_tok']:.1f}ms/tok)")
        if net_prefill_result and net_decode_result:
            print(f"  {BOLD}Network:{NC}")
            print(f"    Prefill: {GREEN}{BOLD}{net_prefill_result['best_tok_s']:.1f} tok/s{NC}  TTFT={net_prefill_result['best_ttft']:.2f}s")
            print(f"    Decode:  {GREEN}{BOLD}{net_decode_result['avg_tok_s']:.2f} tok/s{NC}  ({net_decode_result['avg_ms_per_tok']:.1f}ms/tok)")
        if archive_path:
            print(f"  Log:     {DIM}{archive_path}{NC}")
        print()

        # 11. Return structured results
        result = {
            "system": sys_info,
            "model": model_info,
            "vram": vram_usage,
            "prefill": prefill_result,
            "decode": decode_result,
        }
        if net_prefill_result:
            result["network_prefill"] = net_prefill_result
        if net_decode_result:
            result["network_decode"] = net_decode_result
        return result
