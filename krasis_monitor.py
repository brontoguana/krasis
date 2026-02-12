#!/usr/bin/env python3
"""Krasis load progress monitor — modern terminal UI.

Watches Krasis log output and system metrics to display real-time
loading progress with vertical bar charts. Fills the entire terminal.

Usage:
    python3 krasis_monitor.py                    # monitor live process
    python3 krasis_monitor.py --log /path/to.log # monitor log file

Press 'q' to quit.
"""

import argparse
import os
import re
import select
import shutil
import sys
import termios
import time
import tty
from dataclasses import dataclass, field
from pathlib import Path

# ANSI colors — modern light theme (slate/teal/white)
BORDER = "\033[38;5;67m"       # muted steel blue — borders & frames
TITLE = "\033[38;5;75m"        # soft sky blue — titles
TEXT = "\033[38;5;252m"        # light grey — primary text
DIM_TEXT = "\033[38;5;244m"    # mid grey — secondary/muted text
ACCENT = "\033[38;5;80m"       # teal — highlights, OK status
WARN = "\033[38;5;221m"        # warm amber — active/warnings
ERR = "\033[38;5;167m"         # soft coral — errors
SUCCESS = "\033[38;5;114m"     # soft green — done/complete
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CLEAR = "\033[2J\033[H"

# Legacy aliases used throughout
GREEN = BORDER
BRIGHT_GREEN = SUCCESS
DIM_GREEN = DIM_TEXT
YELLOW = WARN
RED = ERR
WHITE = TEXT

# 256-color gradient: teal → amber (6 shades for 6 bars)
BAR_GRADIENT = [
    "\033[38;5;73m",   # light teal
    "\033[38;5;79m",   # cyan
    "\033[38;5;115m",  # seafoam
    "\033[38;5;151m",  # sage
    "\033[38;5;186m",  # soft gold
    "\033[38;5;222m",  # warm amber
]

# Box drawing
H = "─"
V = "│"
TL = "┌"
TR = "┐"
BL = "└"
BR = "┘"
T_RIGHT = "├"
T_LEFT = "┤"

BLOCK_FULL = "█"
BLOCKS = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

# Stages shown as vertical bars (not ram_check/watchdog)
BAR_STAGE_KEYS = [
    "gpu_weights", "cache_build", "cache_load",
    "gpu_prefill", "kv_cache", "tokenizer",
]


@dataclass
class StageInfo:
    name: str
    short: str
    progress: float = 0.0
    status: str = "pending"
    detail: str = ""
    elapsed_s: float = 0.0


@dataclass
class MonitorState:
    stages: dict = field(default_factory=dict)
    cache_size_gb: float = 0.0
    cache_target_gb: float = 0.0
    model_name: str = ""
    start_time: float = 0.0
    last_log_line: str = ""
    tracked_pid: int = 0
    cache_build_start: float = 0.0  # when cache building started
    cache_prev_size: float = 0.0     # previous cache size for rate calc
    cache_prev_time: float = 0.0     # previous measurement time
    cache_rate_mbs: float = 0.0      # rolling write rate MB/s
    cache_layers_done: int = 0       # layers completed so far
    cache_layers_total: int = 0      # total layers (auto-detected from model)
    num_model_layers: int = 0        # total model layers (from config)
    base_ram_gb: float = 0.0         # RAM before expert loading (auto-measured)
    context_tokens: int = 0          # total KV cache context length

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.stages = {
            "ram_check": StageInfo("RAM Budget Check", "RAM\nCheck"),
            "watchdog": StageInfo("RAM Watchdog", "Watch\ndog"),
            "gpu_weights": StageInfo("GPU Weights (INT8)", "GPU\nWts"),
            "cache_build": StageInfo("Marlin Cache Build", "Cache\nBuild"),
            "cache_load": StageInfo("Marlin Cache Load", "Cache\nLoad"),
            "gpu_prefill": StageInfo("GPU Prefill Init", "GPU\nPrefill"),
            "kv_cache": StageInfo("KV Cache Alloc", "KV\nCache"),
            "tokenizer": StageInfo("Tokenizer", "Token\nizer"),
        }
        self.cache_size_gb = 0.0
        self.model_name = ""
        self.start_time = time.time()
        self.last_log_line = ""
        self.context_tokens = 0


# ── System info ──────────────────────────────────────────────────────────

def read_meminfo():
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
    except (OSError, ValueError):
        pass
    return info


def read_gpu_info():
    gpus = {}
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 3:
                gpus[int(parts[0])] = (int(parts[1]), int(parts[2]))
    except Exception:
        pass
    return gpus


def find_cache_file(model_path: str):
    cache_dir = Path(model_path) / ".krasis_cache"
    if not cache_dir.exists():
        return None, 0
    for pattern in ["*.bin.tmp", "*.bin"]:
        files = [f for f in cache_dir.glob(pattern) if not str(f).endswith(".lock")]
        if files:
            try:
                return files[0], files[0].stat().st_size
            except OSError:
                pass
    return None, 0


def get_process_start_time(pid: int) -> float:
    """Get process start time as Unix timestamp from /proc."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().split(")")[-1].split()
            # Field 20 after the comm field (0-indexed) = starttime in clock ticks
            starttime_ticks = int(fields[19])
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("btime"):
                    btime = int(line.split()[1])
                    break
            else:
                return 0.0
        hz = os.sysconf("SC_CLK_TCK")
        return btime + starttime_ticks / hz
    except Exception:
        return 0.0


def find_krasis_pid():
    """Find running Krasis python process PID, or 0 if none."""
    import subprocess
    try:
        # Use pgrep -x to match only the python3 process, not bash wrappers
        result = subprocess.run(
            ["pgrep", "-x", "python3"],
            capture_output=True, text=True, timeout=2,
        )
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line or not line.isdigit():
                continue
            pid = int(line)
            # Verify this python3 process is actually running krasis/test_kimi/test_qwen
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmdline = f.read().decode("utf-8", errors="replace")
                if any(kw in cmdline for kw in ("krasis", "test_kimi", "test_qwen")):
                    return pid
            except OSError:
                continue
    except Exception:
        pass
    return 0


# ── Log parsing ──────────────────────────────────────────────────────────

def parse_log_line(line: str, state: MonitorState):
    m = re.search(r"KrasisModel: (\d+) layers", line)
    if m:
        state.num_model_layers = int(m.group(1))
        state.model_name = f"{m.group(1)} layers"

    if "RAM budget" in line:
        state.stages["ram_check"].status = "done"
        state.stages["ram_check"].progress = 1.0
        m = re.search(r"total=([\d.]+) GB needed", line)
        if m:
            state.stages["ram_check"].detail = f"{m.group(1)} GB needed"

    if "INSUFFICIENT RAM" in line:
        state.stages["ram_check"].status = "error"
        state.stages["ram_check"].progress = 1.0

    if "RAM watchdog started" in line:
        state.stages["watchdog"].status = "done"
        state.stages["watchdog"].progress = 1.0
        state.stages["watchdog"].detail = ""

    if "Phase 1: Loading GPU weights" in line:
        state.stages["gpu_weights"].status = "active"

    m = re.search(r"Layer (\d+) loaded.*GPU alloc: (\d+) MB", line)
    if m:
        layer = int(m.group(1))
        state.stages["gpu_weights"].status = "active"
        total_layers = state.num_model_layers if state.num_model_layers > 0 else (layer + 2)
        state.stages["gpu_weights"].progress = min(1.0, (layer + 1) / total_layers)
        state.stages["gpu_weights"].detail = f"L{layer}/{total_layers - 1} ({m.group(2)}MB)"

    if "GPU weights loaded in" in line:
        state.stages["gpu_weights"].status = "done"
        state.stages["gpu_weights"].progress = 1.0
        m = re.search(r"loaded in ([\d.]+)s", line)
        if m:
            state.stages["gpu_weights"].detail = f"{m.group(1)}s"

    if "Phase 2: Loading CPU expert" in line:
        state.stages["cache_build"].status = "active"

    if "Marlin cache:" in line and "layers" in line:
        m = re.search(r"(\d+)/(\d+) layers", line)
        if m:
            state.stages["cache_build"].progress = int(m.group(1)) / int(m.group(2))
            state.stages["cache_build"].detail = f"{m.group(1)}/{m.group(2)} layers"

    # Track per-layer progress for ETA
    m = re.search(r"Layer \d+: \d+ experts in [\d.]+s.*\[(\d+)/(\d+)\]", line)
    if m:
        state.cache_layers_done = int(m.group(1))
        state.cache_layers_total = int(m.group(2))

    if "Acquired Marlin cache build lock" in line:
        state.stages["cache_build"].status = "active"
        state.stages["cache_build"].detail = "Building..."

    if "Marlin cache built:" in line:
        state.stages["cache_build"].status = "done"
        state.stages["cache_build"].progress = 1.0
        m = re.search(r"([\d.]+) GB in ([\d.]+)s", line)
        if m:
            state.stages["cache_build"].detail = f"{m.group(1)}GB in {m.group(2)}s"

    if "Loading Marlin cache" in line:
        state.stages["cache_load"].status = "active"

    if "CPU experts loaded in" in line:
        state.stages["cache_build"].status = "done"
        state.stages["cache_build"].progress = 1.0
        state.stages["cache_load"].status = "done"
        state.stages["cache_load"].progress = 1.0
        m = re.search(r"loaded in ([\d.]+)s", line)
        if m:
            state.stages["cache_load"].detail = f"{m.group(1)}s"

    if "prefill manager created" in line:
        state.stages["gpu_prefill"].status = "done"
        state.stages["gpu_prefill"].progress = 1.0

    # Parse KV cache context length (keep max across ranks — total is smallest rank's capacity)
    m = re.search(r"KV cache: auto-sized to \d+ pages.*?([\d.]+)K tokens", line)
    if m:
        tokens = int(float(m.group(1)) * 1000)
        # Use the MINIMUM across ranks (that's the actual usable context)
        if state.context_tokens == 0:
            state.context_tokens = tokens
        else:
            state.context_tokens = min(state.context_tokens, tokens)

    if "Model fully loaded in" in line:
        for s in state.stages.values():
            if s.status != "error":
                s.status = "done"
                s.progress = 1.0
        m = re.search(r"loaded in ([\d.]+)s", line)
        if m:
            state.stages["tokenizer"].detail = f"Total: {m.group(1)}s"

    state.last_log_line = line.strip()[-120:] if line.strip() else state.last_log_line


# ── Rendering ────────────────────────────────────────────────────────────

def bar_char(fraction: float) -> str:
    return BLOCKS[max(0, min(8, int(fraction * 8)))]


def render_vertical_bar(height: int, fill_pct: float, color: str, bar_w: int) -> list:
    """Render a vertical bar as list of strings (top to bottom), each bar_w chars wide."""
    lines = []
    filled_rows = fill_pct / 100.0 * height
    for row in range(height):
        pos = height - 1 - row
        if pos < int(filled_rows):
            lines.append(f"{color}{BLOCK_FULL * bar_w}{RESET}")
        elif pos < filled_rows:
            ch = bar_char(filled_rows - int(filled_rows))
            lines.append(f"{color}{ch * bar_w}{RESET}")
        else:
            lines.append(f"{DIM_GREEN}{'·' * bar_w}{RESET}")
    return lines


def status_color(status: str) -> str:
    if status == "done":
        return BRIGHT_GREEN
    elif status == "active":
        return YELLOW
    elif status == "error":
        return RED
    return DIM_GREEN


def bar_color(idx: int, status: str) -> str:
    """Get gradient color for bar index, respecting status overrides."""
    if status == "error":
        return RED
    if status == "pending":
        return DIM_GREEN
    # Use gradient for active/done bars
    return BAR_GRADIENT[idx % len(BAR_GRADIENT)]


def h_bar(used: float, total: float, bar_len: int, color: str) -> str:
    """Horizontal bar using block chars."""
    pct = (used / total) if total > 0 else 0
    filled = int(pct * bar_len)
    return f"{color}{BLOCK_FULL * filled}{RESET}{DIM_GREEN}{'·' * (bar_len - filled)}{RESET}"


def pad_row(content: str, width: int) -> str:
    """Wrap content in box borders, padding to width."""
    visible = re.sub(r'\033\[[0-9;]*m', '', content)
    pad = width - 2 - len(visible)
    return f"{GREEN}{V}{RESET}{content}{' ' * max(0, pad)}{GREEN}{V}{RESET}"


def render(state: MonitorState, width: int, height: int) -> str:
    lines = []
    inner = width - 2

    # ── Title bar ────────────────────────────────────────────────────
    elapsed = time.time() - state.start_time
    title = " KRASIS MONITOR "
    hrs, rem = divmod(int(elapsed), 3600)
    mins, secs = divmod(rem, 60)
    elapsed_str = f" {hrs}:{mins:02d}:{secs:02d} "
    model_str = f" {state.model_name} " if state.model_name else ""
    pad = inner - len(title) - len(elapsed_str) - len(model_str)
    lines.append(f"{BORDER}{BOLD}{TL}{title}{H * max(0, pad)}{model_str}{elapsed_str}{TR}{RESET}")

    # ── Layout math ──────────────────────────────────────────────────
    mem = read_meminfo()
    gpus = read_gpu_info()
    cache_stage = state.stages["cache_build"]
    n_gpu_rows = sum(1 for idx in gpus if gpus[idx][0] > 100)
    n_cache_row = 1 if cache_stage.status in ("active", "done") and state.cache_target_gb > 0 else 0
    # Count all non-bar-chart rows:
    # title(1) + "Loading Stages"(1) + separator(1) = 3 (header)
    # labels(2) + pct(1) = 3 (under bars)
    # separator(1) + checks(1) = 2 (pre-flight)
    # separator(1) = 1 (resources header)
    # RAM(1) + gap_lines between h-bars + GPU rows + cache row
    # separator(1) + status(1) + log(1) = 3
    # separator(1) + footer(1) + bottom_border(1) = 3
    n_h_bars = 1 + n_gpu_rows + n_cache_row
    n_h_gaps = max(0, n_h_bars - 1)
    fixed_rows = 3 + 3 + 2 + 1 + n_h_bars + n_h_gaps + 3 + 3
    bar_height = max(4, height - fixed_rows)

    # ── Stage bar chart (6 bars, no ram_check/watchdog) ──────────────
    num_bars = len(BAR_STAGE_KEYS)
    label_w = 6  # left margin for y-axis / resource labels
    gap = 1
    avail_for_bars = inner - label_w - 1
    bar_w = max(2, (avail_for_bars - (num_bars - 1) * gap) // num_bars)
    bars_section_w = num_bars * bar_w + (num_bars - 1) * gap

    lines.append(pad_row(f"  {'Loading Stages':^{inner - 2}}", width))
    lines.append(f"{GREEN}{T_RIGHT}{H * inner}{T_LEFT}{RESET}")

    # Build bar columns with gradient colors
    bar_columns = []
    for bi, key in enumerate(BAR_STAGE_KEYS):
        s = state.stages[key]
        col_color = bar_color(bi, s.status)
        bar_columns.append(render_vertical_bar(bar_height, s.progress * 100, col_color, bar_w))

    for row in range(bar_height):
        row_str = ""
        if row == 0:
            row_str += f" {DIM_GREEN}100%{RESET} "
        elif row == bar_height // 2:
            row_str += f" {DIM_GREEN} 50%{RESET} "
        elif row == bar_height - 1:
            row_str += f" {DIM_GREEN}  0%{RESET} "
        else:
            row_str += " " * label_w

        for ci, col in enumerate(bar_columns):
            row_str += col[row]
            if ci < len(bar_columns) - 1:
                row_str += " " * gap

        lines.append(pad_row(row_str, width))

    # ── Bar labels (2 lines) ─────────────────────────────────────────
    for line_idx in range(2):
        label_str = " " * label_w
        for bi, key in enumerate(BAR_STAGE_KEYS):
            s = state.stages[key]
            parts = s.short.split("\n")
            col_color = bar_color(bi, s.status)
            txt = parts[line_idx] if line_idx < len(parts) else ""
            label_str += f"{col_color}{txt:^{bar_w}}{RESET}"
            if bi < num_bars - 1:
                label_str += " " * gap
        lines.append(pad_row(label_str, width))

    # ── Percentage labels ────────────────────────────────────────────
    pct_str = " " * label_w
    for bi, key in enumerate(BAR_STAGE_KEYS):
        s = state.stages[key]
        col_color = bar_color(bi, s.status)
        pct = int(s.progress * 100)
        if s.status == "done":
            txt = "DONE"
        elif s.status == "error":
            txt = "ERR!"
        elif pct > 0:
            txt = f"{pct}%"
        else:
            txt = "---"
        pct_str += f"{col_color}{txt:^{bar_w}}{RESET}"
        if bi < num_bars - 1:
            pct_str += " " * gap
    lines.append(pad_row(pct_str, width))

    # ── Pre-flight checks (text, not bars) ───────────────────────────
    lines.append(f"{GREEN}{T_RIGHT}{H * inner}{T_LEFT}{RESET}")

    ram_chk = state.stages["ram_check"]
    wdog = state.stages["watchdog"]

    if ram_chk.status == "done":
        ram_icon = f"{BRIGHT_GREEN}OK{RESET}"
    elif ram_chk.status == "error":
        ram_icon = f"{RED}FAIL{RESET}"
    else:
        ram_icon = f"{DIM_GREEN}--{RESET}"
    ram_detail = f"  {DIM_GREEN}{ram_chk.detail}{RESET}" if ram_chk.detail else ""

    if wdog.status == "done":
        wdog_icon = f"{BRIGHT_GREEN}OK{RESET}"
    else:
        wdog_icon = f"{DIM_GREEN}--{RESET}"
    wdog_detail = f"  {DIM_GREEN}{wdog.detail}{RESET}" if wdog.detail else ""

    # Process alive indicator
    if state.tracked_pid and os.path.exists(f"/proc/{state.tracked_pid}"):
        proc_icon = f"{BRIGHT_GREEN}ALIVE{RESET}"
    elif state.tracked_pid:
        proc_icon = f"{RED}DEAD{RESET}"
    else:
        proc_icon = f"{DIM_GREEN}NONE{RESET}"

    ctx_str = ""
    if state.context_tokens > 0:
        ctx_str = f"    Context: {ACCENT}{state.context_tokens // 1000}K tokens{RESET}"
    lines.append(pad_row(f" RAM Check: [{ram_icon}]{ram_detail}    Watchdog: [{wdog_icon}]{wdog_detail}    Process: [{proc_icon}]{ctx_str}", width))

    # ── System Resources ─────────────────────────────────────────────
    lines.append(f"{GREEN}{T_RIGHT}{H * inner}{T_LEFT}{RESET}")

    suffix_w = 14
    res_bar_len = bars_section_w - suffix_w
    if res_bar_len < 10:
        res_bar_len = inner - label_w - suffix_w

    # RAM bar
    if mem:
        total_gb = mem.get("MemTotal", 0) / 1024 / 1024
        avail_gb = mem.get("MemAvailable", 0) / 1024 / 1024
        used_gb = total_gb - avail_gb
        used_pct = (used_gb / total_gb * 100) if total_gb > 0 else 0
        ram_color = "\033[38;5;78m" if used_pct < 70 else (YELLOW if used_pct < 90 else RED)
        b = h_bar(used_gb, total_gb, res_bar_len, ram_color)
        lines.append(pad_row(f" RAM   {b} {ram_color}{used_gb:5.0f}{RESET}/{total_gb:.0f}GB", width))

    # GPU bars — only show GPUs with significant allocation (> 100 MB)
    active_gpus = [(idx, gpus[idx]) for idx in sorted(gpus.keys()) if gpus[idx][0] > 100]
    for i, (idx, (used_mb, total_mb)) in enumerate(active_gpus):
        if i > 0 or (mem):  # gap after RAM bar and between GPUs
            lines.append(pad_row("", width))
        used_pct = (used_mb / total_mb * 100) if total_mb > 0 else 0
        gpu_color = "\033[38;5;69m" if used_pct < 70 else ("\033[38;5;141m" if used_pct < 90 else RED)
        b = h_bar(used_mb, total_mb, res_bar_len, gpu_color)
        used_g = used_mb / 1024
        total_g = total_mb / 1024
        lines.append(pad_row(f" GPU{idx}  {b} {gpu_color}{used_g:5.1f}{RESET}/{total_g:.0f}GB", width))

    # Cache progress (always show if cache exists or is building)
    if cache_stage.status in ("active", "done") and state.cache_target_gb > 0:
        lines.append(pad_row("", width))  # gap
        cache_color = SUCCESS if cache_stage.status == "done" else YELLOW
        b = h_bar(state.cache_size_gb, max(state.cache_size_gb, state.cache_target_gb), res_bar_len, cache_color)
        lines.append(pad_row(f" CACHE {b} {cache_color}{state.cache_size_gb:5.0f}{RESET}/{max(state.cache_size_gb, state.cache_target_gb):.0f}GB", width))

    # ── Status + log ─────────────────────────────────────────────────
    lines.append(f"{GREEN}{T_RIGHT}{H * inner}{T_LEFT}{RESET}")

    active = [s for s in state.stages.values() if s.status == "active"]
    if active:
        status = f" {active[0].name}: {active[0].detail}"
    elif all(s.status == "done" for s in state.stages.values()):
        status = f" {BRIGHT_GREEN}{BOLD}All stages complete!{RESET}"
    else:
        status = " Waiting..."
    lines.append(pad_row(status, width))

    log_display = state.last_log_line[:inner - 2]
    lines.append(pad_row(f" {DIM_GREEN}{log_display}{RESET}", width))

    # ── Footer ───────────────────────────────────────────────────────
    lines.append(f"{GREEN}{T_RIGHT}{H * inner}{T_LEFT}{RESET}")
    pid_str = f"PID {state.tracked_pid}" if state.tracked_pid else "No process"
    footer = f" {DIM_GREEN}Press 'q' to quit | Refresh: 2s | {pid_str}{RESET}"
    lines.append(pad_row(footer, width))
    lines.append(f"{GREEN}{BL}{H * inner}{BR}{RESET}")

    return "\n".join(lines)


# ── Input handling ───────────────────────────────────────────────────────

def check_keypress():
    """Non-blocking check for keypress. Returns char or None."""
    try:
        fd = sys.stdin.fileno()
        if select.select([fd], [], [], 0.01)[0]:
            ch = os.read(fd, 1)
            if ch:
                return ch.decode('utf-8', errors='ignore')
    except Exception:
        pass
    return None


# ── Model path detection ─────────────────────────────────────────────────

def compute_cache_target_gb(model_path: str) -> float:
    """Compute expected Marlin cache size from model config.json."""
    import json
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return 0.0
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        # Handle nested text_config (VL models)
        if "text_config" in cfg:
            cfg = cfg["text_config"]
        num_layers = cfg.get("num_hidden_layers", 0)
        num_experts = cfg.get("num_experts", cfg.get("num_local_experts",
                     cfg.get("n_routed_experts", 0)))
        hidden = cfg.get("hidden_size", 0)
        intermediate = cfg.get("moe_intermediate_size", cfg.get("intermediate_size", 0))
        first_k_dense = cfg.get("first_k_dense_replace", 0)
        moe_layers = num_layers - first_k_dense
        if moe_layers <= 0 or num_experts <= 0 or hidden <= 0 or intermediate <= 0:
            return 0.0
        # Detect group_size from quantization config
        group_size = 128  # default
        quant_cfg = cfg.get("quantization_config", {})
        if quant_cfg.get("group_size"):
            group_size = quant_cfg["group_size"]
        elif "config_groups" in quant_cfg:
            # compressed-tensors format
            for grp in quant_cfg["config_groups"].values():
                ws = grp.get("weights", {})
                if ws.get("group_size"):
                    group_size = ws["group_size"]
                    break
        # Per expert: w1[intermediate, hidden] + w3[intermediate, hidden] + w2[hidden, intermediate]
        # Marlin packed: weights = K/8 * N * 4, scales = K/group_size * N * 2
        def marlin_bytes(n, k):
            return (k // 8) * n * 4 + (k // group_size) * n * 2
        per_expert = (marlin_bytes(intermediate, hidden) +    # w1
                      marlin_bytes(intermediate, hidden) +    # w3
                      marlin_bytes(hidden, intermediate))     # w2
        total = moe_layers * num_experts * per_expert
        return total / 1024 / 1024 / 1024
    except Exception:
        return 0.0


def find_model_path():
    for p in [
        "/home/main/Documents/Claude/hf-models/Kimi-K2.5",
        "/home/main/models/moonshotai--Kimi-K2.5-A3B",
        "/home/main/Documents/Claude/hf-models/Qwen3-235B-A22B",
    ]:
        if Path(p).exists():
            return p
    return None


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Krasis load progress monitor")
    parser.add_argument("--log", help="Log file to monitor")
    parser.add_argument("--model-path", help="Model directory to monitor cache in")
    parser.add_argument("--cache-target", type=float, default=0.0,
                        help="Expected cache size in GB (0=auto-detect from model config)")
    args = parser.parse_args()

    state = MonitorState()
    model_path = args.model_path or find_model_path()
    if args.cache_target > 0:
        state.cache_target_gb = args.cache_target
    elif model_path:
        state.cache_target_gb = compute_cache_target_gb(model_path)
    # Read model config for layer count
    if model_path:
        try:
            import json
            cfg_path = Path(model_path) / "config.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = json.load(f)
                if "text_config" in cfg:
                    cfg = cfg["text_config"]
                state.num_model_layers = cfg.get("num_hidden_layers", 0)
                first_dense = cfg.get("first_k_dense_replace", 0)
                state.cache_layers_total = state.num_model_layers - first_dense
        except Exception:
            pass
    if state.cache_target_gb <= 0:
        state.cache_target_gb = 100.0  # fallback until we get real data

    # Set up raw terminal for keypress detection
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        sys.stdout.write("\033[?25l")  # hide cursor
        sys.stdout.flush()

        quit_flag = False
        while not quit_flag:
            term = shutil.get_terminal_size((80, 24))
            width = term.columns
            height = term.lines

            # Detect new Krasis process — reset state on new run
            current_pid = find_krasis_pid()
            if current_pid != 0 and current_pid != state.tracked_pid:
                if state.tracked_pid != 0:
                    old_target = state.cache_target_gb
                    state.reset()
                    state.cache_target_gb = old_target
                state.tracked_pid = current_pid
                # Use the process's actual start time
                proc_start = get_process_start_time(current_pid)
                if proc_start > 0:
                    state.start_time = proc_start
            elif current_pid == 0 and state.tracked_pid != 0:
                state.tracked_pid = 0

            # Update cache file size and compute ETA
            if model_path:
                _, size = find_cache_file(model_path)
                if size > 0:
                    now = time.time()
                    state.cache_size_gb = size / 1024 / 1024 / 1024
                    state.stages["cache_build"].progress = min(
                        1.0, state.cache_size_gb / state.cache_target_gb
                    )
                    if state.stages["cache_build"].status == "pending":
                        state.stages["cache_build"].status = "active"
                        state.cache_build_start = now

                    # Compute rolling write rate
                    if state.cache_prev_time > 0:
                        dt = now - state.cache_prev_time
                        if dt > 1.0:
                            delta_gb = state.cache_size_gb - state.cache_prev_size
                            rate = delta_gb / dt  # GB/s
                            # Exponential moving average
                            if state.cache_rate_mbs > 0:
                                state.cache_rate_mbs = 0.7 * state.cache_rate_mbs + 0.3 * (rate * 1024)
                            else:
                                state.cache_rate_mbs = rate * 1024
                            state.cache_prev_size = state.cache_size_gb
                            state.cache_prev_time = now
                    else:
                        state.cache_prev_size = state.cache_size_gb
                        state.cache_prev_time = now

                    # Auto-compute target from per-layer size
                    if state.cache_layers_done > 0 and state.cache_layers_total > 0:
                        per_layer_gb = state.cache_size_gb / state.cache_layers_done
                        state.cache_target_gb = per_layer_gb * state.cache_layers_total
                    elif state.cache_size_gb > state.cache_target_gb:
                        # Safety: bump target if we exceed it (no layer data available)
                        state.cache_target_gb = state.cache_size_gb * 1.1

                    # Build detail string with ETA (layer-based)
                    remaining_layers = state.cache_layers_total - state.cache_layers_done
                    if state.cache_layers_done > 0 and remaining_layers > 0:
                        # Use layer-based ETA: time_so_far / layers_done * layers_remaining
                        elapsed_cache = time.time() - state.cache_build_start
                        if elapsed_cache > 5:
                            eta_s = elapsed_cache / state.cache_layers_done * remaining_layers
                            layer_str = f"L{state.cache_layers_done}/{state.cache_layers_total}"
                            if eta_s < 3600:
                                eta_str = f" {layer_str} ~{int(eta_s // 60)}m{int(eta_s % 60):02d}s left"
                            else:
                                eta_str = f" {layer_str} ~{eta_s / 3600:.1f}h left"
                        else:
                            eta_str = f" L{state.cache_layers_done}/{state.cache_layers_total}"
                    else:
                        eta_str = ""
                    state.stages["cache_build"].detail = (
                        f"{state.cache_size_gb:.1f}/{state.cache_target_gb:.0f} GB{eta_str}"
                    )

                    cache_dir = Path(model_path) / ".krasis_cache"
                    tmps = list(cache_dir.glob("*.bin.tmp"))
                    bins = [b for b in cache_dir.glob("*.bin") if not str(b).endswith(".lock")]
                    if bins and not tmps:
                        state.stages["cache_build"].status = "done"
                        state.stages["cache_build"].progress = 1.0
                        state.cache_target_gb = state.cache_size_gb  # exact match
                        state.stages["cache_build"].detail = f"{state.cache_size_gb:.0f} GB"

            # Read log — auto-detect from process if not specified
            log_path = args.log
            if not log_path and state.tracked_pid:
                # Try to find log from process stdout/stderr redirect
                try:
                    for fd in ("1", "2"):
                        link = os.readlink(f"/proc/{state.tracked_pid}/fd/{fd}")
                        if link.endswith(".log") and Path(link).exists():
                            log_path = link
                            break
                except OSError:
                    pass
            if log_path and Path(log_path).exists():
                try:
                    with open(log_path) as f:
                        for line in f:
                            parse_log_line(line, state)
                except OSError:
                    pass

            # Auto-detect from system state
            gpus = read_gpu_info()
            mem = read_meminfo()
            if mem:
                total = mem.get("MemTotal", 0) / 1024 / 1024
                avail = mem.get("MemAvailable", 0) / 1024 / 1024
                used = total - avail
                if state.stages["cache_build"].status == "done" and state.cache_target_gb > 0:
                    # Estimate base RAM from GPU allocation + overhead (not expert weights)
                    if state.base_ram_gb == 0:
                        gpu_total_mb = sum(u for u, t in gpus.values())
                        state.base_ram_gb = gpu_total_mb / 1024 + 5  # GPU weights + ~5 GB overhead
                    load_threshold = state.base_ram_gb + state.cache_target_gb * 0.9
                    if used > load_threshold:
                        state.stages["cache_load"].status = "done"
                        state.stages["cache_load"].progress = 1.0
                    else:
                        loaded_gb = max(0, used - state.base_ram_gb)
                        progress = min(0.99, loaded_gb / state.cache_target_gb)
                        if progress > 0.01:
                            state.stages["cache_load"].status = "active"
                            state.stages["cache_load"].progress = progress
                            state.stages["cache_load"].detail = f"{loaded_gb:.0f}/{state.cache_target_gb:.0f} GB"

            any_gpu_loaded = False
            for idx, (used_mb, total_mb) in gpus.items():
                if used_mb > 2000:
                    state.stages["gpu_weights"].status = "done"
                    state.stages["gpu_weights"].progress = 1.0
                    any_gpu_loaded = True

            # Infer stage completion from system state (handles monitor restart)
            process_alive = state.tracked_pid and os.path.exists(f"/proc/{state.tracked_pid}")
            if process_alive and any_gpu_loaded:
                # GPU weights loaded → all pre-flight stages must be done
                for key in ("ram_check", "watchdog", "gpu_weights"):
                    state.stages[key].status = "done"
                    state.stages[key].progress = 1.0

                # If cache exists and RAM is loaded, all loading stages complete
                ram_loaded = state.cache_target_gb > 0 and used > state.base_ram_gb + state.cache_target_gb * 0.9
                if state.stages["cache_build"].status == "done" and ram_loaded:
                    for key in ("cache_load", "gpu_prefill", "kv_cache", "tokenizer"):
                        if state.stages[key].status != "done":
                            state.stages[key].status = "done"
                            state.stages[key].progress = 1.0
            elif process_alive and (
                state.stages["cache_build"].status != "pending" or
                state.stages["gpu_weights"].status != "pending"
            ):
                # Process is running and past initial stages
                for key in ("ram_check", "watchdog"):
                    state.stages[key].status = "done"
                    state.stages[key].progress = 1.0

            output = render(state, width, height)
            sys.stdout.write(CLEAR + output)
            sys.stdout.flush()

            # Sleep in small chunks so 'q' is responsive
            for _ in range(20):
                time.sleep(0.1)
                key = check_keypress()
                if key and key.lower() == 'q':
                    quit_flag = True
                    break

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        sys.stdout.write("\033[?25h\n")  # show cursor
        sys.stdout.flush()


if __name__ == "__main__":
    main()
