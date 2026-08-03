"""Paged KV cache supporting both MLA and GQA attention.

MLA (DeepSeek/Kimi): compresses all KV heads into a single latent vector per token.
  Split:
    - ckv_cache: [num_layers, num_pages, page_size, kv_lora_rank]
    - kpe_cache: [num_layers, num_pages, page_size, qk_rope_head_dim]
  Combined:
    - kv_cache: [num_layers, num_pages, page_size, kv_lora_rank + qk_rope_head_dim]

GQA (Qwen3): standard K/V caches with head dimension (NHD layout).
    - k_cache: [num_layers, num_pages, page_size, num_kv_heads, head_dim]
    - v_cache: [num_layers, num_pages, page_size, num_kv_heads, head_dim]
"""

import logging
import math
from typing import List, Optional, Tuple

import torch

from krasis.config import ModelConfig

logger = logging.getLogger(__name__)

PAGE_SIZE = 16  # tokens per page

# TRTLLM kernel constraint: block_num % (128 / page_size) == 0
TRTLLM_BLOCK_CONSTRAINT = 128

# Native MLA decode kernels use a 512-wide compressed-KV tile. Models with a
# smaller learned latent rank are zero-padded to this kernel contract; larger
# learned ranks retain their actual dimension.
MLA_CKV_KERNEL_MIN_DIM = 512


class PagedKVCache:
    """Manages paged KV cache for a set of layers on one GPU.

    Allocates a fixed pool of pages at init. Sequences claim pages
    from the free list as they grow; pages are returned on free.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        num_layers: int,
        device: torch.device,
        max_pages: Optional[int] = None,
        kv_dtype: torch.dtype = torch.float8_e4m3fn,
        page_size: int = PAGE_SIZE,
        combined: bool = False,
        max_mb: Optional[int] = None,
        kv_format: str = "fp8",
        layer_indices: Optional[List[int]] = None,
        enable_ring_window: bool = False,
    ):
        self.cfg = cfg
        self.num_layers = num_layers
        self.layer_indices = list(layer_indices) if layer_indices is not None else list(range(num_layers))
        if len(self.layer_indices) != num_layers:
            raise ValueError(
                f"KV cache layer_indices length {len(self.layer_indices)} does not match num_layers {num_layers}"
            )
        self.device = device
        self.page_size = page_size
        self.kv_dtype = kv_dtype
        self.combined = combined
        self.attention_type = cfg.attention_type  # "mla" or "gqa"
        kv_aliases = {
            "fp8": "fp8",
            "fp8_e4m3": "fp8",
            "bf16": "bf16",
            "bfloat16": "bf16",
            "polar4": "polar4",
            "k8v4": "k8v4",
            "k8v6": "k8v6",
            "k7v4": "k7v4",
            "k6v6": "k6v6",
            "k6v4": "k6v4",
            "k4v4": "k4v4",
            "tq4": "tq4",
        }
        self.kv_format_str = kv_aliases.get(kv_format, kv_format)
        self.enable_ring_window = bool(enable_ring_window)
        self.kv_format = 0  # bf16
        if self.kv_format_str == "fp8":
            self.kv_format = 1
        elif self.kv_format_str == "polar4":
            self.kv_format = 2
        elif self.kv_format_str == "k8v4":
            self.kv_format = 3
        elif self.kv_format_str == "tq4":
            self.kv_format = 4
        elif self.kv_format_str == "k6v4":
            self.kv_format = 5
        elif self.kv_format_str == "k7v4":
            self.kv_format = 6
        elif self.kv_format_str == "k6v6":
            self.kv_format = 7
        elif self.kv_format_str == "k8v6":
            self.kv_format = 8
        elif self.kv_format_str == "k4v4":
            self.kv_format = 9

        # Compute cache dimensions based on attention type. DeepSeek-V4 has a
        # distinct K-only sparse-attention state; it is neither MLA nor GQA.
        if cfg.is_deepseek_v4:
            if combined:
                raise ValueError("DeepSeek-V4 does not use a combined MLA KV cache")
            if self.kv_format_str != "bf16" or kv_dtype != torch.bfloat16:
                raise ValueError(
                    "DeepSeek-V4 correctness bring-up requires source-faithful BF16 KV state; "
                    f"got kv_format={self.kv_format_str!r}, dtype={kv_dtype}. "
                    "No quantized-cache fallback is available."
                )
            if cfg.attention_head_dim <= 0 or cfg.index_head_dim <= 0:
                raise ValueError(
                    "DeepSeek-V4 requires positive attention_head_dim and index_head_dim"
                )
            if cfg.sliding_window <= 0:
                raise ValueError("DeepSeek-V4 requires a positive sliding_window")
            if cfg.compress_ratios is None:
                raise ValueError("DeepSeek-V4 requires per-layer compress_ratios")
            self.dsv4_compress_ratios = [
                int(cfg.compress_ratios[layer_idx]) for layer_idx in self.layer_indices
            ]
            invalid_ratios = sorted(
                {ratio for ratio in self.dsv4_compress_ratios if ratio not in (0, 4, 128)}
            )
            if invalid_ratios:
                raise ValueError(
                    "DeepSeek-V4 cache supports the shipped raw/CSA/HCA ratios 0, 4, and 128; "
                    f"got {invalid_ratios}"
                )
            self.ckv_dim = None
            self.kpe_dim = None
            self.num_kv_heads = None
            self.gqa_head_dim = None
            self.kv_cache_dim = cfg.attention_head_dim
            self.variable_gqa_dims = False
        elif cfg.is_mla:
            if combined:
                raise ValueError(
                    "Native MLA k4v4 cache uses separate compressed and positional stores; "
                    "the legacy combined cache layout is unsupported."
                )
            self.ckv_dim = max(cfg.kv_lora_rank, MLA_CKV_KERNEL_MIN_DIM)
            self.kpe_dim = cfg.qk_rope_head_dim
            self.kv_cache_dim = self.ckv_dim + self.kpe_dim
            self.num_kv_heads = None
            self.gqa_head_dim = None
            if self.kv_format != 9:
                raise ValueError(
                    "Native MLA execution currently requires k4v4 KV cache; "
                    f"got kv_format={self.kv_format_str!r}. No fallback cache format is available."
                )
            if self.ckv_dim % 16 != 0 or self.kpe_dim % 16 != 0:
                raise ValueError(
                    "Native MLA k4v4 cache requires dimensions divisible by 16; "
                    f"got compressed_dim={self.ckv_dim}, positional_dim={self.kpe_dim}."
                )
            # One BF16 least-squares scale plus eight packed INT4 bytes for
            # each 16-element block.
            self.ckv_row_bytes = (self.ckv_dim // 16) * 10
            self.kpe_row_bytes = (self.kpe_dim // 16) * 10
        else:
            # GQA: standard K/V with head dimension
            self.ckv_dim = None
            self.kpe_dim = None
            self.num_kv_heads = cfg.num_key_value_heads
            self.gqa_head_dim = cfg.gqa_head_dim
            self.kv_cache_dim = cfg.num_key_value_heads * cfg.gqa_head_dim * 2  # K + V
            self.variable_gqa_dims = bool(getattr(cfg, "gemma4_text", False))
        self.ring_window_gqa = (
            self.enable_ring_window
            and
            cfg.is_gqa
            and bool(getattr(cfg, "gemma4_text", False))
            and int(getattr(cfg, "sliding_window", 0) or 0) > 0
            and self.kv_format in (7, 9)
        )
        self._sliding_window_pages = 0
        if self.ring_window_gqa:
            self._sliding_window_pages = max(1, math.ceil(cfg.sliding_window / page_size))
        self.layer_page_counts: List[int] = []

        # Size from max_mb (preferred) or max_pages (explicit)
        if max_pages is None:
            if max_mb is None:
                max_mb = 2000  # default 2 GB
            budget_bytes = max_mb * 1024 * 1024
            bytes_per_page = self._bytes_per_page()

            # Cap to actual free VRAM minus computed safety margin.
            # Safety = Rust prefill scratch + max prefill intermediate
            # (MoE or dense MLP) + 200 MB base.
            free_bytes, _ = torch.cuda.mem_get_info(device)
            chunk_est = 5000
            # MoE intermediates: [M*topk, 2*moe_inter] + [M*topk, hidden], bf16
            moe_inter = cfg.moe_intermediate_size
            top_k = cfg.num_experts_per_tok
            hidden = cfg.hidden_size
            moe_ws = (chunk_est * top_k * 2 * moe_inter * 2 +
                       chunk_est * top_k * hidden * 2) if moe_inter > 0 and top_k > 0 else 0
            # Dense MLP intermediates: gate + up + cat, peak = M * inter * 8
            dense_inter = cfg.intermediate_size
            dense_ws = chunk_est * dense_inter * 8 if dense_inter > 0 else 0
            prefill_ws = max(moe_ws, dense_ws)
            # Rust prefill scratch estimate
            rust_prefill_scratch = max(cfg.hidden_size, cfg.moe_intermediate_size * 2) * 5000 * 4
            base_headroom = 200 * 1024 * 1024
            safety_bytes = rust_prefill_scratch + prefill_ws + base_headroom
            safety_mb = safety_bytes / (1024 * 1024)
            available_bytes = max(0, free_bytes - safety_bytes)
            if budget_bytes > available_bytes:
                old_mb = budget_bytes / (1024 * 1024)
                budget_bytes = available_bytes
                new_mb = budget_bytes / (1024 * 1024)
                logger.warning(
                    "KV cache: requested %d MB but only %.0f MB available "
                    "(%.0f MB free - %.0f MB safety [%.0f prefill + 200 base]), "
                    "capping to %.0f MB",
                    int(old_mb), available_bytes / (1024 * 1024),
                    free_bytes / (1024 * 1024), safety_mb,
                    prefill_ws / (1024 * 1024), new_mb,
                )

            if self.attention_type == "deepseek_v4":
                max_pages = self._max_dsv4_pages_for_budget(budget_bytes)
            elif getattr(self, "ring_window_gqa", False):
                max_pages = self._max_pages_for_budget(budget_bytes)
            else:
                max_pages = max(64, budget_bytes // bytes_per_page)
            runtime_limit_pages = max(
                1,
                math.ceil(cfg.max_position_embeddings / page_size),
            )
            max_pages = min(max_pages, runtime_limit_pages)
            logger.info(
                "KV cache: %d MB → %d pages (%.1fK tokens, runtime limit %d)",
                max_mb,
                max_pages,
                max_pages * page_size / 1000,
                cfg.max_position_embeddings,
            )

        runtime_limit_pages = max(
            1,
            math.ceil(cfg.max_position_embeddings / page_size),
        )
        max_pages = min(max_pages, runtime_limit_pages)
        self.max_pages = max_pages

        # GQA caches (separate K and V)
        self.k_cache = None
        self.v_cache = None
        # Polar4 caches
        self.k_radius_cache = None
        self.v_radius_cache = None
        self.k_angles_cache = None
        self.v_angles_cache = None
        # MLA caches
        self.ckv_cache = None
        self.kpe_cache = None
        self.kv_cache = None

        # DeepSeek-V4 cache/state stores. Lists are indexed by the local layer
        # offset for this GPU split. Runtime execution consumes their pointers
        # directly from Rust/CUDA; Python only owns the setup allocations.
        self.dsv4_raw_cache = None
        self.dsv4_compressed_cache = None
        self.dsv4_index_cache = None
        self.dsv4_compressor_kv_state = None
        self.dsv4_compressor_score_state = None
        self.dsv4_index_kv_state = None
        self.dsv4_index_score_state = None

        if cfg.is_deepseek_v4:
            self.layer_page_counts = [max_pages] * num_layers
            self.dsv4_raw_cache = []
            self.dsv4_compressed_cache = []
            self.dsv4_index_cache = []
            self.dsv4_compressor_kv_state = []
            self.dsv4_compressor_score_state = []
            self.dsv4_index_kv_state = []
            self.dsv4_index_score_state = []
            context_tokens = max_pages * page_size
            alloc_bytes = 0
            for ratio in self.dsv4_compress_ratios:
                raw = torch.zeros(
                    cfg.sliding_window,
                    cfg.attention_head_dim,
                    dtype=torch.bfloat16,
                    device=device,
                )
                self.dsv4_raw_cache.append(raw)
                alloc_bytes += raw.nbytes

                if ratio == 0:
                    self.dsv4_compressed_cache.append(None)
                    self.dsv4_index_cache.append(None)
                    self.dsv4_compressor_kv_state.append(None)
                    self.dsv4_compressor_score_state.append(None)
                    self.dsv4_index_kv_state.append(None)
                    self.dsv4_index_score_state.append(None)
                    continue

                compressed_rows = max(1, math.ceil(context_tokens / ratio))
                compressed = torch.zeros(
                    compressed_rows,
                    cfg.attention_head_dim,
                    dtype=torch.bfloat16,
                    device=device,
                )
                overlap_width = 2 if ratio == 4 else 1
                state_rows = overlap_width * ratio
                state_cols = overlap_width * cfg.attention_head_dim
                compressor_kv = torch.zeros(
                    state_rows, state_cols, dtype=torch.float32, device=device
                )
                compressor_score = torch.full(
                    (state_rows, state_cols),
                    float("-inf"),
                    dtype=torch.float32,
                    device=device,
                )
                self.dsv4_compressed_cache.append(compressed)
                self.dsv4_compressor_kv_state.append(compressor_kv)
                self.dsv4_compressor_score_state.append(compressor_score)
                alloc_bytes += compressed.nbytes + compressor_kv.nbytes + compressor_score.nbytes

                if ratio == 4:
                    index_cache = torch.zeros(
                        compressed_rows,
                        cfg.index_head_dim,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    index_state_cols = overlap_width * cfg.index_head_dim
                    index_kv = torch.zeros(
                        state_rows, index_state_cols, dtype=torch.float32, device=device
                    )
                    index_score = torch.full(
                        (state_rows, index_state_cols),
                        float("-inf"),
                        dtype=torch.float32,
                        device=device,
                    )
                    self.dsv4_index_cache.append(index_cache)
                    self.dsv4_index_kv_state.append(index_kv)
                    self.dsv4_index_score_state.append(index_score)
                    alloc_bytes += index_cache.nbytes + index_kv.nbytes + index_score.nbytes
                else:
                    self.dsv4_index_cache.append(None)
                    self.dsv4_index_kv_state.append(None)
                    self.dsv4_index_score_state.append(None)

            expected_bytes = self._dsv4_bytes_for_pages(max_pages)
            if alloc_bytes != expected_bytes:
                raise RuntimeError(
                    "DeepSeek-V4 KV allocation disagrees with its budget model: "
                    f"allocated={alloc_bytes}, expected={expected_bytes}"
                )
            alloc_mb = alloc_bytes / (1024**2)
            layout_str = "deepseek-v4-bf16-raw-csa-hca-index"
        elif cfg.is_gqa:
            if self.kv_format == 2:
                # Polar4: radius (BF16) + angles (4-bit uint8)
                num_blocks = (self.num_kv_heads * self.gqa_head_dim) // 16
                self.k_radius_cache = torch.zeros(
                    num_layers, max_pages, page_size, num_blocks,
                    dtype=torch.bfloat16, device=device
                )
                self.v_radius_cache = torch.zeros(
                    num_layers, max_pages, page_size, num_blocks,
                    dtype=torch.bfloat16, device=device
                )
                self.k_angles_cache = torch.zeros(
                    num_layers, max_pages, page_size, num_blocks * 8,
                    dtype=torch.uint8, device=device
                )
                self.v_angles_cache = torch.zeros(
                    num_layers, max_pages, page_size, num_blocks * 8,
                    dtype=torch.uint8, device=device
                )
                alloc_mb = (self.k_radius_cache.nbytes + self.v_radius_cache.nbytes +
                            self.k_angles_cache.nbytes + self.v_angles_cache.nbytes) / (1024**2)
                layout_str = "gqa-polar4"
            elif self.kv_format == 3:
                # k8v4: K in FP8 E4M3, V in Polar4 radius/angle format.
                num_blocks = (self.num_kv_heads * self.gqa_head_dim) // 16
                self.k_cache = torch.zeros(
                    num_layers, max_pages, page_size, self.num_kv_heads, self.gqa_head_dim,
                    dtype=torch.float8_e4m3fn, device=device,
                )
                self.v_radius_cache = torch.zeros(
                    num_layers, max_pages, page_size, num_blocks,
                    dtype=torch.bfloat16, device=device,
                )
                self.v_angles_cache = torch.zeros(
                    num_layers, max_pages, page_size, num_blocks * 8,
                    dtype=torch.uint8, device=device,
                )
                alloc_mb = (
                    self.k_cache.nbytes
                    + self.v_radius_cache.nbytes
                    + self.v_angles_cache.nbytes
                ) / (1024**2)
                layout_str = "gqa-k8v4"
            elif self.kv_format in (5, 6, 7, 8, 9):
                # k4v4/k6v4/k7v4: integer K plus Polar4 V. k6v6/k8v6 use
                # the same slots with integer V scale/indices instead.
                k_packed_bytes = 16 if self.kv_format == 8 else 14 if self.kv_format == 6 else 8 if self.kv_format == 9 else 12
                v_packed_bytes = 12 if self.kv_format in (7, 8) else 8
                if self.variable_gqa_dims:
                    self.k_radius_cache = []
                    self.k_angles_cache = []
                    self.v_radius_cache = []
                    self.v_angles_cache = []
                    self.layer_page_counts = [
                        self._pages_for_layer(global_layer_idx, max_pages)
                        for global_layer_idx in self.layer_indices
                    ]
                    alloc_bytes = 0
                    for global_layer_idx, layer_pages in zip(self.layer_indices, self.layer_page_counts):
                        num_blocks = self._gqa_num_blocks_for_layer(global_layer_idx)
                        kr = torch.zeros(
                            layer_pages, page_size, num_blocks,
                            dtype=torch.bfloat16, device=device,
                        )
                        ka = torch.zeros(
                            layer_pages, page_size, num_blocks * k_packed_bytes,
                            dtype=torch.uint8, device=device,
                        )
                        vr = torch.zeros(
                            layer_pages, page_size, num_blocks,
                            dtype=torch.bfloat16, device=device,
                        )
                        va = torch.zeros(
                            layer_pages, page_size, num_blocks * v_packed_bytes,
                            dtype=torch.uint8, device=device,
                        )
                        alloc_bytes += kr.nbytes + ka.nbytes + vr.nbytes + va.nbytes
                        self.k_radius_cache.append(kr)
                        self.k_angles_cache.append(ka)
                        self.v_radius_cache.append(vr)
                        self.v_angles_cache.append(va)
                    alloc_mb = alloc_bytes / (1024**2)
                else:
                    self.layer_page_counts = [max_pages] * num_layers
                    num_blocks = (self.num_kv_heads * self.gqa_head_dim) // 16
                    self.k_radius_cache = torch.zeros(
                        num_layers, max_pages, page_size, num_blocks,
                        dtype=torch.bfloat16, device=device,
                    )
                    self.k_angles_cache = torch.zeros(
                        num_layers, max_pages, page_size, num_blocks * k_packed_bytes,
                        dtype=torch.uint8, device=device,
                    )
                    self.v_radius_cache = torch.zeros(
                        num_layers, max_pages, page_size, num_blocks,
                        dtype=torch.bfloat16, device=device,
                    )
                    self.v_angles_cache = torch.zeros(
                        num_layers, max_pages, page_size, num_blocks * v_packed_bytes,
                        dtype=torch.uint8, device=device,
                    )
                    alloc_mb = (
                        self.k_radius_cache.nbytes
                        + self.k_angles_cache.nbytes
                        + self.v_radius_cache.nbytes
                        + self.v_angles_cache.nbytes
                    ) / (1024**2)
                if self.kv_format == 8:
                    layout_str = "gqa-k8v6"
                elif self.kv_format == 7:
                    layout_str = "gqa-k6v6"
                elif self.kv_format == 6:
                    layout_str = "gqa-k7v4"
                elif self.kv_format == 9:
                    layout_str = "gqa-k4v4"
                else:
                    layout_str = "gqa-k6v4"
            elif self.kv_format == 4:
                # tq4: K uses 4-bit Lloyd-Max indices + FP16 norm per KV
                # head; V uses 4-bit uniform indices + FP16 scale/zero per KV
                # head. Index tensors pack two 4-bit values per byte.
                packed_per_token = self.num_kv_heads * ((self.gqa_head_dim + 1) // 2)
                self.k_radius_cache = torch.zeros(
                    num_layers, max_pages, page_size, self.num_kv_heads,
                    dtype=torch.float16, device=device,
                )
                self.k_angles_cache = torch.zeros(
                    num_layers, max_pages, page_size, packed_per_token,
                    dtype=torch.uint8, device=device,
                )
                self.v_radius_cache = torch.zeros(
                    num_layers, max_pages, page_size, self.num_kv_heads * 2,
                    dtype=torch.float16, device=device,
                )
                self.v_angles_cache = torch.zeros(
                    num_layers, max_pages, page_size, packed_per_token,
                    dtype=torch.uint8, device=device,
                )
                alloc_mb = (
                    self.k_radius_cache.nbytes
                    + self.k_angles_cache.nbytes
                    + self.v_radius_cache.nbytes
                    + self.v_angles_cache.nbytes
                ) / (1024**2)
                layout_str = "gqa-tq4"
            else:
                # GQA: separate K and V caches [layers, pages, page_size, heads, head_dim]
                if self.variable_gqa_dims:
                    self.k_cache = []
                    self.v_cache = []
                    self.layer_page_counts = [max_pages] * num_layers
                    alloc_bytes = 0
                    for global_layer_idx in self.layer_indices:
                        num_kv_heads = cfg.gqa_num_kv_heads_for_layer(global_layer_idx)
                        head_dim = cfg.gqa_head_dim_for_layer(global_layer_idx)
                        k_layer = torch.zeros(
                            max_pages, page_size, num_kv_heads, head_dim,
                            dtype=kv_dtype, device=device,
                        )
                        v_layer = torch.zeros(
                            max_pages, page_size, num_kv_heads, head_dim,
                            dtype=kv_dtype, device=device,
                        )
                        alloc_bytes += k_layer.nbytes + v_layer.nbytes
                        self.k_cache.append(k_layer)
                        self.v_cache.append(v_layer)
                    alloc_mb = alloc_bytes / (1024**2)
                    layout_str = "gqa-split-variable"
                else:
                    self.layer_page_counts = [max_pages] * num_layers
                    self.k_cache = torch.zeros(
                        num_layers, max_pages, page_size, self.num_kv_heads, self.gqa_head_dim,
                        dtype=kv_dtype, device=device,
                    )
                    self.v_cache = torch.zeros(
                        num_layers, max_pages, page_size, self.num_kv_heads, self.gqa_head_dim,
                        dtype=kv_dtype, device=device,
                    )
                    alloc_mb = (self.k_cache.nbytes + self.v_cache.nbytes) / (1024**2)
                    layout_str = "gqa-split"
        elif combined:
            # TRTLLM MLA format: single combined cache
            self.kv_cache = torch.zeros(
                num_layers, max_pages, page_size, self.kv_cache_dim,
                dtype=kv_dtype, device=device,
            )
            alloc_mb = self.kv_cache.nbytes / (1024**2)
            layout_str = "mla-combined"
        else:
            # MLA k4v4 split format. Each logical 16-element block occupies
            # 10 bytes: one BF16 scale followed by eight packed INT4 bytes.
            self.ckv_cache = torch.zeros(
                num_layers, max_pages, page_size, self.ckv_row_bytes,
                dtype=torch.uint8, device=device,
            )
            self.kpe_cache = torch.zeros(
                num_layers, max_pages, page_size, self.kpe_row_bytes,
                dtype=torch.uint8, device=device,
            )
            alloc_mb = (self.ckv_cache.nbytes + self.kpe_cache.nbytes) / (1024**2)
            layout_str = "mla-k4v4-split"

        logger.info(
            "KV cache allocated: %d layers × %d pages × %d tokens = %.0f MB (%s, %s)",
            num_layers, max_pages, page_size, alloc_mb,
            self.attention_type, layout_str,
        )
        if self.ring_window_gqa:
            sliding_layers = sum(
                1 for layer_idx in self.layer_indices
                if self.cfg.is_sliding_attention_layer(layer_idx)
            )
            ring_tokens = self._sliding_window_pages * self.page_size
            logger.info(
                "KV ring-window enabled: %d sliding layers use %d tokens physical cache; %d full layers use %d tokens",
                sliding_layers,
                ring_tokens,
                len(self.layer_indices) - sliding_layers,
                self.max_pages * self.page_size,
            )

        # Free page tracking
        self._free_pages: List[int] = list(range(max_pages))
        self._free_pages.reverse()  # pop from end

    def _dsv4_bytes_for_pages(self, max_pages: int) -> int:
        """Exact BF16 V4 cache/state bytes for a logical page capacity."""
        if max_pages <= 0:
            raise ValueError(f"DeepSeek-V4 max_pages must be positive, got {max_pages}")
        context_tokens = max_pages * self.page_size
        head_dim = self.cfg.attention_head_dim
        index_dim = self.cfg.index_head_dim
        total = 0
        for ratio in self.dsv4_compress_ratios:
            total += self.cfg.sliding_window * head_dim * 2
            if ratio == 0:
                continue

            compressed_rows = max(1, math.ceil(context_tokens / ratio))
            total += compressed_rows * head_dim * 2
            overlap_width = 2 if ratio == 4 else 1
            state_rows = overlap_width * ratio
            state_cols = overlap_width * head_dim
            total += state_rows * state_cols * 4 * 2  # KV and score FP32 states

            if ratio == 4:
                total += compressed_rows * index_dim * 2
                index_state_cols = overlap_width * index_dim
                total += state_rows * index_state_cols * 4 * 2
        return total

    def _max_dsv4_pages_for_budget(self, budget_bytes: int) -> int:
        """Largest config-bounded V4 context whose measured layout fits budget."""
        runtime_pages = max(
            1, math.ceil(self.cfg.max_position_embeddings / self.page_size)
        )
        minimum_bytes = self._dsv4_bytes_for_pages(1)
        if minimum_bytes > budget_bytes:
            raise ValueError(
                "DeepSeek-V4 KV budget cannot hold even one logical page: "
                f"need {minimum_bytes / (1024**2):.1f} MiB, "
                f"have {budget_bytes / (1024**2):.1f} MiB"
            )
        if self._dsv4_bytes_for_pages(runtime_pages) <= budget_bytes:
            return runtime_pages

        lo, hi = 1, runtime_pages
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._dsv4_bytes_for_pages(mid) <= budget_bytes:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _bytes_per_page(self) -> int:
        if self.attention_type == "deepseek_v4":
            return max(
                1,
                self._dsv4_bytes_for_pages(2) - self._dsv4_bytes_for_pages(1),
            )
        if self.attention_type == "mla":
            return self.page_size * (
                self.ckv_row_bytes + self.kpe_row_bytes
            ) * self.num_layers
        if getattr(self, "ring_window_gqa", False):
            # Ring-window layers cap their physical storage at sliding_window.
            # max_pages still defines the logical context capacity for full layers.
            return self._bytes_for_pages(self.max_pages if hasattr(self, "max_pages") else 1)
        if getattr(self, "variable_gqa_dims", False) and self.kv_format in (5, 6, 7, 8, 9):
            k_packed_bytes = 16 if self.kv_format == 8 else 14 if self.kv_format == 6 else 8 if self.kv_format == 9 else 12
            v_packed_bytes = 12 if self.kv_format in (7, 8) else 8
            bytes_per_token = 0
            for global_layer_idx in self.layer_indices:
                num_blocks = self._gqa_num_blocks_for_layer(global_layer_idx)
                bytes_per_token += num_blocks * (2 + k_packed_bytes + 2 + v_packed_bytes)
            return self.page_size * bytes_per_token
        if self.kv_format == 2:
            # Polar4: 10 bytes per 16 elements
            # num_blocks = kv_cache_dim // 16
            return self.page_size * (self.kv_cache_dim // 16) * 10 * self.num_layers
        if self.kv_format == 3:
            # k8v4: FP8 K (1 byte/element) + Polar4 V (10 bytes/16 elements).
            kv_elems = self.num_kv_heads * self.gqa_head_dim
            num_blocks = kv_elems // 16
            return self.page_size * (kv_elems + num_blocks * 10) * self.num_layers
        if self.kv_format == 5:
            # k6v4: K scale + 12 packed bytes per 16 elements, plus Polar4 V.
            kv_elems = self.num_kv_heads * self.gqa_head_dim
            num_blocks = kv_elems // 16
            return self.page_size * (num_blocks * 14 + num_blocks * 10) * self.num_layers
        if self.kv_format == 9:
            # k4v4: K scale + 8 packed bytes per 16 elements, plus Polar4 V.
            kv_elems = self.num_kv_heads * self.gqa_head_dim
            num_blocks = kv_elems // 16
            return self.page_size * (num_blocks * 10 + num_blocks * 10) * self.num_layers
        if self.kv_format == 6:
            # k7v4: K scale + 14 packed bytes per 16 elements, plus Polar4 V.
            kv_elems = self.num_kv_heads * self.gqa_head_dim
            num_blocks = kv_elems // 16
            return self.page_size * (num_blocks * 16 + num_blocks * 10) * self.num_layers
        if self.kv_format == 7:
            # k6v6: K and V each use scale + 12 packed bytes per 16 elements.
            kv_elems = self.num_kv_heads * self.gqa_head_dim
            num_blocks = kv_elems // 16
            return self.page_size * (num_blocks * 14 + num_blocks * 14) * self.num_layers
        if self.kv_format == 8:
            # k8v6: K uses scale + 16 INT8 bytes; V uses scale + 12 INT6 bytes.
            kv_elems = self.num_kv_heads * self.gqa_head_dim
            num_blocks = kv_elems // 16
            return self.page_size * (num_blocks * 18 + num_blocks * 14) * self.num_layers
        if self.kv_format == 4:
            # tq4: per token/layer = K packed indices + K norms + V packed
            # indices + V scale/zero. This stays exact for arbitrary head_dim.
            packed_per_token = self.num_kv_heads * ((self.gqa_head_dim + 1) // 2)
            meta_per_token = self.num_kv_heads * 6  # 2-byte K norm + 2-byte V scale + 2-byte V zero
            return self.page_size * (packed_per_token * 2 + meta_per_token) * self.num_layers

        elem_size = 1 if self.kv_dtype == torch.float8_e4m3fn else 2
        if getattr(self, "variable_gqa_dims", False):
            elems_per_token = 0
            for global_layer_idx in self.layer_indices:
                elems_per_token += (
                    self.cfg.gqa_num_kv_heads_for_layer(global_layer_idx)
                    * self.cfg.gqa_head_dim_for_layer(global_layer_idx)
                    * 2
                )
            return self.page_size * elems_per_token * elem_size
        return self.page_size * self.kv_cache_dim * elem_size * self.num_layers

    def _pages_for_layer(self, global_layer_idx: int, max_pages: int) -> int:
        if (
            getattr(self, "ring_window_gqa", False)
            and self.cfg.is_sliding_attention_layer(global_layer_idx)
        ):
            return min(max_pages, self._sliding_window_pages)
        return max_pages

    def _bytes_for_pages(self, max_pages: int) -> int:
        if self.attention_type == "deepseek_v4":
            return self._dsv4_bytes_for_pages(max_pages)
        if not (getattr(self, "variable_gqa_dims", False) and self.kv_format in (5, 6, 7, 8, 9)):
            return max_pages * self._bytes_per_page()
        k_packed_bytes = 16 if self.kv_format == 8 else 14 if self.kv_format == 6 else 8 if self.kv_format == 9 else 12
        v_packed_bytes = 12 if self.kv_format in (7, 8) else 8
        total = 0
        for global_layer_idx in self.layer_indices:
            layer_pages = self._pages_for_layer(global_layer_idx, max_pages)
            num_blocks = self._gqa_num_blocks_for_layer(global_layer_idx)
            total += layer_pages * self.page_size * num_blocks * (
                2 + k_packed_bytes + 2 + v_packed_bytes
            )
        return total

    def _max_pages_for_budget(self, budget_bytes: int) -> int:
        if not getattr(self, "ring_window_gqa", False):
            return max(64, budget_bytes // self._bytes_per_page())
        high = max(
            64,
            math.ceil(self.cfg.max_position_embeddings / self.page_size),
            self._sliding_window_pages,
        )
        while self._bytes_for_pages(high) <= budget_bytes and high < (1 << 31):
            high *= 2
        lo, hi = 1, high
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._bytes_for_pages(mid) <= budget_bytes:
                lo = mid
            else:
                hi = mid - 1
        return max(64, lo)

    def _gqa_num_blocks_for_layer(self, global_layer_idx: int) -> int:
        kv_elems = (
            self.cfg.gqa_num_kv_heads_for_layer(global_layer_idx)
            * self.cfg.gqa_head_dim_for_layer(global_layer_idx)
        )
        if kv_elems % 16 != 0:
            raise ValueError(
                f"Layer {global_layer_idx} KV stride {kv_elems} is not divisible by 16; "
                f"{self.kv_format_str} requires 16-element blocks."
            )
        return kv_elems // 16

    def max_num_blocks(self) -> int:
        if getattr(self, "variable_gqa_dims", False):
            return max(self._gqa_num_blocks_for_layer(layer_idx) for layer_idx in self.layer_indices)
        return (self.num_kv_heads * self.gqa_head_dim) // 16

    def max_seq_for_global_layer(self, global_layer_idx: int) -> int:
        if global_layer_idx not in self.layer_indices:
            return self.max_pages * self.page_size
        local_idx = self.layer_indices.index(global_layer_idx)
        if self.layer_page_counts:
            return self.layer_page_counts[local_idx] * self.page_size
        return self.max_pages * self.page_size

    @property
    def max_context_tokens(self) -> int:
        """Maximum number of tokens this cache can hold."""
        return self.max_pages * self.page_size

    @property
    def free_page_count(self) -> int:
        return len(self._free_pages)

    def alloc_pages(self, n: int) -> List[int]:
        """Allocate n pages from the free pool."""
        if n > len(self._free_pages):
            raise RuntimeError(
                f"KV cache exhausted: need {n} pages, have {len(self._free_pages)}"
            )
        pages = [self._free_pages.pop() for _ in range(n)]
        return pages

    def free_pages(self, pages: List[int]):
        """Return pages to the free pool.

        Re-sorts in descending order so pop() always gives sequential
        page indices (0, 1, 2, ...). This is required because Rust decode
        reads the KV cache contiguously — the paged layout must match the
        contiguous layout, which only works when pages are in order.
        """
        self._free_pages.extend(pages)
        self._free_pages.sort(reverse=True)

    # ── MLA cache access ──

    def get_layer_caches(self, layer_offset: int):
        """Get split cache tensors for MLA.

        Returns (ckv_cache, kpe_cache) each [max_pages, page_size, dim].
        """
        assert self.attention_type == "mla" and not self.combined
        return self.ckv_cache[layer_offset], self.kpe_cache[layer_offset]

    def get_combined_layer_cache(self, layer_offset: int) -> torch.Tensor:
        """Get combined KV cache for MLA (TRTLLM format)."""
        assert self.attention_type == "mla" and self.combined
        return self.kv_cache[layer_offset].unsqueeze(0)

    def get_deepseek_v4_layer_caches(self, layer_offset: int) -> dict:
        """Return the exact V4 raw/compressed/index state for one local layer."""
        if self.attention_type != "deepseek_v4":
            raise ValueError("DeepSeek-V4 cache access requested for another architecture")
        if layer_offset < 0 or layer_offset >= self.num_layers:
            raise IndexError(
                f"DeepSeek-V4 layer offset {layer_offset} outside [0, {self.num_layers})"
            )
        return {
            "raw": self.dsv4_raw_cache[layer_offset],
            "compressed": self.dsv4_compressed_cache[layer_offset],
            "index": self.dsv4_index_cache[layer_offset],
            "compressor_kv_state": self.dsv4_compressor_kv_state[layer_offset],
            "compressor_score_state": self.dsv4_compressor_score_state[layer_offset],
            "index_kv_state": self.dsv4_index_kv_state[layer_offset],
            "index_score_state": self.dsv4_index_score_state[layer_offset],
            "ratio": self.dsv4_compress_ratios[layer_offset],
        }

    # ── GQA cache access ──

    def get_gqa_layer_caches(self, layer_offset: int):
        """Get (k_cache, v_cache) for GQA.

        Returns (k, v) each [max_pages, page_size, num_kv_heads, head_dim].
        """
        assert self.attention_type == "gqa"
        return self.k_cache[layer_offset], self.v_cache[layer_offset]


class SequenceKVState:
    """KV cache state for a single sequence (request).

    Tracks which pages are allocated and current position.
    Provides paged KV index arrays.
    """

    def __init__(self, cache: PagedKVCache, seq_id: int = 0):
        self.cache = cache
        self.seq_id = seq_id
        self.pages: List[int] = []
        self.seq_len: int = 0  # number of tokens in cache

    def ensure_capacity(self, new_tokens: int):
        """Ensure we have enough pages for new_tokens more tokens."""
        total_needed = self.seq_len + new_tokens
        pages_needed = (total_needed + self.cache.page_size - 1) // self.cache.page_size
        if pages_needed > len(self.pages):
            extra = pages_needed - len(self.pages)
            new_pages = self.cache.alloc_pages(extra)
            self.pages.extend(new_pages)

    def advance(self, num_tokens: int):
        """Record that num_tokens were appended to the cache."""
        self.seq_len += num_tokens

    def free(self):
        """Release all pages back to the pool."""
        if self.pages:
            self.cache.free_pages(self.pages)
            self.pages = []
            self.seq_len = 0

    def kv_indices(self, device: torch.device) -> torch.Tensor:
        """Page indices: all allocated pages."""
        return torch.tensor(self.pages, dtype=torch.int32, device=device) if self.pages else torch.zeros(0, dtype=torch.int32, device=device)

    def kv_indptr(self, device: torch.device) -> torch.Tensor:
        """Page indptr (single sequence): [0, num_allocated_pages]."""
        return torch.tensor([0, len(self.pages)], dtype=torch.int32, device=device)

    def kv_len_arr(self, device: torch.device) -> torch.Tensor:
        """Sequence length array: [seq_len]."""
        return torch.tensor([self.seq_len], dtype=torch.int32, device=device)

    def last_page_len(self) -> int:
        """Number of valid tokens in the last page."""
        if self.seq_len == 0:
            return 0
        rem = self.seq_len % self.cache.page_size
        return rem if rem > 0 else self.cache.page_size

    def last_page_len_tensor(self, device: torch.device) -> torch.Tensor:
        """Last page length as tensor (for decode)."""
        return torch.tensor([self.last_page_len()], dtype=torch.int32, device=device)

    def block_tables(self, device: torch.device, pad_to_multiple: int = 8) -> torch.Tensor:
        """Block (page) indices for TRTLLM decode kernel.

        Returns [1, padded_num_blocks] int32.
        """
        num_blocks = len(self.pages)
        constraint = TRTLLM_BLOCK_CONSTRAINT // self.cache.page_size
        padded = math.ceil(num_blocks / constraint) * constraint if num_blocks > 0 else constraint
        table = torch.full((1, padded), -1, dtype=torch.int32, device=device)
        if num_blocks > 0:
            table[0, :num_blocks] = torch.tensor(self.pages, dtype=torch.int32, device=device)
        return table

    def store_kv_combined(
        self,
        layer_offset: int,
        kv_combined: torch.Tensor,
        positions: torch.Tensor,
    ):
        """Store combined KV [M, kv_cache_dim] into the paged cache (TRTLLM MLA)."""
        assert self.cache.combined, "store_kv_combined requires combined cache"
        page_size = self.cache.page_size
        pages_tensor = torch.tensor(self.pages, dtype=torch.long, device=kv_combined.device)

        page_indices = pages_tensor[positions.long() // page_size]
        slots = (positions.long() % page_size)

        self.cache.kv_cache[layer_offset, page_indices, slots] = kv_combined.to(self.cache.kv_dtype)
