"""Transformer layer: attention + MoE/dense MLP.

Each layer performs:
  1. fused_add_rmsnorm (residual + input norm)
  2. MLA attention
  3. fused_add_rmsnorm (residual + post-attention norm)
  4. MLP:
     - Dense (layer 0): gate_up → SiLU → down (INT8)
     - MoE (layers 1-60): gate → routing → shared expert (GPU) + routed experts (CPU)

For MoE layers, shared expert and routed experts overlap on GPU and CPU respectively.
"""

import logging
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import flashinfer

from krasis.config import ModelConfig
from krasis.kv_cache import PagedKVCache, SequenceKVState
from krasis.weight_loader import int8_linear

logger = logging.getLogger(__name__)

# Gate per-MoE-layer diagnostics behind env var to avoid GPU sync overhead
_DIAG_ENABLED = os.environ.get("KRASIS_DIAG", "") == "1"

# Runtime-togglable timing (check env var dynamically — set/unset without restart)
def _decode_timing():
    return os.environ.get("KRASIS_DECODE_TIMING", "") == "1"

def _prefill_timing():
    return os.environ.get("KRASIS_PREFILL_TIMING", "") == "1"


def _linear(x: torch.Tensor, weight_data) -> torch.Tensor:
    """Dispatch to INT8 or BF16 linear based on weight type.

    INT8 weights are stored as (weight_int8, scale) tuples.
    BF16 weights are stored as plain tensors.
    """
    if isinstance(weight_data, tuple):
        return int8_linear(x, *weight_data)
    return torch.nn.functional.linear(x, weight_data)


class TransformerLayer:
    """One transformer layer: attention + MLP (dense or MoE)."""

    def __init__(
        self,
        cfg: ModelConfig,
        layer_idx: int,
        weights: Dict,
        device: torch.device,
        krasis_engine=None,
        gpu_prefill_manager=None,
        gpu_prefill_threshold: int = 300,
        attention_backend: str = "flashinfer",
    ):
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.device = device
        self.is_moe = weights["is_moe"]
        self.layer_type = weights.get("layer_type", "full_attention")

        # Layer norms (BF16)
        self.input_norm_weight = weights["norms"]["input_layernorm"]
        self.post_attn_norm_weight = weights["norms"]["post_attention_layernorm"]

        # Attention — select backend based on layer type and attention type
        if self.layer_type == "linear_attention":
            from krasis.linear_attention import GatedDeltaNetAttention
            self.attention = GatedDeltaNetAttention(cfg, layer_idx, weights["linear_attention"], device)
        elif cfg.is_gqa:
            from krasis.attention import GQAAttention
            self.attention = GQAAttention(cfg, layer_idx, weights["attention"], device)
        elif attention_backend == "trtllm":
            from krasis.trtllm_attention import TRTLLMMLAAttention
            self.attention = TRTLLMMLAAttention(cfg, layer_idx, weights["attention"], device)
        else:
            from krasis.attention import MLAAttention
            self.attention = MLAAttention(cfg, layer_idx, weights["attention"], device)

        # MLP weights
        if self.is_moe:
            self.gate_weight = weights["gate"]["weight"]  # [n_experts, hidden]
            self.e_score_correction_bias = weights["gate"].get("e_score_correction_bias")
            self.shared_expert = weights.get("shared_expert")  # {gate_proj, up_proj, down_proj}
            # Fuse gate_proj + up_proj into single matmul to save kernel launch
            if self.shared_expert is not None:
                gp = self.shared_expert["gate_proj"]
                up = self.shared_expert["up_proj"]
                if isinstance(gp, tuple):
                    # INT8: (weight_int8, scale) — concat both along dim 0
                    self.shared_expert["gate_up_proj"] = (
                        torch.cat([gp[0], up[0]], dim=0),
                        torch.cat([gp[1], up[1]], dim=0),
                    )
                else:
                    # BF16: single tensor
                    self.shared_expert["gate_up_proj"] = torch.cat([gp, up], dim=0)
                del self.shared_expert["gate_proj"], self.shared_expert["up_proj"]
            # Shared expert sigmoid gate (Qwen3-Next): [1, hidden_size]
            self.shared_expert_gate = (
                self.shared_expert.get("shared_expert_gate")
                if self.shared_expert is not None else None
            )
            self.dense_mlp = None
        else:
            self.dense_mlp = weights.get("dense_mlp")
            self.gate_weight = None
            self.shared_expert = None

        # Krasis CPU engine for routed experts
        self.krasis_engine = krasis_engine

        # GPU prefill manager for large batches (INT4 Marlin)
        self.gpu_prefill_manager = gpu_prefill_manager
        self.gpu_prefill_threshold = gpu_prefill_threshold

        # CUDA stream for overlapping shared expert with routed expert dispatch
        self._shared_stream = None
        if self.is_moe and self.shared_expert_gate is not None:
            self._shared_stream = torch.cuda.Stream(device=device)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_cache: Optional[PagedKVCache],
        seq_state: Optional[SequenceKVState],
        layer_offset: int,
        moe_layer_idx: Optional[int] = None,
        num_new_tokens: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for one layer.

        Args:
            hidden: [M, hidden_size] BF16
            residual: [M, hidden_size] BF16 or None (first layer)
            positions: [M] int32 position indices
            kv_cache: Paged KV cache (None for linear attention layers)
            seq_state: Sequence KV state (None for linear attention layers)
            layer_offset: Index within this GPU's cache layers (-1 for linear attention)
            moe_layer_idx: MoE layer index for Krasis engine (0-based)
            num_new_tokens: Number of new tokens being processed (for kv_len correction)

        Returns:
            (hidden, residual) where hidden is input to next norm,
            residual is the running residual stream.
        """
        M = hidden.shape[0]
        layer_timing = (_decode_timing() and M == 1) or (_prefill_timing() and M > 1)

        if layer_timing:
            torch.cuda.synchronize()
            t_layer_start = time.perf_counter()

        # ── Pre-attention norm ──
        if residual is None:
            residual = hidden
            hidden = flashinfer.norm.rmsnorm(
                hidden, self.input_norm_weight, self.cfg.rms_norm_eps
            )
        else:
            # fused_add_rmsnorm is IN-PLACE: residual += hidden, then hidden = rmsnorm(residual)
            flashinfer.norm.fused_add_rmsnorm(
                hidden, residual, self.input_norm_weight, self.cfg.rms_norm_eps
            )

        if layer_timing:
            torch.cuda.synchronize()
            t_after_norm1 = time.perf_counter()

        # ── Attention ──
        if self.layer_type == "linear_attention":
            # Linear attention: no KV cache, uses internal recurrent state
            attn_out = self.attention.forward(hidden, is_decode=(M == 1))
        else:
            # Full attention: uses KV cache
            attn_out = self.attention.forward(
                hidden, positions, kv_cache, seq_state, layer_offset,
                num_new_tokens=num_new_tokens,
            )

        if layer_timing:
            torch.cuda.synchronize()
            t_after_attn = time.perf_counter()

        # ── Post-attention norm ──
        # fused_add_rmsnorm is IN-PLACE: residual += attn_out, then attn_out = rmsnorm(residual)
        flashinfer.norm.fused_add_rmsnorm(
            attn_out, residual, self.post_attn_norm_weight, self.cfg.rms_norm_eps
        )
        hidden = attn_out  # now contains normed value

        if layer_timing:
            torch.cuda.synchronize()
            t_after_norm2 = time.perf_counter()

        # ── MLP ──
        if self.is_moe:
            mlp_out = self._moe_forward(hidden, moe_layer_idx)
        else:
            mlp_out = self._dense_mlp_forward(hidden)

        if layer_timing:
            torch.cuda.synchronize()
            t_layer_end = time.perf_counter()
            abs_idx = moe_layer_idx if moe_layer_idx is not None else "dense"
            prefix = "PREFILL-TIMING" if M > 1 else "LAYER-TIMING"
            logger.info(
                "%s L%s(%s) M=%d: norm1=%.2fms attn=%.2fms norm2=%.2fms mlp=%.2fms total=%.2fms",
                prefix, abs_idx, self.layer_type[:3], M,
                (t_after_norm1 - t_layer_start) * 1000,
                (t_after_attn - t_after_norm1) * 1000,
                (t_after_norm2 - t_after_attn) * 1000,
                (t_layer_end - t_after_norm2) * 1000,
                (t_layer_end - t_layer_start) * 1000,
            )

        return mlp_out, residual

    def _dense_mlp_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Dense MLP: gate_proj + up_proj → SiLU(gate) * up → down_proj."""
        gate = _linear(hidden, self.dense_mlp["gate_proj"])
        up = _linear(hidden, self.dense_mlp["up_proj"])
        # SiLU(gate) * up
        activated = flashinfer.activation.silu_and_mul(
            torch.cat([gate, up], dim=-1)
        )
        del gate, up
        return _linear(activated, self.dense_mlp["down_proj"])

    def _shared_expert_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Shared expert MLP on GPU, with optional sigmoid gate (Qwen3-Next)."""
        # Fused gate+up: single matmul → [M, 2N], then silu_and_mul → [M, N]
        gate_up = _linear(hidden, self.shared_expert["gate_up_proj"])
        activated = flashinfer.activation.silu_and_mul(gate_up)
        del gate_up
        output = _linear(activated, self.shared_expert["down_proj"])

        # Apply sigmoid gate if present (Qwen3-Next):
        # sigmoid(hidden @ gate_weight.T) * shared_expert_output
        if self.shared_expert_gate is not None:
            gate_value = torch.sigmoid(
                torch.nn.functional.linear(hidden, self.shared_expert_gate)
            )  # [M, 1]
            output = gate_value * output

        return output

    def _moe_forward(
        self,
        hidden: torch.Tensor,
        moe_layer_idx: Optional[int],
    ) -> torch.Tensor:
        """MoE forward: routing + shared expert (GPU) + routed experts (CPU).

        Kimi K2.5 specifics:
        - Scoring: sigmoid (not softmax)
        - e_score_correction_bias added before top-k selection
        - norm_topk_prob: divide by sum of selected weights
        - routed_scaling_factor: 2.827
        """
        M = hidden.shape[0]
        timing = _decode_timing() and M == 1

        if timing:
            t_start = time.perf_counter()

        # ── Routing ──
        # gate: [M, hidden] @ [n_experts, hidden]^T → [M, n_experts]
        router_logits = torch.matmul(hidden.float(), self.gate_weight.float().t())

        # Sigmoid scoring (Kimi K2.5)
        if self.cfg.scoring_func == "sigmoid":
            scores = torch.sigmoid(router_logits)
        else:
            scores = torch.softmax(router_logits, dim=-1)

        # Add correction bias before top-k selection
        if self.e_score_correction_bias is not None:
            scores_for_selection = scores + self.e_score_correction_bias.float()
        else:
            scores_for_selection = scores

        # Top-k selection
        topk = self.cfg.num_experts_per_tok
        topk_weights, topk_ids = torch.topk(scores_for_selection, topk, dim=-1)

        # Use original scores (without bias) for the selected experts' weights
        topk_weights = scores.gather(1, topk_ids)

        # Normalize weights
        if self.cfg.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        topk_weights = topk_weights.to(torch.float32)
        topk_ids = topk_ids.to(torch.int32)

        # DIAG: per-MoE-layer logging for first N calls (gated behind KRASIS_DIAG=1)
        diag_moe = False
        if _DIAG_ENABLED:
            if not hasattr(self, '_moe_call_count'):
                self._moe_call_count = 0
            self._moe_call_count += 1
            diag_moe = self._moe_call_count <= 3  # first 3 forward passes
            if diag_moe and moe_layer_idx is not None and M == 1:
                h_rms = hidden.float().pow(2).mean().sqrt().item()
                ids_list = topk_ids[0].tolist()
                wts_list = [f"{w:.4f}" for w in topk_weights[0].tolist()]
                wt_sum = topk_weights[0].sum().item()
                logger.info(
                    "MOE-DIAG L%d call#%d: in_rms=%.4f ids=%s wts=[%s] wt_sum=%.4f",
                    moe_layer_idx, self._moe_call_count,
                    h_rms, ids_list, ",".join(wts_list), wt_sum,
                )
            # Log unique expert activation for prefill batches (large M)
            if diag_moe and moe_layer_idx is not None and M > 1:
                unique_experts = torch.unique(topk_ids).numel()
                total_experts = self.cfg.n_routed_experts
                # Per-expert activation counts
                flat_ids = topk_ids.reshape(-1)
                counts = torch.bincount(flat_ids.long(), minlength=total_experts)
                zero_experts = (counts == 0).sum().item()
                min_count = counts[counts > 0].min().item() if zero_experts < total_experts else 0
                max_count = counts.max().item()
                median_count = counts[counts > 0].float().median().item() if zero_experts < total_experts else 0
                logger.info(
                    "MOE-ACTIVATION L%d M=%d: %d/%d experts activated (%.1f%%), "
                    "%d inactive, counts: min=%d median=%.0f max=%d",
                    moe_layer_idx, M, unique_experts, total_experts,
                    100.0 * unique_experts / total_experts,
                    zero_experts, min_count, median_count, max_count,
                )

        if timing:
            t_routing = time.perf_counter()

        # ── Overlap shared expert with routed expert dispatch for M=1 ──
        # Shared expert only reads `hidden` (no dependency on routing results),
        # so we launch it on a separate CUDA stream before dispatch.
        shared_future = None
        if M == 1 and self._shared_stream is not None and self.shared_expert is not None:
            # Make shared stream wait for hidden to be ready on default stream
            self._shared_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self._shared_stream):
                shared_future = self._shared_expert_forward(hidden)

        # ── Dispatch: GPU path (prefill or decode) vs CPU decode ──
        # GPU path for: (1) prefill above threshold, or (2) M=1 decode (gpu_decode)
        if self.gpu_prefill_manager is not None and (
            M >= self.gpu_prefill_threshold or M == 1
        ):
            # GPU path: INT4 Marlin kernel handles routing + expert computation
            output = self._gpu_prefill_forward(hidden, topk_ids, topk_weights, moe_layer_idx)
        else:
            # CPU path: Krasis engine handles routed experts
            output = self._routed_expert_forward(hidden, topk_ids, topk_weights, moe_layer_idx)

        if timing:
            t_dispatch = time.perf_counter()

        # If shared_expert_gate exists, Rust engine skips shared expert —
        # we handle it on GPU with the sigmoid gate
        if shared_future is not None:
            # Sync shared stream and add result
            torch.cuda.current_stream(self.device).wait_stream(self._shared_stream)
            output = output + shared_future
        elif self.shared_expert_gate is not None and self.shared_expert is not None:
            output = output + self._shared_expert_forward(hidden)

        if timing:
            t_shared = time.perf_counter()
            logger.info(
                "DECODE-TIMING L%s: routing=%.1fms dispatch=%.1fms shared=%.1fms total=%.1fms",
                moe_layer_idx,
                (t_routing - t_start) * 1000,
                (t_dispatch - t_routing) * 1000,
                (t_shared - t_dispatch) * 1000,
                (t_shared - t_start) * 1000,
            )

        if _DIAG_ENABLED and diag_moe and moe_layer_idx is not None and M == 1:
            o_rms = output.float().pow(2).mean().sqrt().item()
            logger.info(
                "MOE-DIAG L%d call#%d: out_rms=%.4f",
                moe_layer_idx, self._moe_call_count, o_rms,
            )

        return output

    def _gpu_prefill_forward(
        self,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        moe_layer_idx: Optional[int],
    ) -> torch.Tensor:
        """GPU MoE forward using INT4 Marlin kernel (for prefill with large M).

        GpuPrefillManager handles: routing, shared expert, routed_scaling_factor.
        """
        if moe_layer_idx is None:
            raise RuntimeError("moe_layer_idx required for GPU prefill")

        return self.gpu_prefill_manager.forward(
            moe_layer_idx, hidden, topk_ids, topk_weights,
        )

    def _routed_expert_forward(
        self,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        moe_layer_idx: Optional[int],
    ) -> torch.Tensor:
        """Dispatch routed experts to Krasis CPU engine + shared expert on GPU.

        The Rust engine computes:
            routed_scaling_factor * Σ(w_i * expert_i(x)) + shared_expert(x)
        So the Python side just returns the Rust output directly.
        """
        if self.krasis_engine is None:
            raise RuntimeError("Krasis engine not set for MoE layer")

        M = hidden.shape[0]
        timing = _decode_timing() and M == 1

        if timing:
            t0 = time.perf_counter()

        # GPU → CPU transfer (implicit sync via .cpu())
        act_cpu = hidden.detach().cpu().contiguous()
        ids_cpu = topk_ids.detach().cpu().contiguous()
        wts_cpu = topk_weights.detach().cpu().contiguous()

        if timing:
            t1 = time.perf_counter()

        # Convert to bytes for Rust engine
        act_bytes = act_cpu.view(torch.uint16).numpy().view(np.uint8).tobytes()
        ids_bytes = ids_cpu.numpy().view(np.uint8).tobytes()
        wts_bytes = wts_cpu.numpy().view(np.uint8).tobytes()

        if timing:
            t2 = time.perf_counter()

        # Submit async
        self.krasis_engine.submit_forward(
            moe_layer_idx, act_bytes, ids_bytes, wts_bytes, M
        )

        # Sync — blocks until CPU experts finish
        output_bytes = self.krasis_engine.sync_forward()

        if timing:
            t3 = time.perf_counter()

        # Convert BF16 output bytes → tensor
        # Rust engine already applied: routed_scaling_factor * routed + shared_expert
        output = torch.frombuffer(
            bytearray(output_bytes), dtype=torch.bfloat16
        ).reshape(M, self.cfg.hidden_size)

        # CPU → GPU
        output = output.to(self.device)

        if timing:
            t4 = time.perf_counter()
            logger.info(
                "  CPU-EXPERT L%s: gpu→cpu=%.1fms bytes=%.1fms rust=%.1fms cpu→gpu=%.1fms total=%.1fms",
                moe_layer_idx,
                (t1 - t0) * 1000,
                (t2 - t1) * 1000,
                (t3 - t2) * 1000,
                (t4 - t3) * 1000,
                (t4 - t0) * 1000,
            )

        return output
