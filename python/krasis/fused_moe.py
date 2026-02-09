"""Thin SGLang FusedMoE replacement that delegates to the Krasis Rust backend.

SGLang calls this as a drop-in replacement for its FusedMoE layer.
This module translates PyTorch tensors to/from the Rust .so via PyO3.

Usage:
    from krasis.fused_moe import KrasisFusedMoE

    moe = KrasisFusedMoE("/path/to/model")
    output = moe.forward(hidden_states_bf16, expert_indices, expert_weights, layer_idx=0)
"""

import struct
from typing import Optional

try:
    import torch
except ImportError:
    torch = None

from krasis import KrasisEngine


class KrasisFusedMoE:
    """SGLang-compatible FusedMoE layer backed by Krasis Rust CPU engine."""

    def __init__(
        self,
        model_dir: str,
        group_size: int = 128,
        parallel: bool = True,
    ):
        self.engine = KrasisEngine(parallel=parallel)
        self.engine.load(model_dir, group_size=group_size)
        self._hidden_size = self.engine.hidden_size()

    def forward(
        self,
        hidden_states: "torch.Tensor",
        expert_indices: "torch.Tensor",
        expert_weights: "torch.Tensor",
        moe_layer_idx: int = 0,
    ) -> "torch.Tensor":
        """Run MoE forward for one or more tokens.

        Args:
            hidden_states: [num_tokens, hidden_size] BF16 tensor (CPU or GPU)
            expert_indices: [num_tokens, top_k] int64 tensor
            expert_weights: [num_tokens, top_k] float32 tensor
            moe_layer_idx: 0-based MoE layer index

        Returns:
            [num_tokens, hidden_size] BF16 tensor on same device as input
        """
        device = hidden_states.device
        num_tokens = hidden_states.shape[0]

        # Move to CPU if on GPU
        h_cpu = hidden_states.detach().cpu().contiguous()
        idx_cpu = expert_indices.detach().cpu()
        wt_cpu = expert_weights.detach().cpu().float()

        # Process each token
        outputs = []
        for t in range(num_tokens):
            # BF16 activation as bytes
            act_bytes = h_cpu[t].numpy().view("uint8").tobytes()

            # Expert indices and weights as Python lists
            indices = idx_cpu[t].tolist()
            weights = wt_cpu[t].tolist()

            # Call Rust engine
            out_bytes = self.engine.moe_forward(
                moe_layer_idx, act_bytes, indices, weights
            )

            # Convert f32 bytes back to tensor
            out_tensor = torch.frombuffer(bytearray(out_bytes), dtype=torch.float32)
            outputs.append(out_tensor)

        # Stack and convert to BF16
        result = torch.stack(outputs).to(torch.bfloat16)

        # Move back to original device
        if device.type != "cpu":
            result = result.to(device)

        return result

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_moe_layers(self) -> int:
        return self.engine.num_moe_layers()

    @property
    def num_experts(self) -> int:
        return self.engine.num_experts()

    @property
    def top_k(self) -> int:
        return self.engine.top_k()
