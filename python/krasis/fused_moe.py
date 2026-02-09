"""Thin SGLang FusedMoE replacement that delegates to the Krasis Rust backend.

SGLang calls this as a drop-in replacement for its FusedMoE layer.
This module translates PyTorch tensors to/from the Rust .so via PyO3.
"""

# TODO: import krasis native module
# TODO: implement FusedMoE-compatible forward() that:
#   1. Receives hidden_states, router_logits from SGLang
#   2. Calls MoERunner.forward() in the Rust backend
#   3. Returns combined expert output as PyTorch tensor
