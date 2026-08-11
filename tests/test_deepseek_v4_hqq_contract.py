#!/usr/bin/env python3
"""No-GPU contract test for DeepSeek-V4 phase-one HQQ tensor selection."""

import os
from types import SimpleNamespace

import torch

from krasis.model import KrasisModel


def make_tensor(rows: int, cols: int, start: int) -> torch.Tensor:
    values = torch.arange(start, start + rows * cols, dtype=torch.float32)
    return values.reshape(rows, cols) / 97.0


def main() -> None:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit("Run via ./dev model-config-test")

    model = KrasisModel.__new__(KrasisModel)
    model.cfg = SimpleNamespace(is_deepseek_v4=True, is_mla=False)
    weights = {
        "wq_a": make_tensor(8, 16, 0),
        "wq_b": make_tensor(16, 8, 1000),
        "wkv": make_tensor(8, 16, 2000),
        "wo_a": make_tensor(8, 8, 3000),
        "wo_b": make_tensor(16, 8, 4000),
        "compressor": {
            "wkv": make_tensor(8, 16, 5000),
            "wgate": make_tensor(1, 16, 6000),
        },
        "indexer": {
            "wq_b": make_tensor(8, 8, 7000),
            "weights_proj": make_tensor(1, 8, 8000),
        },
    }
    tensor_map = model._hqq_attention_tensor_map("deepseek_v4", weights)
    expected = ["wq_a", "wq_b", "wkv", "wo_b"]
    assert list(tensor_map) == expected, (
        f"DeepSeek-V4 phase-one HQQ tensor map mismatch: "
        f"got {list(tensor_map)}, expected {expected}"
    )
    print("PASS: DeepSeek-V4 phase-one HQQ tensor map is exact")


if __name__ == "__main__":
    main()
