#!/usr/bin/env python3
"""No-GPU contracts for DeepSeek-V4-Flash-Vision preprocessing and modules."""

import os
import unittest
from types import SimpleNamespace

import torch
from PIL import Image

from krasis.deepseek_v4_vision import (
    DeepseekV4ImagePreprocessor,
    DeepseekV4VisionAligner,
    DeepseekV4VisionModel,
    IMAGE,
    IMAGE_END,
    IMAGE_START,
)


def _config():
    return SimpleNamespace(
        hidden_size=24,
        vision_n_layers=1,
        vision_dim=16,
        vision_n_heads=4,
        vision_inter_dim=12,
        vision_patch_size=2,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=2,
        vision_max_n_token=32,
        vision_min_pixels=64,
        vision_max_wh_ratio=8.0,
    )


class DeepseekV4VisionContractTests(unittest.TestCase):
    def test_checkpoint_module_names_and_output_geometry(self):
        cfg = _config()
        vision = DeepseekV4VisionModel(cfg).to(dtype=torch.bfloat16)
        aligner = DeepseekV4VisionAligner(cfg).to(dtype=torch.bfloat16)
        state = vision.state_dict()
        self.assertIn("patch_embed.proj.weight", state)
        self.assertIn("blocks.0.attn.wqkv.weight", state)
        self.assertIn("blocks.0.mlp.w1.weight", state)
        self.assertIn("norm.weight", state)
        patches = torch.zeros(16, 3, 2, 2, dtype=torch.bfloat16)
        with torch.inference_mode():
            features = vision(patches, 4, 4)
            aligned = aligner(features, 4, 4)
        self.assertEqual(tuple(features.shape), (16, 16))
        self.assertEqual(tuple(aligned.shape), (4, 24))

    def test_n_layout_stays_within_checkpoint_token_budget(self):
        cfg = _config()
        processor = DeepseekV4ImagePreprocessor(cfg)
        image = Image.new("RGB", (9, 7), color=(13, 89, 233))
        prepared = processor.prepare(image, start_pos=5)
        self.assertLessEqual(int(prepared.types.numel()), cfg.vision_max_n_token)
        self.assertEqual(int((prepared.types == IMAGE_START).sum()), 1)
        self.assertEqual(int((prepared.types == IMAGE_END).sum()), 1)
        self.assertEqual(
            int((prepared.types == IMAGE).sum()), int(prepared.perm.numel())
        )
        self.assertEqual(int(prepared.types[-1]), IMAGE_END)


if __name__ == "__main__":
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit("Run through ./dev model-config-test")
    unittest.main()
