#!/usr/bin/env python3
"""No-GPU model configuration contract tests."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from krasis.config import ModelConfig
from krasis.kv_cache import MLA_CKV_KERNEL_MIN_DIM
from krasis.layer import NativeMLAWeights, TransformerLayer


def _write_config(test_case: unittest.TestCase, config: dict) -> str:
    root = tempfile.mkdtemp(prefix="krasis-model-config-")
    test_case.addCleanup(shutil.rmtree, root)
    Path(root, "config.json").write_text(json.dumps(config), encoding="utf-8")
    return root


def _glm_dsa_config() -> dict:
    return {
        "model_type": "glm_moe_dsa",
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 8,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "vocab_size": 154880,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "first_k_dense_replace": 3,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "index_topk": 2048,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
        "indexer_types": [
            "full",
            "full",
            "full",
            "shared",
            "shared",
            "shared",
            "full",
            "shared",
        ],
        "indexer_rope_interleave": True,
        "index_share_for_mtp_iteration": True,
    }


class ModelConfigContractTests(unittest.TestCase):
    def test_glm_moe_dsa_indexshare_contract(self) -> None:
        cfg = ModelConfig.from_model_path(_write_config(self, _glm_dsa_config()))

        self.assertTrue(cfg.is_mla)
        self.assertTrue(cfg.is_dsa)
        self.assertEqual(cfg.index_topk, 2048)
        self.assertEqual(cfg.index_head_dim, 128)
        self.assertEqual(cfg.index_n_heads, 32)
        self.assertEqual(cfg.index_topk_freq, 4)
        self.assertEqual(cfg.index_skip_topk_offset, 3)
        self.assertEqual(
            cfg.indexer_types,
            [
                "full",
                "full",
                "full",
                "shared",
                "shared",
                "shared",
                "full",
                "shared",
            ],
        )
        self.assertTrue(cfg.indexer_rope_interleave)
        self.assertTrue(cfg.index_share_for_mtp_iteration)
        self.assertEqual(cfg.num_moe_layers, 5)

    def test_glm_moe_dsa_requires_complete_indexer_schedule(self) -> None:
        raw = _glm_dsa_config()
        raw["indexer_types"] = raw["indexer_types"][:-1]
        with self.assertRaisesRegex(
            ValueError,
            r"indexer_types length 7 != num_hidden_layers 8",
        ):
            ModelConfig.from_model_path(_write_config(self, raw))

    def test_glm_moe_dsa_rejects_unowned_shared_indexer(self) -> None:
        raw = _glm_dsa_config()
        raw["indexer_types"][0] = "shared"
        with self.assertRaisesRegex(
            ValueError,
            r"shared indexer at layer 0 has no preceding full indexer",
        ):
            ModelConfig.from_model_path(_write_config(self, raw))

    def test_non_dsa_defaults_remain_disabled(self) -> None:
        raw = _glm_dsa_config()
        raw["model_type"] = "deepseek_v3"
        for key in (
            "index_topk",
            "index_head_dim",
            "index_n_heads",
            "index_topk_freq",
            "index_skip_topk_offset",
            "indexer_types",
            "indexer_rope_interleave",
            "index_share_for_mtp_iteration",
        ):
            raw.pop(key)
        cfg = ModelConfig.from_model_path(_write_config(self, raw))
        self.assertFalse(cfg.is_dsa)
        self.assertEqual(cfg.index_topk, 0)
        self.assertIsNone(cfg.indexer_types)

    def test_native_mla_setup_contract_pads_only_the_kernel_dimension(self) -> None:
        raw = _glm_dsa_config()
        raw.update(
            {
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 4,
                "num_attention_heads": 2,
                "q_lora_rank": 4,
                "kv_lora_rank": 4,
                "qk_nope_head_dim": 2,
                "qk_rope_head_dim": 2,
                "v_head_dim": 2,
            }
        )
        cfg = ModelConfig.from_model_path(_write_config(self, raw))
        weights = {
            "q_a_proj": torch.zeros((4, 8), dtype=torch.bfloat16),
            "q_b_proj": torch.zeros((8, 4), dtype=torch.bfloat16),
            "q_a_layernorm": torch.ones((4,), dtype=torch.bfloat16),
            "kv_a_proj_with_mqa": torch.zeros((6, 8), dtype=torch.bfloat16),
            "o_proj": torch.zeros((8, 4), dtype=torch.bfloat16),
            "kv_a_layernorm": torch.ones((4,), dtype=torch.bfloat16),
            "w_kc": torch.ones((2, 2, 4), dtype=torch.bfloat16),
            "w_vc": torch.ones((2, 2, 4), dtype=torch.bfloat16),
        }

        layer = TransformerLayer(
            cfg,
            0,
            {
                "norms": {
                    "input_layernorm": torch.ones((8,), dtype=torch.bfloat16),
                    "post_attention_layernorm": torch.ones(
                        (8,), dtype=torch.bfloat16
                    ),
                },
                "is_moe": False,
                "layer_type": "full_attention",
                "attention": weights,
            },
            torch.device("cpu"),
        )
        attention = layer.attention

        self.assertIsInstance(attention, NativeMLAWeights)
        self.assertEqual(attention.ckv_dim, MLA_CKV_KERNEL_MIN_DIM)
        self.assertEqual(attention.w_kc.shape, (2, 2, MLA_CKV_KERNEL_MIN_DIM))
        self.assertEqual(attention.w_vc.shape, (2, 2, MLA_CKV_KERNEL_MIN_DIM))
        self.assertTrue(torch.all(attention.w_kc[..., :4] == 1))
        self.assertTrue(torch.all(attention.w_kc[..., 4:] == 0))
        with self.assertRaisesRegex(
            RuntimeError,
            r"native Rust/CUDA runtime",
        ):
            attention.forward(torch.zeros((1, 8), dtype=torch.bfloat16))


if __name__ == "__main__":
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit("Run through ./dev model-config-test")
    unittest.main()
