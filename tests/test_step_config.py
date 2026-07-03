import json
import os
import unittest

from krasis.config import ModelConfig
from krasis.vram_budget import (
    _hqq_attention_tensor_shapes_for_layer,
    _read_model_config,
)


STEP_MODEL_PATH = "/home/main/.krasis/models/Step-3.7-Flash"


def _safetensor_shape_dtype(tensor_name):
    from safetensors import safe_open

    index_path = os.path.join(STEP_MODEL_PATH, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    shard = weight_map[tensor_name]
    shard_path = os.path.join(STEP_MODEL_PATH, shard)
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(tensor_name)
        return list(tensor_slice.get_shape()), str(tensor_slice.get_dtype())


@unittest.skipUnless(os.path.exists(os.path.join(STEP_MODEL_PATH, "config.json")), "Step-3.7-Flash not downloaded")
class StepConfigTest(unittest.TestCase):
    def test_step37_flash_text_config_normalization(self):
        cfg = ModelConfig.from_model_path(STEP_MODEL_PATH)

        self.assertEqual(cfg.model_type, "step3p5")
        self.assertTrue(cfg.step3_text)
        self.assertEqual(cfg.layers_prefix, "model")
        self.assertEqual(cfg.num_hidden_layers, 45)
        self.assertEqual(len(cfg.layer_types), 45)

        self.assertEqual(cfg.n_routed_experts, 288)
        self.assertEqual(cfg.num_experts_per_tok, 8)
        self.assertEqual(cfg.moe_intermediate_size, 1280)
        self.assertEqual(cfg.n_shared_experts, 1)
        self.assertEqual(cfg.shared_expert_intermediate_size, 1280)
        self.assertEqual(cfg.effective_shared_expert_intermediate, 1280)
        self.assertEqual(cfg.num_moe_layers, 42)
        self.assertFalse(cfg.is_moe_layer(0))
        self.assertFalse(cfg.is_moe_layer(1))
        self.assertFalse(cfg.is_moe_layer(2))
        self.assertTrue(cfg.is_moe_layer(3))
        self.assertTrue(cfg.is_moe_layer(44))

        self.assertEqual(cfg.scoring_func, "sigmoid")
        self.assertEqual(cfg.routed_scaling_factor, 3.0)
        self.assertTrue(cfg.norm_topk_prob)
        self.assertTrue(cfg.use_moe_router_bias)
        self.assertTrue(cfg.need_fp32_gate)
        self.assertTrue(cfg.head_wise_attention_gate)
        self.assertTrue(cfg.norm_bias_one)
        self.assertEqual(cfg.rms_norm_eps, 1e-5)
        self.assertEqual(cfg.eos_token_id, 1)
        self.assertEqual(cfg.extra_stop_token_ids, (2, 128007))
        self.assertEqual(cfg.yarn_only_types, ["full_attention"])

        self.assertEqual(cfg.gqa_num_heads_for_layer(0), 64)
        self.assertEqual(cfg.gqa_num_heads_for_layer(1), 96)
        self.assertEqual(cfg.gqa_num_kv_heads_for_layer(0), 8)
        self.assertEqual(cfg.gqa_num_kv_heads_for_layer(1), 8)
        self.assertEqual(cfg.gqa_head_dim_for_layer(0), 128)
        self.assertEqual(cfg.gqa_head_dim_for_layer(1), 128)

        self.assertEqual(cfg.rope_theta_for_layer(0), 5_000_000.0)
        self.assertEqual(cfg.rope_theta_for_layer(1), 10_000.0)
        self.assertEqual(cfg.rotary_dim_for_layer(0), 64)
        self.assertEqual(cfg.rotary_dim_for_layer(1), 128)

        self.assertEqual(cfg.swiglu_limit_for_layer(42), 0.0)
        self.assertEqual(cfg.swiglu_limit_for_layer(43), 7.0)
        self.assertEqual(cfg.shared_swiglu_limit_for_layer(43), 16.0)

    def test_step37_flash_weight_header_shapes(self):
        expected = {
            "model.layers.0.self_attn.q_proj.weight": [8192, 4096],
            "model.layers.0.self_attn.k_proj.weight": [1024, 4096],
            "model.layers.0.self_attn.v_proj.weight": [1024, 4096],
            "model.layers.0.self_attn.o_proj.weight": [4096, 8192],
            "model.layers.0.self_attn.g_proj.weight": [64, 4096],
            "model.layers.0.self_attn.q_norm.weight": [128],
            "model.layers.1.self_attn.q_proj.weight": [12288, 4096],
            "model.layers.1.self_attn.o_proj.weight": [4096, 12288],
            "model.layers.1.self_attn.g_proj.weight": [96, 4096],
            "model.layers.3.moe.gate_proj.weight": [288, 1280, 4096],
            "model.layers.3.share_expert.gate_proj.weight": [1280, 4096],
        }

        for tensor_name, shape in expected.items():
            with self.subTest(tensor_name=tensor_name):
                actual_shape, dtype = _safetensor_shape_dtype(tensor_name)
                self.assertEqual(actual_shape, shape)
                self.assertEqual(dtype, "BF16")

    def test_step37_hqq_budget_shapes_use_per_layer_gqa_geometry(self):
        cfg = _read_model_config(STEP_MODEL_PATH)

        full_shapes = dict(
            (name, (rows, cols))
            for name, rows, cols in _hqq_attention_tensor_shapes_for_layer(cfg, "full_attention", 0)
        )
        sliding_shapes = dict(
            (name, (rows, cols))
            for name, rows, cols in _hqq_attention_tensor_shapes_for_layer(cfg, "sliding_attention", 1)
        )

        self.assertEqual(full_shapes["q_proj"], (8192, 4096))
        self.assertEqual(full_shapes["k_proj"], (1024, 4096))
        self.assertEqual(full_shapes["v_proj"], (1024, 4096))
        self.assertEqual(full_shapes["o_proj"], (4096, 8192))
        self.assertEqual(full_shapes["fused_qkv"], (10240, 4096))

        self.assertEqual(sliding_shapes["q_proj"], (12288, 4096))
        self.assertEqual(sliding_shapes["k_proj"], (1024, 4096))
        self.assertEqual(sliding_shapes["v_proj"], (1024, 4096))
        self.assertEqual(sliding_shapes["o_proj"], (4096, 12288))
        self.assertEqual(sliding_shapes["fused_qkv"], (14336, 4096))


if __name__ == "__main__":
    unittest.main()
