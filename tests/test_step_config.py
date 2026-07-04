import json
import os
import unittest
from types import SimpleNamespace

from krasis.config import ModelConfig, QuantConfig
from krasis.model import KrasisModel
from krasis.vram_budget import (
    _hqq_attention_tensor_shapes_for_layer,
    _read_model_config,
)


STEP_MODEL_PATH = "/home/main/.krasis/models/Step-3.7-Flash"
GEMMA4_MODEL_PATH = "/home/main/.krasis/models/gemma-4-26b-a4b-it"


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

    def test_step37_vision_support_and_processor_placeholder_expansion(self):
        from PIL import Image
        from transformers import AutoTokenizer

        model = KrasisModel.__new__(KrasisModel)
        model.cfg = SimpleNamespace(model_path=STEP_MODEL_PATH)
        model._step_vision_modules = None

        self.assertTrue(model.supports_step_image_inputs())
        self.assertTrue(model.supports_image_inputs())

        _, _, processor_mod = model._ensure_step_vision_modules()
        tokenizer = AutoTokenizer.from_pretrained(STEP_MODEL_PATH, trust_remote_code=True)
        template_path = os.path.join(STEP_MODEL_PATH, "chat_template.jinja")
        if not getattr(tokenizer, "chat_template", None):
            with open(template_path, encoding="utf-8") as f:
                tokenizer.chat_template = f.read()
        processor = processor_mod.Step3VLProcessor(tokenizer=tokenizer)

        rendered = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {"type": "image", "image": "placeholder"},
                    ],
                }
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        self.assertEqual(rendered.count("<im_patch>"), 1)

        image = Image.new("RGB", (728, 728), (32, 64, 96))
        batch = processor(text=[rendered], images=[image], return_tensors="pt")

        self.assertEqual(tuple(batch["pixel_values"].shape), (1, 3, 728, 728))
        self.assertEqual(batch["num_patches"].tolist(), [0])
        self.assertEqual(int((batch["input_ids"] == processor.image_token_id).sum().item()), 169)

    def test_step37_vision_int4_config_validation(self):
        self.assertEqual(QuantConfig().step_vision_quant, "int4")
        self.assertEqual(QuantConfig(step_vision_quant="int4").step_vision_quant, "int4")
        self.assertEqual(QuantConfig(step_vision_group_size="64").step_vision_group_size, 64)
        with self.assertRaises(ValueError):
            QuantConfig(step_vision_quant="int8")
        with self.assertRaises(ValueError):
            QuantConfig(step_vision_quant="int4", step_vision_group_size=96)

    def test_step37_vision_int4_linear_and_conv_shapes(self):
        import torch

        from krasis.step_vision_int4 import StepVisionInt4Conv2d, StepVisionInt4Linear

        torch.manual_seed(7)
        linear = torch.nn.Linear(37, 19, bias=True).to(torch.bfloat16)
        q_linear = StepVisionInt4Linear.from_linear(linear, group_size=32)
        x = torch.randn(2, 5, 37, dtype=torch.bfloat16)
        y = q_linear(x)
        self.assertEqual(tuple(y.shape), (2, 5, 19))
        self.assertEqual(q_linear.packed_weight.dtype, torch.uint8)
        self.assertLess(q_linear.quantized_weight_bytes(), linear.weight.numel() * linear.weight.element_size())

        conv = torch.nn.Conv2d(3, 11, kernel_size=3, stride=2, padding=1, bias=True).to(torch.bfloat16)
        q_conv = StepVisionInt4Conv2d.from_conv2d(conv, group_size=32)
        pixels = torch.randn(2, 3, 16, 16, dtype=torch.bfloat16)
        out = q_conv(pixels)
        self.assertEqual(tuple(out.shape), (2, 11, 8, 8))
        self.assertEqual(q_conv.packed_weight.dtype, torch.uint8)
        self.assertLess(q_conv.quantized_weight_bytes(), conv.weight.numel() * conv.weight.element_size())


@unittest.skipUnless(os.path.exists(os.path.join(GEMMA4_MODEL_PATH, "config.json")), "Gemma4 not downloaded")
class Gemma4VisionConfigTest(unittest.TestCase):
    def test_gemma4_vision_support_and_dynamic_placeholder_expansion(self):
        from PIL import Image

        from krasis.gemma4_vision import Gemma4ImagePreprocessor

        model = KrasisModel.__new__(KrasisModel)
        model.cfg = SimpleNamespace(model_path=GEMMA4_MODEL_PATH, hidden_size=2816)
        model._qwen_vision_processor = None
        model._qwen_vision_model = None
        model._step_vision_modules = None
        model._step_vision_processor = None
        model._step_vision_model = None
        model._step_vision_projector = None
        model._gemma_vision_raw_config = {
            "image_token_id": 258880,
            "boi_token_id": 255999,
            "eoi_token_id": 258882,
        }

        self.assertTrue(model.supports_gemma_image_inputs())
        self.assertTrue(model.supports_image_inputs())

        processor = Gemma4ImagePreprocessor(
            patch_size=16,
            pooling_kernel_size=3,
            max_soft_tokens=280,
            rescale_factor=1.0 / 255.0,
        )
        image = Image.new("RGB", (728, 728), (32, 64, 96))
        batch = processor([image])
        self.assertEqual(tuple(batch["pixel_values"].shape), (1, 2520, 768))
        self.assertEqual(tuple(batch["image_position_ids"].shape), (1, 2520, 2))
        self.assertEqual(batch["num_soft_tokens_per_image"], [256])

        expanded, block_ids = model._gemma_expand_image_token_ids(
            [2, 258880, 106],
            batch["num_soft_tokens_per_image"],
        )
        self.assertEqual(expanded[0], 2)
        self.assertEqual(expanded[1], 255999)
        self.assertEqual(expanded[-2], 258882)
        self.assertEqual(expanded[-1], 106)
        self.assertEqual(expanded.count(258880), 256)
        self.assertEqual(block_ids.count(0), 256)
        self.assertEqual(block_ids[0], -1)
        self.assertEqual(block_ids[1], -1)
        self.assertEqual(block_ids[-2], -1)

    def test_gemma4_patch_embedder_runs_with_int4_input_projection(self):
        import torch

        from krasis.gemma4_vision import Gemma4VisionConfig, Gemma4VisionPatchEmbedder
        from krasis.step_vision_int4 import StepVisionInt4Linear

        cfg = Gemma4VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            patch_size=4,
            position_embedding_size=8,
        )
        embedder = Gemma4VisionPatchEmbedder(cfg).to(dtype=torch.bfloat16)
        embedder.input_proj = StepVisionInt4Linear.from_linear(embedder.input_proj, group_size=32)

        pixel_values = torch.rand(1, 6, 3 * cfg.patch_size * cfg.patch_size, dtype=torch.float32)
        pixel_position_ids = torch.zeros(1, 6, 2, dtype=torch.long)
        padding_positions = torch.zeros(1, 6, dtype=torch.bool)

        hidden = embedder(pixel_values, pixel_position_ids, padding_positions)
        self.assertEqual(tuple(hidden.shape), (1, 6, 32))
        self.assertEqual(hidden.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
