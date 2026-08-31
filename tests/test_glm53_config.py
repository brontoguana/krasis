import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from krasis.config import ModelConfig
from krasis.model import (
    KrasisModel,
    _dsa_owner_layers_for_segment,
    _dsa_resource_layers_for_segment,
)
from krasis.glm53_vision import Glm53ImagePreprocessor, Glm53VisionConfig, Glm53VisionModel


def _glm53_config() -> dict:
    dsa_layers = list(range(3, 45, 4))
    kda_layers = [layer for layer in range(45) if layer not in dsa_layers]
    return {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
        "text_config": {
            "model_type": "glm5_next_text",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 45,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "vocab_size": 154880,
            "first_k_dense_replace": 3,
            "n_routed_experts": 288,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 256,
            "qk_rope_head_dim": 0,
            "v_head_dim": 256,
            "index_topk": 2048,
            "index_head_dim": 128,
            "index_n_heads": 32,
            "indexer_types": ["full"] * 45,
            "indexer_rope_interleave": True,
            "index_share_for_mtp_iteration": True,
            "index_kpool": 4,
            "index_kpool_compress": True,
            "index_kpool_always_select_tail": True,
            "mhc": True,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "linear_attn_config": {
                "num_heads": 64,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "kda_layers": kda_layers,
                "full_attn_layers": dsa_layers,
            },
            "layer_types": [
                "deepseek_sparse_attention" if layer in dsa_layers
                else "linear_attention"
                for layer in range(45)
            ],
            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
            "routed_scaling_factor": 2.5,
            "swiglu_limit": 10.0,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 1048576,
            "tie_word_embeddings": False,
        },
        "vision_config": {
            "depth": 24,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_heads": 16,
            "out_hidden_size": 4096,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "projection_intermediate_size": 10240,
            "rms_norm_eps": 1e-5,
            "swiglu_limit": 10.0,
            "attention_bias": True,
            "hidden_act": "silu",
        },
        "image_token_id": 154854,
    }


class Glm53ConfigTests(unittest.TestCase):
    def _load(self, mutate=None) -> ModelConfig:
        raw = _glm53_config()
        if mutate is not None:
            mutate(raw["text_config"])
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.json").write_text(json.dumps(raw))
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.language_model.layers.0.self_attn.q_proj.weight":
                                "model-00001-of-00001.safetensors",
                            "lm_head.weight": "model-00001-of-00001.safetensors",
                        }
                    }
                )
            )
            return ModelConfig.from_model_path(str(root))

    def test_exact_hybrid_contract(self):
        cfg = self._load()
        self.assertTrue(cfg.is_glm5_next)
        self.assertTrue(cfg.has_hyper_connection)
        self.assertEqual(cfg.attention_type, "hybrid_kda_dsa")
        self.assertEqual(cfg.source_fp8_block_size, (128, 128))
        self.assertEqual(cfg.linear_attention_family, "kimi_delta_attention")
        self.assertEqual(cfg.linear_num_key_heads, 64)
        self.assertEqual(cfg.linear_num_value_heads, 64)
        self.assertEqual(cfg.linear_key_head_dim, 128)
        self.assertEqual(cfg.linear_value_head_dim, 128)
        self.assertEqual(cfg.linear_gate_lower_bound, -5.0)
        self.assertEqual(cfg.index_kpool, 4)
        self.assertEqual(cfg.num_full_attention_layers, 11)
        self.assertEqual(cfg.num_moe_layers, 42)
        self.assertTrue(cfg.is_kimi_delta_attention_layer(0))
        self.assertTrue(cfg.is_dsa_layer(3))
        self.assertEqual(cfg.dsa_indexer_owner_layer(3), 3)
        self.assertIsNone(cfg.dsa_indexer_owner_layer(4))
        self.assertEqual(
            _dsa_owner_layers_for_segment(cfg, 0, 45),
            list(range(3, 45, 4)),
        )
        self.assertEqual(_dsa_owner_layers_for_segment(cfg, 0, 3), [])
        self.assertEqual(
            _dsa_resource_layers_for_segment(cfg, 4, 8),
            ([7], []),
        )
        self.assertEqual(cfg.layers_prefix, "model.language_model")

    def test_native_dsa_registration_is_attention_format_independent(self):
        cfg = self._load()
        model = KrasisModel.__new__(KrasisModel)
        model.cfg = cfg
        owner = SimpleNamespace(layer_idx=3)
        attn = SimpleNamespace(
            dsa_indexer_owner_layer=3,
            dsa_indexer=owner,
        )
        registrations = []
        store = SimpleNamespace(
            register_dsa_indexer_layer=lambda **kwargs: registrations.append(kwargs)
        )

        self.assertTrue(
            model._register_dsa_indexer_layer_on_store(store, 3, attn)
        )
        self.assertEqual(len(registrations), 1)
        self.assertEqual(registrations[0]["layer_idx"], 3)
        self.assertEqual(registrations[0]["owner_layer_idx"], 3)
        self.assertTrue(registrations[0]["owner_weights_present"])
        self.assertEqual(registrations[0]["index_topk"], cfg.index_topk)
        self.assertEqual(registrations[0]["index_kpool"], cfg.index_kpool)
        self.assertTrue(registrations[0]["index_kpool_compress"])

        self.assertFalse(
            model._register_dsa_indexer_layer_on_store(store, 0, attn)
        )
        self.assertEqual(len(registrations), 1)

        broken = SimpleNamespace(
            dsa_indexer_owner_layer=3,
            dsa_indexer=None,
        )
        with self.assertRaisesRegex(RuntimeError, "owner_weights_present=False"):
            model._register_dsa_indexer_layer_on_store(store, 3, broken)

    def test_layer_schedule_mismatch_fails_closed(self):
        def mutate(cfg):
            cfg["linear_attn_config"]["full_attn_layers"] = [3]

        with self.assertRaisesRegex(ValueError, "disagree"):
            self._load(mutate)

    def test_kpool_contract_fails_closed(self):
        def mutate(cfg):
            cfg["index_kpool_compress"] = False

        with self.assertRaisesRegex(ValueError, "compressed index_kpool"):
            self._load(mutate)

    def test_vision_processor_matches_pinned_patch_contract(self):
        vision_config = Glm53VisionConfig.from_dict(_glm53_config()["vision_config"])
        processor_config = {
            "do_rescale": True,
            "patch_expand_factor": 1,
            "merge_size": 2,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "temporal_patch_size": 2,
            "patch_size": 14,
            "min_image_tokens": 16,
            "max_image_tokens": 8000,
            "image_processor_type": "Glm5NextImageProcessor",
        }
        processor = Glm53ImagePreprocessor.from_checkpoint_config(
            processor_config,
            vision_config,
        )
        batch = processor([Image.new("RGB", (448, 448), (17, 31, 47))])
        self.assertEqual(tuple(batch["image_grid_thw"].shape), (1, 3))
        self.assertEqual(batch["image_grid_thw"].tolist(), [[1, 32, 32]])
        self.assertEqual(tuple(batch["pixel_values"].shape), (1024, 1176))
        self.assertTrue(torch.isfinite(batch["pixel_values"]).all())

        incomplete = dict(processor_config)
        del incomplete["image_std"]
        with self.assertRaisesRegex(ValueError, "missing.*image_std"):
            Glm53ImagePreprocessor.from_checkpoint_config(incomplete, vision_config)

        mismatched = dict(processor_config, patch_size=16)
        with self.assertRaisesRegex(ValueError, "geometry mismatch"):
            Glm53ImagePreprocessor.from_checkpoint_config(mismatched, vision_config)

    def test_vision_forward_uses_block_major_positions_and_merges_four_patches(self):
        cfg = Glm53VisionConfig.from_dict(
            {
                "depth": 1,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_heads": 2,
                "out_hidden_size": 8,
                "patch_size": 2,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "projection_intermediate_size": 16,
                "rms_norm_eps": 1e-5,
                "swiglu_limit": 10.0,
                "attention_bias": True,
                "hidden_act": "silu",
            }
        )
        model = Glm53VisionModel(cfg)
        grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
        self.assertEqual(
            model._position_ids(grid)[:8].tolist(),
            [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3]],
        )
        output = model(torch.randn(16, 24), grid)
        self.assertEqual(tuple(output.shape), (4, 8))
        self.assertTrue(torch.isfinite(output).all())

    def test_vision_support_requires_complete_glm53_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.json").write_text(json.dumps(_glm53_config()), encoding="utf-8")
            image_processor = {
                "do_rescale": True,
                "patch_expand_factor": 1,
                "merge_size": 2,
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "temporal_patch_size": 2,
                "patch_size": 14,
                "min_image_tokens": 16,
                "max_image_tokens": 8000,
                "image_processor_type": "Glm5NextImageProcessor",
            }
            (root / "processor_config.json").write_text(
                json.dumps({"image_processor": image_processor}),
                encoding="utf-8",
            )
            keys = {
                "model.visual.patch_embed.proj.weight": "model.safetensors",
                "model.visual.blocks.0.attn.qkv.weight": "model.safetensors",
                "model.visual.merger.proj.weight": "model.safetensors",
                "model.visual.downsample.weight": "model.safetensors",
            }
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": keys}), encoding="utf-8"
            )
            model = KrasisModel.__new__(KrasisModel)
            model.cfg = SimpleNamespace(model_path=str(root))
            model.quant_cfg = SimpleNamespace(step_vision_quant="bf16")
            self.assertTrue(model.supports_glm53_image_inputs())
            model.quant_cfg = SimpleNamespace(step_vision_quant="int4")
            self.assertFalse(model.supports_glm53_image_inputs())
            model.quant_cfg = SimpleNamespace(step_vision_quant="bf16")

            incomplete_processor = dict(image_processor)
            del incomplete_processor["image_std"]
            (root / "processor_config.json").write_text(
                json.dumps({"image_processor": incomplete_processor}),
                encoding="utf-8",
            )
            self.assertFalse(model.supports_glm53_image_inputs())
            (root / "processor_config.json").write_text(
                json.dumps({"image_processor": image_processor}),
                encoding="utf-8",
            )

            del keys["model.visual.merger.proj.weight"]
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": keys}), encoding="utf-8"
            )
            self.assertFalse(model.supports_glm53_image_inputs())


if __name__ == "__main__":
    unittest.main()
