"""Dense Qwen3.5 contracts from Qwen/Qwen3.5-27B's upstream config.

The reference equations deliberately use the unfused checkpoint layout.
GPU full-model numerical acceptance is recorded separately.
"""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from krasis.config import ModelConfig
from krasis.model import _compute_layer_groups, KrasisModel
from krasis.weight_loader import WeightLoader
from krasis.server import _apply_dense_serving_policy
from krasis.vram_budget import compute_launcher_budget
from krasis.launcher import (_STRUCTURALLY_SUPPORTED_MODEL_TYPES, OPTIONS,
    _is_option_visible, LauncherConfig, Launcher)

FIXTURE = Path(__file__).parent / 'fixtures/qwen35-27b-config.json'

class DenseQwen35Tests(unittest.TestCase):
    def load(self, mutate=None):
        raw = json.loads(FIXTURE.read_text())
        if mutate:
            mutate(raw['text_config'])
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        Path(tmp.name, 'config.json').write_text(json.dumps(raw))
        return ModelConfig.from_model_path(tmp.name)

    def test_exact_dense_geometry_and_schedule(self):
        cfg = self.load()
        self.assertTrue(cfg.norm_bias_one)
        self.assertTrue(cfg.gated_attention)
        self.assertEqual((cfg.hidden_size, cfg.intermediate_size), (5120, 17408))
        self.assertEqual((cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim), (24, 4, 256))
        self.assertEqual((cfg.linear_num_key_heads, cfg.linear_num_value_heads), (16, 48))
        contract = json.loads((Path(__file__).parents[1] /
            'python/krasis/fla_sidecar_contract.json').read_text())
        self.assertIn(cfg.linear_num_value_heads, contract['h_values'])
        self.assertEqual(cfg.num_moe_layers, 0)
        self.assertEqual(cfg.first_k_dense_replace, 64)
        self.assertFalse(any(cfg.is_moe_layer(i) for i in range(64)))
        for i in range(64):
            self.assertEqual(cfg.is_full_attention_layer(i), i % 4 == 3)
        rank = SimpleNamespace(layer_start=0, layer_end=64)
        for divisor in (0, 1, 8):
            self.assertEqual(_compute_layer_groups(rank, cfg, divisor), [(list(range(64)), [])])
        self.assertEqual(cfg.partial_rotary_factor, .25)
        self.assertIn(cfg.model_type, _STRUCTURALLY_SUPPORTED_MODEL_TYPES)

    def test_reject_incompatible_dense_variants(self):
        for key, val in [('num_experts', 8), ('hidden_act', 'gelu'),
                         ('linear_num_value_heads', 47), ('hidden_activation', 'gelu'),
                         ('layer_types', ['moe'] * 64)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(lambda c: c.update({key: val}))

    def test_delta_norm_roles_match_upstream_equation(self):
        loader = WeightLoader.__new__(WeightLoader)
        loader.cfg = self.load()
        w = torch.tensor([-.25, .125, .5, -.125])
        x = torch.tensor([[1., -2., .25, 4.]])
        base = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        for label in ('model.language_model.norm.weight',
                      'model.language_model.layers.3.input_layernorm.weight',
                      'model.language_model.layers.3.post_attention_layernorm.weight',
                      'model.language_model.layers.3.self_attn.q_norm.weight',
                      'model.language_model.layers.3.self_attn.k_norm.weight'):
            actual = base * loader._maybe_apply_delta_rms_bias(w, label)
            torch.testing.assert_close(actual, base * (1 + w))
        # GDN's gated norm uses a direct scale, unlike the layer and Q/K norms.
        label = 'model.language_model.layers.0.linear_attn.norm.weight'
        torch.testing.assert_close(loader._maybe_apply_delta_rms_bias(w, label), w)

    def test_gdn_three_value_heads_per_key_fusion(self):
        loader = WeightLoader.__new__(WeightLoader)
        loader.cfg = SimpleNamespace(layers_prefix='model', linear_num_key_heads=2,
            linear_key_head_dim=4, linear_num_value_heads=6, linear_value_head_dim=4,
            norm_bias_one=True)
        loader._weight_map = {}
        torch.manual_seed(17)
        raw = {name: torch.randn(rows, 8) for name, rows in
               [('in_proj_qkv.weight', 40), ('in_proj_z.weight', 24),
                ('in_proj_b.weight', 6), ('in_proj_a.weight', 6), ('out_proj.weight', 8)]}
        for name in ('conv1d.weight', 'A_log', 'dt_bias', 'norm.weight'):
            raw[name] = torch.ones(4)
        loader._load_bf16 = lambda name, device: raw[name.split('linear_attn.')[1]]
        fused = loader.load_linear_attention_weights(0, torch.device('cpu'))
        x = torch.randn(3, 8)
        q, k, v = (x @ raw['in_proj_qkv.weight'].T).split([8, 8, 24], -1)
        z = x @ raw['in_proj_z.weight'].T
        expected = torch.cat([q.reshape(3,2,4), k.reshape(3,2,4),
                              v.reshape(3,2,12), z.reshape(3,2,12)], -1).flatten(1)
        torch.testing.assert_close(x @ fused['in_proj_qkvz'].T, expected)
        b, a = [x @ raw[f'in_proj_{name}.weight'].T for name in ('b', 'a')]
        expected_ba = torch.cat([b.reshape(3,2,3), a.reshape(3,2,3)], -1).flatten(1)
        torch.testing.assert_close(x @ fused['in_proj_ba'].T, expected_ba)
        torch.testing.assert_close(fused['norm_weight'], raw['norm.weight'])

    def test_unaligned_gdn_prefill_uses_native_materialization(self):
        cfg = self.load()
        self.assertTrue(cfg.requires_hqq_prefill_materialization)
        aligned = self.load(lambda c: c.update(linear_num_value_heads=32))
        self.assertFalse(aligned.requires_hqq_prefill_materialization)

    def test_dense_defaults_and_explicit_precision_roundtrip(self):
        cfg = self.load()
        launcher = Launcher.__new__(Launcher)
        launcher.cfg = LauncherConfig()
        launcher.cfg.model_path = cfg.model_path
        launcher.model_info = {'arch':'qwen3_5_text', 'path':cfg.model_path, 'max_context':262144}
        launcher._apply_model_recommended_defaults()
        for key in ('attention_quant','kv_dtype','dense_mlp_quant','lm_head_quant'):
            self.assertEqual(getattr(launcher.cfg, key), 'bf16')
        self.assertIn('bf16', launcher._attention_choices())
        for key in ('dense_mlp_quant','lm_head_quant'):
            launcher._cycle_value(next(o for o in OPTIONS if o.key == key), 1)
            self.assertEqual(getattr(launcher.cfg, key), 'int8')
        launcher._apply_model_recommended_defaults()
        self.assertEqual(launcher.cfg.dense_mlp_quant, 'int8')
        self.assertEqual(launcher.cfg.lm_head_quant, 'int8')
        launcher.cfg.apply_saved({'CFG_DENSE_MLP_QUANT':'int8', 'CFG_LM_HEAD_QUANT':'int8',
            'CFG_ATTENTION_QUANT':'hqq6', 'CFG_KV_DTYPE':'k6v6'})
        launcher._apply_model_recommended_defaults()
        self.assertEqual(launcher.cfg.attention_quant, 'hqq6')
        self.assertEqual(launcher.cfg.kv_dtype, 'k6v6')

    def test_dense_text_admission_does_not_advertise_unqualified_vision(self):
        cfg = self.load()
        Path(cfg.model_path, 'preprocessor_config.json').write_text('{}')
        Path(cfg.model_path, 'model.safetensors.index.json').write_text(json.dumps({
            'weight_map': {'model.visual.patch_embed.proj.weight': 'vision.safetensors'}}))
        model = SimpleNamespace(cfg=cfg)
        self.assertFalse(KrasisModel.supports_qwen_image_inputs(model))
        # The existing supported Qwen vision family retains its prior behavior.
        cfg.model_type = 'qwen3_5_moe_text'
        self.assertTrue(KrasisModel.supports_qwen_image_inputs(model))

    def test_dense_topology_rejects_unvalidated_multi_gpu(self):
        launcher = Launcher.__new__(Launcher)
        launcher.cfg = LauncherConfig()
        launcher.model_info = {'arch': 'qwen3_5_text'}
        launcher.cfg.selected_gpu_indices = [0]
        launcher._validate_model_topology()
        launcher.cfg.selected_gpu_indices = [0, 1]
        with self.assertRaisesRegex(ValueError, 'one selected GPU'):
            launcher._validate_model_topology()

    def test_dense_serving_does_not_enter_hcs_expert_admission(self):
        args = SimpleNamespace(hcs=True)
        _apply_dense_serving_policy(args, self.load())
        self.assertFalse(args.hcs)
        for requested in (True, False):
            args = SimpleNamespace(hcs=requested)
            _apply_dense_serving_policy(args, SimpleNamespace(n_routed_experts=8))
            self.assertEqual(args.hcs, requested)

    def test_dense_load_skips_routed_expert_engine(self):
        model = SimpleNamespace(cfg=self.load(), krasis_engine='must be cleared')
        with mock.patch('krasis.KrasisEngine') as engine:
            KrasisModel._load_cpu_experts(model, gpu_only=True)
            engine.assert_not_called()
        self.assertIsNone(model.krasis_engine)

    def test_dense_warmup_does_not_launch_unused_expert_kernel(self):
        model = SimpleNamespace(cfg=self.load(), layers=[])
        with mock.patch('torch.cuda.mem_get_info', return_value=(1000000, 2000000)), \
             mock.patch('torch.cuda.set_device'), mock.patch('torch.cuda.synchronize'), \
             mock.patch('torch.cuda.empty_cache'), \
             mock.patch('krasis.marlin_utils.marlin_make_workspace') as workspace, \
             mock.patch('krasis.marlin_utils.gptq_marlin_gemm') as gemm:
            KrasisModel.warmup_cuda_runtime(model, [torch.device('cuda:0')], execution_backend='rust')
            workspace.assert_not_called()
            gemm.assert_not_called()

    def test_dense_tui_exposes_precision_and_hides_inapplicable_experts(self):
        info = {"arch": "qwen3_5_text", "dense_layers": 64, "experts": 0}
        cfg = LauncherConfig()
        visible = {o.key for o in OPTIONS if _is_option_visible(o, info, cfg, True)}
        self.assertIn("dense_mlp_quant", visible)
        self.assertIn("lm_head_quant", visible)
        self.assertIn("attention_quant", visible)
        self.assertNotIn("gpu_expert_bits", visible)
        self.assertNotIn("hcs_host_cache_mode", visible)
        self.assertNotIn("dynamic_hcs_tail_blocks", visible)
        dense = next(o for o in OPTIONS if o.key == "dense_mlp_quant")
        self.assertEqual(dense.choices, ["int8", "bf16"])

    def test_budget_charges_all_dense_mlps_and_no_experts(self):
        cfg = self.load()
        budget = compute_launcher_budget(cfg.model_path, [64], attention_quant='bf16',
            gpu_vram_mb=96000, total_ram_gb=512, max_context_tokens=32768)
        self.assertGreater(budget['ram_dense_load_mb'], 50000)
        rank = budget['ranks'][0]
        self.assertEqual(rank['dense_layers'], 64)
        self.assertEqual(rank['moe_layers'], 0)
        # Native dense execution retains INT8 source plus BF16 execution matrices.
        self.assertGreaterEqual(rank['dense_mlp_mb'] * 1024**2, 64 * 3 * 5120 * 17408 * 3)
        bf16 = compute_launcher_budget(cfg.model_path, [64], attention_quant='bf16',
            dense_mlp_quant='bf16', lm_head_quant='bf16',
            gpu_vram_mb=96000, total_ram_gb=512, max_context_tokens=32768)['ranks'][0]
        self.assertAlmostEqual(bf16['dense_mlp_mb'] * 1024**2, 64 * 3 * 5120 * 17408 * 2)
        self.assertGreater(rank['dense_mlp_mb'], bf16['dense_mlp_mb'])
        self.assertGreater(rank['lm_head_mb'], bf16['lm_head_mb'])

if __name__ == '__main__':
    unittest.main()
