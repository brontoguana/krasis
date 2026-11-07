"""Configuration and joint-memory contracts for bounded Qwen, without downloads."""
import json
import io
import os
import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from krasis.config import ModelConfig
from krasis.qwen_dense_stream import memory_plan, enabled, source_bf16_bytes, host_memory_plan, load_host_backed
from unittest import mock

class BoundedQwenContracts(unittest.TestCase):
    def test_arete_recurrent_gate_semantics_and_variants(self):
        fixture = Path(__file__).parent/'fixtures/arete-27b-config.json'
        raw = json.loads(fixture.read_text())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'config.json'
            for gate in ('swish','silu'):
                raw['text_config']['output_gate_type']=gate
                path.write_text(json.dumps(raw))
                cfg=ModelConfig.from_model_path(d)
                self.assertTrue(cfg.gated_attention)  # independent sigmoid full attention
                self.assertEqual(cfg.num_hidden_layers,64)
            raw['text_config']['output_gate_type']='sigmoid'
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'recurrent output gates'):
                ModelConfig.from_model_path(d)
        # vLLM's GDN reader selects SiLU for swish; these equations differ
        # substantially from replacing GDN with the full-attention sigmoid.
        z=torch.tensor([-4.,-1.,0.,1.,4.])
        self.assertTrue(torch.equal(torch.nn.functional.silu(z),z*torch.sigmoid(z)))
        self.assertGreater((torch.nn.functional.silu(z)-torch.sigmoid(z)).abs().max(),2.)

    @mock.patch.dict('os.environ', {}, clear=True)
    def test_default_policy_preserves_other_model_sizes(self):
        target=SimpleNamespace(model_type='qwen3_5_text',hidden_size=5120,num_hidden_layers=64,intermediate_size=17408)
        self.assertTrue(enabled(target))
        for model_type,hidden,layers in [('qwen3_5_text',4096,32),('qwen3_5_moe_text',5120,64),('glm5_next_text',5120,64)]:
            self.assertFalse(enabled(SimpleNamespace(model_type=model_type,hidden_size=hidden,num_hidden_layers=layers,intermediate_size=17408)))

    def test_host_budget_counts_complete_text_model_and_rejects_early(self):
        raw=json.loads((Path(__file__).parent/'fixtures/qwen35-27b-config.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            Path(d,'config.json').write_text(json.dumps(raw));cfg=ModelConfig.from_model_path(d)
            self.assertEqual(source_bf16_bytes(cfg),53786705920)
            cfg.max_position_embeddings=4096
            with mock.patch('krasis.qwen_dense_stream.available_host_bytes',return_value=1<<30):
                loader=mock.Mock()
                model=SimpleNamespace(cfg=cfg,all_devices=['cuda:0'],stream_attention=False,
                    quant_cfg=SimpleNamespace(attention='bf16',dense_mlp='bf16',lm_head='bf16'))
                with self.assertRaisesRegex(RuntimeError,'available host RAM'):load_host_backed(model,loader)
                self.assertFalse(loader.mock_calls)
            model.quant_cfg.dense_mlp='int8'
            with self.assertRaisesRegex(ValueError,'source BF16'):load_host_backed(model,loader)
            self.assertFalse(loader.mock_calls)

    def test_bounded_launcher_renders_shared_slab_without_printing(self):
        from krasis.launcher import Launcher,LauncherConfig
        from krasis.vram_budget import compute_launcher_budget
        raw=json.loads((Path(__file__).parent/'fixtures/qwen35-27b-config.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            Path(d,'config.json').write_text(json.dumps(raw))
            launcher=Launcher.__new__(Launcher);launcher.cfg=LauncherConfig();launcher.cfg.model_path=d
            launcher.cfg.attention_quant='bf16';launcher.cfg.dense_mlp_quant='bf16';launcher.cfg.lm_head_quant='bf16';launcher.cfg.kv_dtype='bf16';launcher.cfg.kv_cache_mb=256;launcher.cfg.max_context_tokens=4096
            launcher.selected_gpus=[dict(index=0,name='Test GPU',vram_mb=6144)]
            launcher.hw=dict(gpu_count=1,gpu_model='Test GPU',gpu_vram_mb=6144,total_ram_gb=128)
            launcher.model_info=dict(arch='qwen3_5_text',layers=64,experts=0,dense_layers=64,max_context=262144,kv_dim=2048,num_kv_layers=16,validation_status='unvalidated')
            launcher.budget_error=None;launcher.krasis_home=d;launcher.models_dir=d
            launcher.budget=compute_launcher_budget(d,[64],attention_quant='bf16',dense_mlp_quant='bf16',lm_head_quant='bf16',kv_dtype='bf16',gpu_vram_mb=6144,total_ram_gb=128,max_context_tokens=4096,kv_cache_mb=256)
            output=io.StringIO()
            with mock.patch('shutil.get_terminal_size',return_value=os.terminal_size((160,100))),contextlib.redirect_stdout(output):
                rendered=launcher._render_config_screen(0)
            self.assertEqual(output.getvalue(),'')
            self.assertIn('Weight slab',rendered)
            launcher._compute_budget=lambda:launcher.budget
            with contextlib.redirect_stdout(output):launcher.print_summary()
            self.assertIn('Weight slab',output.getvalue())

    def test_budget_includes_all_phases_and_reserve(self):
        c=SimpleNamespace(num_full_attention_layers=16,num_key_value_heads=4,num_attention_heads=24,
            head_dim=256,num_hidden_layers=64,linear_num_value_heads=48,
            linear_key_head_dim=128,linear_value_head_dim=128)
        p=memory_plan(c,248320*5120*2,4096)
        self.assertLess(p['estimated_peak_bytes'],8192<<20)
        self.assertEqual(p['allowance_bytes']['reserve'],600<<20)
        self.assertEqual(p['allowance_bytes']['kv'],256<<20)
        self.assertLess(memory_plan(c,248320*5120*2,4096,6144)['estimated_peak_bytes'],6144<<20)
        self.assertGreater(p['allowance_bytes']['reference_attention'],96<<20)
        self.assertEqual(p['allowance_bytes']['vision_linear_workspace'],1<<20)
        for context,limit in [(4097,8192),(4096,2048)]:
            with self.assertRaises(ValueError):memory_plan(c,248320*5120*2,context,limit)

if __name__=='__main__':unittest.main()
