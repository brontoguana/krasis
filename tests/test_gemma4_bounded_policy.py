"""Public selection, explicit opt-out and the existing unsafe KV guard."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from krasis.gemma4_bounded import enabled, DEFAULT_CONTEXT
from krasis.server import _apply_dense_serving_policy

class GemmaBoundedPolicy(unittest.TestCase):
    def test_small_device_selection_and_optout(self):
        cfg=SimpleNamespace(gemma4_text=True)
        with patch.dict(os.environ,{'KRASIS_GEMMA_8GB':'auto'}),patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(total_memory=8<<30)):
            self.assertTrue(enabled(cfg))
            with patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(total_memory=48<<30)):
                self.assertFalse(enabled(cfg))
        with patch.dict(os.environ,{'KRASIS_GEMMA_8GB':'0'}):self.assertFalse(enabled(cfg))
        with patch.dict(os.environ,{'KRASIS_GEMMA_8GB':'1'}):self.assertFalse(enabled(SimpleNamespace(gemma4_text=False)))

    def args(self,**changes):
        values=dict(gemma_8gb=True,attention_quant='bf16',dense_mlp_quant='int8',lm_head_quant='int8',kv_dtype='k6v6',ring_window_kv=False,step_vision_quant='bf16',max_context_tokens=0)
        values.update(changes);return SimpleNamespace(**values)

    def test_public_server_profile(self):
        args=self.args();cfg=SimpleNamespace(gemma4_text=True,n_routed_experts=128,model_type='gemma4_text')
        with patch.dict(os.environ,{},clear=False):_apply_dense_serving_policy(args,cfg)
        self.assertEqual((args.attention_quant,args.dense_mlp_quant,args.lm_head_quant,args.kv_dtype,args.step_vision_quant),('hqq4','int4','int4','k6v6','int4'))
        self.assertTrue(args.ring_window_kv)
        self.assertEqual(args.max_context_tokens,DEFAULT_CONTEXT)

    def test_unsafe_ring_precision_stays_rejected(self):
        with patch.dict(os.environ,{},clear=False),self.assertRaisesRegex(ValueError,'k4v4'):
            _apply_dense_serving_policy(self.args(kv_dtype='k4v4'),SimpleNamespace(gemma4_text=True,n_routed_experts=128,model_type='gemma4_text'))

    def test_launcher_small_device_defaults(self):
        from krasis.launcher import Launcher, LauncherConfig
        launcher=Launcher.__new__(Launcher)
        launcher.cfg=LauncherConfig()
        launcher.model_info={'name':'Gemma4','arch':'gemma4_text','max_context':262144}
        launcher.selected_gpus=[{'index':0,'memory_total_mb':8192}]
        with patch.dict(os.environ,{'KRASIS_GEMMA_8GB':'auto'}):
            launcher._apply_model_recommended_defaults()
            self.assertEqual(launcher._attention_modes(),['hqq4'])
            self.assertEqual(launcher._kv_choices(),['k6v6'])
            self.assertEqual(launcher._vision_choices(),['int4'])
        self.assertEqual((launcher.cfg.dense_mlp_quant,launcher.cfg.lm_head_quant,launcher.cfg.vision_quant),('int4','int4','int4'))
        self.assertEqual(launcher.cfg.max_context_tokens,DEFAULT_CONTEXT)

    def test_changed_native_kernel_invalidates_heatmap_identity(self):
        import tempfile
        from pathlib import Path
        from krasis import server
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            package=root/'python/krasis';package.mkdir(parents=True)
            module=package/'server.py';module.write_text('# identity fixture')
            kernel=root/'src/cuda/gemma4_attention.cu';kernel.parent.mkdir(parents=True)
            kernel.write_text('first kernel')
            with patch.object(server,'__file__',str(module)):
                before=server._krasis_runtime_code_hash()
                kernel.write_text('changed kernel')
                after=server._krasis_runtime_code_hash()
            self.assertNotEqual(before['sha256'],after['sha256'])

    def test_stage_precision_is_bound_to_heatmap_compatibility(self):
        from krasis import server
        from unittest.mock import Mock
        store=Mock();store.native_scalar_runtime_identity.return_value=None
        model=SimpleNamespace(_gpu_decode_store=store,_gemma_prefill_stage_format='bf16')
        with patch.object(server,'_runtime_heatmap_capture_config',side_effect=lambda args:{'kv_dtype':'k6v6'}):
            bf16=server._actual_runtime_heatmap_capture_config(model,None)
            model._gemma_prefill_stage_format='fp8_e4m3'
            fp8=server._actual_runtime_heatmap_capture_config(model,None)
        self.assertEqual(bf16['gemma_prefill_stage_format'],'bf16')
        self.assertEqual(fp8['gemma_prefill_stage_format'],'fp8_e4m3')
        matches,_=server._runtime_matches_approved_heatmap_policy(fp8,[bf16],{'ignored_runtime_fields':['attention_quant','kv_dtype']})
        self.assertFalse(matches)
        with patch.object(server,'_runtime_heatmap_capture_config',side_effect=lambda args:{'kv_dtype':'k6v6'}):
            other=server._actual_runtime_heatmap_capture_config(SimpleNamespace(_gpu_decode_store=store),None)
        self.assertNotIn('gemma_prefill_stage_format',other)

    def test_diagnostic_arithmetic_cannot_reuse_default_heatmaps(self):
        from krasis import server
        from unittest.mock import Mock
        store=Mock();store.native_scalar_runtime_identity.return_value=None
        model=SimpleNamespace(_gpu_decode_store=store,_gemma_prefill_stage_format='bf16')
        with patch.dict(os.environ,{},clear=True),patch.object(server,'_runtime_heatmap_capture_config',side_effect=lambda args:{'kv_dtype':'k6v6'}):
            ordinary=server._actual_runtime_heatmap_capture_config(model,None)
            with patch.dict(os.environ,{'KRASIS_GEMMA_DIAG_BF16_BOUNDARIES':'1'}):
                diagnostic=server._actual_runtime_heatmap_capture_config(model,None)
                other=server._actual_runtime_heatmap_capture_config(SimpleNamespace(_gpu_decode_store=store),None)
        self.assertNotIn('gemma_numerical_diagnostics',ordinary)
        self.assertNotIn('gemma_numerical_diagnostics',other)
        self.assertEqual(diagnostic['gemma_numerical_diagnostics']['KRASIS_GEMMA_DIAG_BF16_BOUNDARIES'],'1')
        matches,_=server._runtime_matches_approved_heatmap_policy(diagnostic,[ordinary],{'ignored_runtime_fields':['attention_quant','kv_dtype']})
        self.assertFalse(matches)

    def test_explicit_profile_rejects_other_models(self):
        with patch.dict(os.environ,{},clear=False),self.assertRaisesRegex(ValueError,'requires a Gemma4'):
            _apply_dense_serving_policy(self.args(),SimpleNamespace(gemma4_text=False,n_routed_experts=128,model_type='qwen3_moe'))

if __name__=='__main__':unittest.main()
