"""Exercise native dense defaults through the public launch/config surfaces."""
import json
import argparse
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from krasis.attention_backend import attention_quant_cache_nbits, attention_quant_nbits
from krasis.launcher import (Launcher, LauncherConfig, _apply_cli_overrides,
    _format_attention_quant_value, parse_args, _save_config)
from krasis import server
from krasis.server import _apply_dense_serving_policy
from krasis.config import ModelConfig
from krasis.qwen_dense_stream import estimated_host_peak_bytes
from krasis.vram_budget import compute_launcher_budget


class QwenInt4LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name
        fixture = Path(__file__).parent/'fixtures/qwen35-27b-config.json'
        Path(self.root, 'config.json').write_bytes(fixture.read_bytes())
        self.env = mock.patch.dict(os.environ, {'KRASIS_QWEN_DENSE_STREAM':'1'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def launcher(self):
        launcher = Launcher.__new__(Launcher)
        launcher.cfg = LauncherConfig()
        launcher.cfg.model_path = self.root
        launcher.model_info = dict(arch='qwen3_5_text', path=self.root, max_context=262144)
        return launcher

    def test_default_profile_survives_save_reload_and_capability_validation(self):
        launcher = self.launcher()
        launcher._apply_model_recommended_defaults()
        self.assertEqual((launcher.cfg.attention_quant, launcher.cfg.dense_mlp_quant,
            launcher.cfg.lm_head_quant, launcher.cfg.kv_dtype), ('int4','int4','int4','bf16'))
        self.assertEqual(launcher.cfg.max_context_tokens,4096)
        self.assertEqual(_format_attention_quant_value(launcher.cfg.attention_quant),'INT4')
        saved = launcher.cfg.to_save_dict()
        launcher.cfg = LauncherConfig()
        launcher.cfg.apply_saved(saved)
        launcher._apply_model_recommended_defaults()
        launcher._validate_model_capabilities()
        self.assertEqual(launcher.cfg.attention_quant,'int4')
        self.assertTrue(launcher.cfg._attention_quant_explicit)

    def test_explicit_cli_accepts_only_native_bounded_int4(self):
        with mock.patch('sys.argv',['krasis','--model-path',self.root,'--attention-quant','int4']):
            args=parse_args()
        cfg=LauncherConfig()
        _apply_cli_overrides(cfg,args)
        self.assertEqual(cfg.attention_quant,'int4')
        with mock.patch.dict(os.environ,{'KRASIS_QWEN_DENSE_STREAM':'0'}):
            with self.assertRaisesRegex(ValueError,'Unsupported --attention-quant'):
                _apply_cli_overrides(LauncherConfig(),args)
            with self.assertRaisesRegex(ValueError,'Unsupported saved CFG_ATTENTION_QUANT'):
                LauncherConfig().apply_saved({'MODEL_PATH':self.root,'CFG_ATTENTION_QUANT':'int4'})

    def test_server_reads_the_actual_launcher_config_before_gpu_startup(self):
        launcher=self.launcher()
        launcher._apply_model_recommended_defaults()
        path=Path(self.root,'launch.conf')
        _save_config(str(path),launcher.cfg.to_save_dict())
        captured={}
        class ParsedConfig(Exception):
            pass
        def stop_after_config(parser,*args,**kwargs):
            captured.update(parser._defaults)
            raise ParsedConfig()
        with mock.patch('sys.argv',['krasis.server','--config',str(path)]), \
             mock.patch.object(argparse.ArgumentParser,'parse_args',stop_after_config), \
             mock.patch.dict(os.environ):
            with self.assertRaises(ParsedConfig):
                server.main()
        self.assertEqual(captured['attention_quant'],'int4')
        self.assertEqual(captured['model_path'],self.root)

    def test_native_precision_does_not_select_hqq_cache(self):
        self.assertEqual(attention_quant_nbits('int4'),4)
        self.assertIsNone(attention_quant_cache_nbits('int4'))
        self.assertEqual(attention_quant_cache_nbits('hqq4'),4)

    def test_launcher_ram_budget_matches_native_single_owner_guard(self):
        cfg=ModelConfig.from_model_path(self.root)
        budget=compute_launcher_budget(self.root,[64],attention_quant='int4',
            dense_mlp_quant='int4',lm_head_quant='int4',kv_dtype='bf16',
            gpu_vram_mb=8192,total_ram_gb=32,max_context_tokens=4096)
        self.assertEqual(budget['ram_total_mb']*(1<<20),estimated_host_peak_bytes(cfg,packed=True))
        self.assertLess(budget['ram_total_mb'],24*1024)
        self.assertEqual(budget['ram_hqq_host_staging_mb'],0)

    def test_other_server_models_cannot_activate_dense_int4_attention(self):
        cfg=SimpleNamespace(model_type='other',gemma4_text=False,n_routed_experts=1)
        args=SimpleNamespace(gemma_8gb=False,attention_quant='int4')
        with self.assertRaisesRegex(ValueError,'bounded Qwen'):
            _apply_dense_serving_policy(args,cfg)
