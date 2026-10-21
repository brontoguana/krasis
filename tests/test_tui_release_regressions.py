"""Release regressions for metadata-only download and cancelled side effects."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from krasis import launcher, vram_budget


class TUIReleaseRegressions(unittest.TestCase):
    def test_catalogue_ram_does_not_resolve_checkpoint_cache(self):
        config = {
            'model_type': 'qwen3_moe', 'hidden_size': 2048,
            'intermediate_size': 4096, 'moe_intermediate_size': 768,
            'num_hidden_layers': 4, 'num_attention_heads': 16,
            'num_key_value_heads': 4, 'n_routed_experts': 32,
            'num_experts_per_tok': 4, 'vocab_size': 1000,
            'max_position_embeddings': 32768,
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'config.json').write_text(json.dumps(config))
            with mock.patch.object(vram_budget, '_hqq_attention_layer_bytes_for_budget',
                                   side_effect=AssertionError('requires full checkpoint')):
                for mode in ('bf16', 'hqq4', 'hqq46', 'hqq46_auto', 'hqq6', 'hqq68_auto', 'hqq8'):
                    with self.subTest(mode=mode):
                        size, source = vram_budget.estimate_int4_runtime_ram_bytes(
                            directory, mode, hqq_auto_budget_pct=10, dimensions_only=True)
                        self.assertGreater(size, 0)
                        self.assertIn('dimensions', source)

    def test_save_escape_writes_nothing_and_editor_keeps_existing_value(self):
        instance = launcher.Launcher.__new__(launcher.Launcher)
        instance.cfg = launcher.LauncherConfig()
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                with mock.patch.object(launcher, '_read_key', return_value=launcher.KEY_ESCAPE), \
                     contextlib.redirect_stdout(io.StringIO()):
                    instance._save_config_screen()
                    self.assertEqual(launcher._edit_value('Host', 'localhost'), 'localhost')
                self.assertEqual(list(Path(directory).iterdir()), [])
            finally:
                os.chdir(previous)
