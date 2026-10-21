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

    def test_suite_cancel_returns_to_modes_without_creating_runner(self):
        instance = launcher.Launcher.__new__(launcher.Launcher)
        instance.script_dir = "/nonexistent-checkout"
        with mock.patch.object(launcher, '_launch_mode_screen', side_effect=['suite', 'launch']), \
             mock.patch.object(launcher, '_edit_value', return_value=''), \
             mock.patch('krasis.suite.SuiteRunner') as runner, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(instance._select_launch_mode(), ('launch', None))
            runner.assert_not_called()

    def test_suite_missing_and_invalid_files_return_to_modes(self):
        instance = launcher.Launcher.__new__(launcher.Launcher)
        instance.script_dir = "/nonexistent-checkout"
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory, 'invalid.toml')
            invalid.write_text('model = [')
            for path in (str(Path(directory, 'missing.toml')), str(invalid)):
                with self.subTest(path=path), \
                     mock.patch.object(launcher, '_launch_mode_screen', side_effect=['suite', None]), \
                     mock.patch.object(launcher, '_edit_value', return_value=path), \
                     mock.patch.object(instance, '_message_screen') as message, \
                     mock.patch('krasis.suite.create_run_dir', return_value=Path(directory, 'run')), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(instance._select_launch_mode(), (None, None))
                    message.assert_called_once()

    def test_suite_selection_uses_editable_path_in_installed_layout(self):
        instance = launcher.Launcher.__new__(launcher.Launcher)
        instance.script_dir = "/nonexistent-installed-layout"
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, 'custom.toml')
            config.write_text('[[config]]\nnum_gpus = 1\n[[model]]\nname = "fixture"\npath = "' + directory + '"\n')
            with mock.patch.object(launcher, '_launch_mode_screen', return_value='suite'), \
                 mock.patch.object(launcher, '_edit_value', return_value=str(config)), \
                 mock.patch('krasis.suite.create_run_dir', return_value=Path(directory, 'run')), \
                 mock.patch('krasis.suite.SuiteRunner.run_all', side_effect=AssertionError('started before selection returned')), \
                 contextlib.redirect_stdout(io.StringIO()):
                mode, runner = instance._select_launch_mode()
                self.assertEqual(mode, 'suite')
                self.assertEqual(runner.config_path, str(config))
                self.assertEqual(len(runner.load_config()), 1)
