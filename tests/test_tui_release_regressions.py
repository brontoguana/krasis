"""Release regressions for metadata-only download and cancelled side effects."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import textwrap
import unittest
from unittest import mock
from types import SimpleNamespace

from krasis import hf_downloader

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

    def test_catalogue_candidate_handoff_reaches_real_details_for_every_model(self):
        for spec in hf_downloader.supported_models():
            with self.subTest(repo=spec.repo_id), tempfile.TemporaryDirectory() as directory:
                candidate = hf_downloader.candidate_from_info(SimpleNamespace(
                    id=spec.repo_id, pipeline_tag='text-generation',
                    tags=['transformers', 'safetensors'],
                    safetensors=SimpleNamespace(total=1000, parameters={}), siblings=[],
                ), include_files=False)
                candidate = hf_downloader._apply_supported_spec(candidate, spec)
                instance = launcher.Launcher.__new__(launcher.Launcher)
                instance.models_dir = directory
                with mock.patch.object(instance, '_hf_auth_label', return_value='not logged in'), \
                     mock.patch.object(instance, '_supported_hf_models_screen', return_value=candidate), \
                     mock.patch.object(hf_downloader, 'get_supported_model_summaries', return_value=[candidate]), \
                     mock.patch.object(hf_downloader, 'get_supported_model_details', return_value=candidate) as details, \
                     mock.patch.object(launcher, '_read_key', side_effect=[launcher.KEY_ENTER, launcher.KEY_ESCAPE, launcher.KEY_ESCAPE]), \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertFalse(instance._hf_downloader_screen())
                    details.assert_called_once_with(spec.repo_id)
                    self.assertIn(spec.repo_id, output.getvalue())
                    self.assertIn('Compatibility: supported by Krasis', output.getvalue())

    def test_download_cancel_exits_promptly_with_transfer_threads_alive(self):
        code = textwrap.dedent("""
            import threading, time
            from types import SimpleNamespace
            from krasis import launcher, hf_downloader
            started = threading.Event()
            def download(*args, **kwargs):
                thread = threading.Thread(target=lambda: time.sleep(60), daemon=False)
                thread.start()
                started.set()
                time.sleep(60)
            def key(timeout):
                assert started.wait(5)
                if TRIGGER == 'signal':
                    raise KeyboardInterrupt
                return launcher.KEY_ESCAPE
            hf_downloader.download_model = download
            launcher._read_key_timeout = key
            app = launcher.Launcher.__new__(launcher.Launcher)
            app._hf_download_progress(SimpleNamespace(
                selected_files=[], selected_bytes=0, repo_id='unit/fixture', revision='abc'
            ), '/tmp/nonexistent-krasis-cancel-fixture')
            raise AssertionError('cancel returned without terminating transfer threads')
        """)
        for trigger in ('key', 'signal'):
            with self.subTest(trigger=trigger):
                script = f"TRIGGER = {trigger!r}\n" + code
                result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 130, result.stderr)
                self.assertIn('Partial files are retained', result.stdout)

    def test_configuration_scrolls_selected_option_on_small_terminals(self):
        instance = launcher.Launcher.__new__(launcher.Launcher)
        instance.cfg = launcher.LauncherConfig()
        instance.cfg.model_path = '/nonexistent-qwen-fixture'
        instance.args = SimpleNamespace()
        instance.hw = {'gpu_count': 0}
        instance.selected_gpus = []
        instance.model_info = {'arch': 'qwen3', 'layers': 36, 'experts': 0, 'max_context': 40960}
        instance.budget = None
        instance.budget_error = None
        for columns, rows in ((80, 24), (100, 30), (160, 60), (40, 12)):
            for advanced in (False, True):
                options = instance._visible_config_options(advanced)
                for cursor, option in enumerate(options):
                    with self.subTest(size=(columns, rows), advanced=advanced, option=option.key), \
                         mock.patch.object(launcher.shutil, 'get_terminal_size', return_value=os.terminal_size((columns, rows))):
                        rendered = instance._render_config_screen(cursor, show_advanced=advanced)
                        lines = rendered.splitlines()
                        self.assertLessEqual(len(lines), rows - 1)
                        selected = next(launcher._ANSI_RE.sub('', line) for line in lines if '▸' in line)
                        if columns >= 80:
                            self.assertIn(option.label, selected)
                        self.assertTrue(all(launcher._visible_len(line) <= columns - 1 for line in lines))
