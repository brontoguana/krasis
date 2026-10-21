import unittest
import contextlib
import io
import os
import tempfile
from pathlib import Path
import urllib.error
from unittest import mock

from krasis import setup


class SetupCudaSelectionTests(unittest.TestCase):
    def test_clean_cuda_bootstrap_does_not_require_wget_or_curl(self):
        calls = []
        def run(cmd, **kwargs):
            if cmd[0] in ('wget', 'curl'):
                raise FileNotFoundError(cmd[0])
            calls.append(cmd)
            return type('Result', (), {'returncode': 0})()
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(setup, '_get_nvcc_version', side_effect=[None, (12, 8)]), \
             mock.patch.object(setup, '_get_required_cuda_version', return_value=(12, 8)), \
             mock.patch.object(setup, '_need_python_dev', return_value=False), \
             mock.patch.object(setup, '_detect_distro', return_value=('debian', '24.04')), \
             mock.patch.object(setup, '_is_wsl', return_value=False), \
             mock.patch.object(setup.shutil, 'which', side_effect=lambda name: '/usr/bin/ninja' if name.startswith('ninja') else None), \
             mock.patch.object(setup.os.path, 'expanduser', return_value=str(Path(directory, '.bashrc'))), \
             mock.patch.object(setup.os, 'geteuid', return_value=0), \
             mock.patch.dict(os.environ), \
             mock.patch.object(setup.urllib.request, 'urlopen', return_value=io.BytesIO(b'keyring fixture')) as download, \
             mock.patch.object(setup, '_run', side_effect=run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(setup._install_system_deps())
            download.assert_called_once()
            self.assertTrue(any(cmd[:2] == ['dpkg', '-i'] for cmd in calls))
            self.assertTrue(any('cuda-toolkit-12-8' in cmd for cmd in calls))

    def test_keyring_network_failure_returns_false(self):
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(setup.urllib.request, 'urlopen', side_effect=urllib.error.URLError('offline')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(setup._download_cuda_keyring('https://example.invalid/keyring', str(Path(directory, 'keyring.deb'))))

    def test_empty_keyring_download_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(setup.urllib.request, 'urlopen', return_value=io.BytesIO()), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(setup._download_cuda_keyring('https://example.invalid/keyring', str(Path(directory, 'keyring.deb'))))

    def test_required_cuda_considers_all_visible_gpus(self):
        with mock.patch.object(
            setup,
            "_get_visible_gpu_compute_caps",
            return_value=[
                {"index": 0, "name": "NVIDIA RTX A4500", "capability": (8, 6)},
                {"index": 1, "name": "NVIDIA GeForce RTX 5090", "capability": (12, 0)},
            ],
        ):
            self.assertEqual(setup._get_required_cuda_version(), (12, 8))

    def test_blackwell_selects_cu128_when_driver_supports_cuda_128(self):
        with mock.patch.object(setup, "_get_required_cuda_version", return_value=(12, 8)):
            with mock.patch.object(setup, "_get_driver_cuda_version", return_value=(12, 8)):
                self.assertEqual(setup._select_torch_cuda()[:2], ("cu128", "12.8"))

    def test_driver_too_old_for_visible_gpu_fails_clearly(self):
        with mock.patch.object(setup, "_get_required_cuda_version", return_value=(12, 8)):
            with mock.patch.object(setup, "_get_driver_cuda_version", return_value=(12, 6)):
                tag, version, error = setup._select_torch_cuda()
        self.assertIsNone(tag)
        self.assertIsNone(version)
        self.assertIn("need CUDA 12.8+", error)

    def test_unsupported_torch_devices_reports_missing_sm(self):
        probe = {
            "installed": True,
            "cuda_available": True,
            "arch_list": ["sm_80", "sm_86", "sm_90"],
            "devices": [
                {"index": 0, "name": "NVIDIA GeForce RTX 5090", "capability": "12.0"},
                {"index": 1, "name": "NVIDIA RTX A4500", "capability": "8.6"},
            ],
        }
        unsupported = setup._unsupported_torch_devices(probe)
        self.assertEqual(unsupported, ["GPU 0 NVIDIA GeForce RTX 5090 (sm_120)"])

    def test_install_replaces_cuda_torch_that_lacks_visible_gpu_arch(self):
        probes = [
            {
                "installed": True,
                "cuda_available": True,
                "version": "2.12.0+cu126",
                "arch_list": ["sm_80", "sm_86", "sm_90"],
                "devices": [
                    {"index": 0, "name": "NVIDIA GeForce RTX 5090", "capability": "12.0"},
                ],
            },
            {
                "installed": True,
                "cuda_available": True,
                "version": "2.11.0+cu128",
                "arch_list": ["sm_80", "sm_86", "sm_90", "sm_120"],
                "devices": [
                    {"index": 0, "name": "NVIDIA GeForce RTX 5090", "capability": "12.0"},
                ],
            },
        ]
        calls = []

        def fake_run(cmd, check=True, **kwargs):
            calls.append(cmd)
            return type("Result", (), {"returncode": 0})()

        with mock.patch.object(setup, "_probe_torch_cuda", side_effect=probes):
            with mock.patch.object(setup, "_select_torch_cuda", return_value=("cu128", "12.8", None)):
                with mock.patch.object(setup, "_run", side_effect=fake_run):
                    self.assertTrue(setup._install_cuda_torch())

        self.assertEqual(len(calls), 1)
        self.assertIn("--force-reinstall", calls[0])
        self.assertIn("https://download.pytorch.org/whl/cu128", calls[0])


if __name__ == "__main__":
    unittest.main()
