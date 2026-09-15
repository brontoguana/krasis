import unittest
from unittest import mock

from krasis import setup


class SetupCudaSelectionTests(unittest.TestCase):
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

    def test_unsupported_torch_devices_reports_genuinely_missing_sm(self):
        """A pre-Ampere GPU (sm_75) is unsupported when torch only has sm_80+."""
        probe = {
            "installed": True,
            "cuda_available": True,
            "arch_list": ["sm_80", "sm_86", "sm_90"],
            "devices": [
                {"index": 0, "name": "NVIDIA GeForce GTX 1650", "capability": "7.5"},
            ],
        }
        unsupported = setup._unsupported_torch_devices(probe)
        self.assertEqual(unsupported, ["GPU 0 NVIDIA GeForce GTX 1650 (sm_75)"])

    def test_unsupported_torch_devices_accepts_forward_compat(self):
        """Ada (sm_89) is supported via sm_86 forward compatibility."""
        probe = {
            "installed": True,
            "cuda_available": True,
            "arch_list": ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"],
            "devices": [
                {"index": 0, "name": "NVIDIA GeForce RTX 4060 Ti", "capability": "8.9"},
            ],
        }
        unsupported = setup._unsupported_torch_devices(probe)
        self.assertEqual(unsupported, [])

    def test_install_replaces_cuda_torch_that_lacks_visible_gpu_arch(self):
        probes = [
            {
                "installed": True,
                "cuda_available": True,
                "version": "2.12.0+cu126",
                "arch_list": ["sm_80", "sm_86", "sm_90"],
                "devices": [
                    {"index": 0, "name": "NVIDIA GeForce GTX 1650", "capability": "7.5"},
                ],
            },
            {
                "installed": True,
                "cuda_available": True,
                "version": "2.11.0+cu124",
                "arch_list": ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"],
                "devices": [
                    {"index": 0, "name": "NVIDIA GeForce GTX 1650", "capability": "7.5"},
                ],
            },
        ]
        calls = []

        def fake_run(cmd, check=True, **kwargs):
            calls.append(cmd)
            return type("Result", (), {"returncode": 0})()

        with mock.patch.object(setup, "_probe_torch_cuda", side_effect=probes):
            with mock.patch.object(setup, "_select_torch_cuda", return_value=("cu124", "12.4", None)):
                with mock.patch.object(setup, "_run", side_effect=fake_run):
                    self.assertTrue(setup._install_cuda_torch())

        self.assertEqual(len(calls), 1)
        self.assertIn("--force-reinstall", calls[0])
        self.assertIn("https://download.pytorch.org/whl/cu124", calls[0])


if __name__ == "__main__":
    unittest.main()
