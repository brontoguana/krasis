import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_build_sidecars_module():
    script = REPO_ROOT / "scripts" / "build_sidecars.py"
    spec = importlib.util.spec_from_file_location("krasis_build_sidecars_test", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ, {"KRASIS_DEV_SCRIPT": "1"}):
        spec.loader.exec_module(module)
    return module


class BuildSidecarsHostCompilerArgsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_build_sidecars_module()

    def test_linux_adds_cuda_glibc_compatibility_defines(self) -> None:
        with (
            mock.patch.object(self.module, "IS_WINDOWS", False),
            mock.patch.dict(os.environ, {"KRASIS_NVCC_CCBIN": ""}),
        ):
            self.assertEqual(
                self.module.nvcc_host_compiler_args(),
                [
                    "-U_GNU_SOURCE",
                    "-D_DEFAULT_SOURCE",
                    "-include",
                    "src/cuda/cuda_glibc_compat.h",
                ],
            )

    def test_linux_preserves_explicit_host_compiler(self) -> None:
        with (
            mock.patch.object(self.module, "IS_WINDOWS", False),
            mock.patch.dict(os.environ, {"KRASIS_NVCC_CCBIN": "/opt/cc/bin/g++"}),
        ):
            self.assertEqual(
                self.module.nvcc_host_compiler_args(),
                [
                    "-ccbin",
                    "/opt/cc/bin/g++",
                    "-U_GNU_SOURCE",
                    "-D_DEFAULT_SOURCE",
                    "-include",
                    "src/cuda/cuda_glibc_compat.h",
                ],
            )

    def test_windows_arguments_are_unchanged(self) -> None:
        with (
            mock.patch.object(self.module, "IS_WINDOWS", True),
            mock.patch.dict(os.environ, {"KRASIS_NVCC_CCBIN": "cl.exe"}),
        ):
            self.assertEqual(
                self.module.nvcc_host_compiler_args(),
                ["-ccbin", "cl.exe"],
            )


if __name__ == "__main__":
    unittest.main()
