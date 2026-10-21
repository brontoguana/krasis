"""Exercise the Linux installer's embedded selector against mixed release assets."""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class InstallAssetTests(unittest.TestCase):
    def select(self, names):
        installer = (Path(__file__).resolve().parents[1] / "install.sh").read_text()
        selector = installer.split('RESULT=$("$PARSE_PY" -c \'', 1)[1].split(
            "' <<< \"$RELEASE_JSON\")", 1
        )[0]
        env = dict(os.environ, KRASIS_ARCH="x86_64", KRASIS_PY_VERS="311",
                   KRASIS_PY_BINS="python3.11", KRASIS_PY_DOTS="3.11")
        release = {"tag_name": "v1.2.3", "assets": [
            {"name": name, "browser_download_url": "https://example.invalid/" + name}
            for name in names
        ]}
        return subprocess.check_output(
            [sys.executable, "-c", selector], input=json.dumps(release),
            text=True, env=env,
        ).splitlines()

    def test_linux_wheel_is_selected_from_cross_platform_release(self):
        wheel = "krasis-1.2.3-cp311-cp311-manylinux_2_28_x86_64.whl"
        result = self.select([
            "krasis-1.2.3-cp311-cp311-win_amd64.whl",
            "krasis-1.2.3-cp311-cp311-macosx_11_0_x86_64.whl",
            "krasis-1.2.3-cp311-cp311-manylinux_2_28_aarch64.whl",
            "krasis-1.2.3-cp312-cp312-manylinux_2_28_x86_64.whl",
            wheel,
        ])
        self.assertEqual(result[0], "WHEEL")
        self.assertTrue(result[-1].endswith(wheel))

    def test_sidecar_bundle_is_not_a_source_distribution(self):
        self.assertEqual(self.select([
            "krasis-sidecars-abc123.tar.gz", "krasis-windows-runtime.tar.gz",
        ])[0], "NONE")

    def test_real_sdist_remains_available_after_sidecar_bundle(self):
        result = self.select(["krasis-sidecars-abc123.tar.gz", "krasis-1.2.3.tar.gz"])
        self.assertEqual(result[0], "SDIST")
        self.assertTrue(result[-1].endswith("krasis-1.2.3.tar.gz"))


if __name__ == "__main__":
    unittest.main()
