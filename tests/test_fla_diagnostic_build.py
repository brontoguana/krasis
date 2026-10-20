"""Admission checks for isolated compiler artifacts; run via ./dev."""
import argparse
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    raise SystemExit("Use ./dev fla-diagnostic-test")

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "krasis_fla_diagnostic_build_tests", _ROOT / "src/cuda/fla/compile_kernels.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class DiagnosticBuildAdmission(unittest.TestCase):
    def test_causal_anchor_rewrite_has_exact_native_source_admission(self):
        source = (_ROOT / "src/cuda/fla/ops/kda_forward.py").read_text()
        start = source.index("def chunk_kda_fwd_kernel_intra_sub_chunk(")
        end = source.index("@triton.jit", start)
        native = source[start:end]
        causal = _MODULE.diagnostic_anchor_source(native)
        self.assertNotEqual(causal, native)
        self.assertIn("(i_ti + 0) * HV*K", causal)
        self.assertIn("tl.full([1, BK], 0,", causal)
        for invalid in ["", native.replace("min(BC//2,", "min(BC // 2,")]:
            with self.assertRaises(ValueError):
                _MODULE.diagnostic_anchor_source(invalid)

    def test_emitted_precision_must_match_request_and_be_present_on_every_dot(self):
        emitted = "%1 = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<16x16xf32>"
        self.assertEqual(_MODULE.diagnostic_dot_precisions(emitted, "tf32"), ["tf32"])
        omitted_default = emitted.replace(", inputPrecision = tf32", "")
        self.assertEqual(_MODULE.diagnostic_dot_precisions(omitted_default, "ieee"), ["ieee"])
        for bad in ["", emitted, emitted.replace(", inputPrecision = tf32", ""),
                    emitted.replace("tf32", "tf32x3") + "\n" + emitted]:
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                _MODULE.diagnostic_dot_precisions(bad, "tf32x3")

    def args(self, **changes):
        values = dict(diagnostic_intra=False, diagnostic_heads=None,
                      diagnostic_dot_precision=None, diagnostic_anchor="native", arch=None,
                      diagnostic_gate_lower_bound=None,
                      output_dir="unused-production-path")
        values.update(changes)
        return argparse.Namespace(**values)

    def test_default_build_keeps_every_kernel_and_head_variant(self):
        specs, heads, prefix = _MODULE.diagnostic_build_selection(self.args())
        self.assertIs(specs, _MODULE.KERNEL_SPECS)
        self.assertIs(heads, _MODULE.H_VALUES)
        self.assertEqual(prefix, "krasis_fla")
        self.assertTrue(all("default_dot_input_precision" not in s["options"] for s in specs))

    def test_partial_precision_cannot_change_a_production_build(self):
        for changes in [dict(diagnostic_heads=[64]),
                        dict(diagnostic_dot_precision="tf32x3"), dict(diagnostic_anchor="first"),
                        dict(diagnostic_gate_lower_bound=-2.0)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                _MODULE.diagnostic_build_selection(self.args(**changes))

    def test_diagnostic_scope_and_filename_cannot_match_package_sidecars(self):
        with tempfile.TemporaryDirectory() as root:
            for precision in ["tf32", "tf32x3", "ieee"]:
                specs, heads, prefix = _MODULE.diagnostic_build_selection(self.args(
                    diagnostic_intra=True, diagnostic_heads=[64],
                    diagnostic_dot_precision=precision, arch=[80, 120],
                    output_dir=str(Path(root) / precision)))
                self.assertEqual(len(specs), 1)
                self.assertEqual(specs[0]["name"], "chunk_kda_fwd_kernel_intra_sub_chunk")
                self.assertEqual(heads, [64])
                self.assertEqual(specs[0]["options"]["default_dot_input_precision"], precision)
                self.assertNotEqual(prefix, "krasis_fla")
                self.assertTrue(all("default_dot_input_precision" not in s["options"]
                                    for s in _MODULE.KERNEL_SPECS))

    def test_causal_center_requires_explicit_model_bound_and_preserves_causal_source(self):
        source = (_ROOT / "src/cuda/fla/ops/kda_forward.py").read_text()
        start = source.index("def chunk_kda_fwd_kernel_intra_sub_chunk(")
        native = source[start:source.index("@triton.jit", start)]
        for bound in [None, 0.0, 1.0, float("nan"), float("-inf")]:
            with self.subTest(bound=bound), self.assertRaises(ValueError):
                _MODULE.diagnostic_anchor_source(native, "causal-center", bound)
        causal = _MODULE.diagnostic_anchor_source(native, "causal-center", -2.0)
        self.assertIn("(i_ti + 0) * HV*K", causal)
        self.assertIn("(BC - 1) * -2.0", causal)
        self.assertNotIn("min(BC//2, T - i_ti - 1)", causal)
        with self.assertRaises(ValueError):
            _MODULE.diagnostic_anchor_source(native, "first", -2.0)
        with tempfile.TemporaryDirectory() as root:
            valid = self.args(diagnostic_intra=True, diagnostic_heads=[64],
                              diagnostic_dot_precision="tf32", diagnostic_anchor="causal-center",
                              diagnostic_gate_lower_bound=-2.0, arch=[80, 120],
                              output_dir=str(Path(root) / "fresh"))
            specs, _, prefix = _MODULE.diagnostic_build_selection(valid)
            self.assertEqual(specs[0]["_diagnostic_gate_lower_bound"], -2.0)
            self.assertEqual(prefix, "krasis_kda_intra_tf32_causal_center")
            valid.diagnostic_gate_lower_bound = None
            with self.assertRaises(ValueError):
                _MODULE.diagnostic_build_selection(valid)

    def test_diagnostic_rejects_missing_unsupported_or_ambiguous_targets(self):
        with tempfile.TemporaryDirectory() as root:
            valid = dict(diagnostic_intra=True, diagnostic_heads=[64],
                         diagnostic_dot_precision="tf32", arch=[120],
                         output_dir=str(Path(root) / "fresh"))
            invalid = [dict(arch=None), dict(arch=[999]), dict(arch=[120, 120]),
                       dict(diagnostic_heads=None), dict(diagnostic_heads=[0]),
                       dict(diagnostic_heads=[64, 64]), dict(diagnostic_dot_precision=None),
                       dict(diagnostic_dot_precision="unknown"), dict(output_dir="relative"),
                       dict(output_dir=root), dict(output_dir=str(_ROOT / "python/krasis/new"))]
            for changes in invalid:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    _MODULE.diagnostic_build_selection(self.args(**dict(valid, **changes)))


if __name__ == "__main__":
    unittest.main()
