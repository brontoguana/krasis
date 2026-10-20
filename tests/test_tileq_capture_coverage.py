import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

os.environ.setdefault("KRASIS_DEV_SCRIPT", "1")
from scripts import tileq_build as tileq


class CaptureCoverageTests(unittest.TestCase):
    def layer(self, ids):
        rows = len(ids)
        return tileq.CaptureLayer(
            model_layer=3, expert_input_size=1, intermediate_size=1,
            topk=1, expert_count=2, expert_inputs=torch.ones(rows, 1),
            topk_ids=torch.tensor(ids).reshape(rows, 1),
            topk_weights=torch.ones(rows, 1), down_inputs=torch.ones(rows, 1, 1),
        )

    def test_aggregate_counts_do_not_mask_missing_corpus_observations(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            report = root / "coverage.json"
            args = argparse.Namespace(capture_dir=root / "a", extra=[root / "b"],
                                      coverage_report=report, minimum_routes=2)
            with patch.object(tileq, "capture_metadata", side_effect=[
                {"calibration_sha256": "a" * 64}, {"calibration_sha256": "b" * 64}
            ]), patch.object(tileq, "capture_inventory", side_effect=[
                ({"layer_00003.ktc": [root / "a.ktc"]}, []),
                ({"layer_00003.ktc": [root / "b.ktc"]}, []),
            ]), patch.object(tileq, "read_capture_layer", side_effect=[
                self.layer([0, 0]), self.layer([1, 1, 1]),
            ]):
                tileq.command_inspect_capture(args)
            value = json.loads(report.read_text())
            self.assertEqual(value["layers"][0]["route_counts"], [2, 3])
            self.assertTrue(value["aggregate_route_gate_pass"])
            self.assertFalse(value["per_corpus_gate_pass"])
            self.assertEqual(value["layers"][0]["corpora"][0]["unobserved_experts"], [1])
            self.assertEqual(value["layers"][0]["corpora"][1]["unobserved_experts"], [0])
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                tileq.command_inspect_capture(args)

    def test_failed_inspection_never_publishes_report(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            args = argparse.Namespace(capture_dir=root, extra=[],
                                      coverage_report=root / "report.json", minimum_routes=2)
            with patch.object(tileq, "capture_metadata", return_value={}), \
                    patch.object(tileq, "capture_inventory", return_value=(
                        {"layer_00003.ktc": [root / "bad.ktc"]}, [])), \
                    patch.object(tileq, "read_capture_layer", side_effect=RuntimeError("corrupt")):
                with self.assertRaisesRegex(RuntimeError, "corrupt"):
                    tileq.command_inspect_capture(args)
            self.assertFalse(args.coverage_report.exists())


if __name__ == "__main__":
    unittest.main()
