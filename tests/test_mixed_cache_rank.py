import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


os.environ.setdefault("KRASIS_DEV_SCRIPT", "1")

from scripts import mixed_cache_rank as rank


class MixedCacheRankTests(unittest.TestCase):
    def test_blended_ranking_leaves_required_experts_unscored(self) -> None:
        rare = {"coverage_required_int8": True, "moe_layer": 0, "expert": 0}
        candidates = [rare, *[
            {"moe_layer": 0, "model_layer": 3, "expert": expert,
             "int4_bytes": 100, "int8_bytes": 180,
             "expected_gain_units": expected, "tail_gain_units": tail}
            for expert, expected, tail in ((1, 10, 1), (2, 1, 20))
        ]]
        rank.assign_quality_ranks(candidates)
        self.assertEqual(rare, {"coverage_required_int8": True, "moe_layer": 0, "expert": 0})
        self.assertEqual([item["expected_rank"] for item in candidates[1:]], [0, 1])
        self.assertEqual([item["tail_rank"] for item in candidates[1:]], [1, 0])
        self.assertEqual([item["blended_rank"] for item in candidates[1:]], [1, 0])
        rank.assign_quality_ranks([rare])
        self.assertEqual(len(rare), 3)

    def test_materialized_int8_uses_actual_rust_codes_and_zero_bits(self) -> None:
        source = torch.tensor([[127., .5, -.5, .1, -.1, 2.5, -2.5, -0.]])
        actual = rank.materialized_int8(source, 8)
        expected = torch.tensor([[127., 1., -1., 0., 0., 3., -3., 0.]], dtype=torch.bfloat16)
        self.assertEqual(actual.view(torch.uint16).tolist(), expected.view(torch.uint16).tolist())

    def test_zero_codes_reconstruct_positive_zero_like_integer_cache(self) -> None:
        grouped = torch.tensor([[[-0.1, 0.1, -0.0, 0.0]]])
        result = rank.rust_dequantize(grouped, torch.ones((1, 1)), -8, 7)
        self.assertEqual(result.to(torch.bfloat16).view(torch.uint16).tolist(), [[[0, 0, 0, 0]]])

    def test_reserve_is_derived_from_counts_and_per_corpus_absence(self) -> None:
        corpora = [torch.tensor([499, 500, 0, 0]), torch.tensor([1, 0, 800, 0])]
        counts = sum(corpora)
        self.assertEqual(rank.coverage_required(counts, corpora, 500).tolist(), [False, True, True, True])
        with self.assertRaisesRegex(RuntimeError, "counts differ"):
            rank.coverage_required(counts + 1, corpora, 500)

    def test_reserve_consumes_budget_without_invented_scores(self) -> None:
        rare = {"moe_layer": 0, "expert": 0, "int4_bytes": 100, "int8_bytes": 180,
                "coverage_required_int8": True}
        measured = [{"moe_layer": 0, "expert": expert, "int4_bytes": 100, "int8_bytes": 180,
                     "expected_rank": expert - 1, "expected_gain_units": gain}
                    for expert, gain in ((1, 50), (2, 0))]
        candidates = [rare, *measured]
        self.assertEqual(rank.select_with_reserve(candidates, "expected_rank", "expected_gain_units", 80, False), ({(0, 0)}, 80))
        self.assertEqual(rank.select_with_reserve(candidates, "expected_rank", "expected_gain_units", 240, False), ({(0, 0), (0, 1)}, 160))
        self.assertEqual(rank.select_with_reserve(candidates, "expected_rank", "expected_gain_units", 240, True), ({(0, 0), (0, 1), (0, 2)}, 240))
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            rank.select_with_reserve(candidates, "expected_rank", "expected_gain_units", 79, False)
        self.assertNotIn("expected_gain_units", rare)

    def test_quantization_uses_rust_ties_and_adjacent_float_values(self) -> None:
        half = torch.tensor(0.5, dtype=torch.float32)
        below = torch.nextafter(half, torch.tensor(0.0))
        above = torch.nextafter(half, torch.tensor(1.0))
        values = torch.tensor([-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5])
        expected = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        torch.testing.assert_close(rank.rust_round(values), expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(values.round(), expected))
        adjacent = torch.stack([-above, -half, -below, below, half, above])
        torch.testing.assert_close(rank.rust_round(adjacent),
                                   torch.tensor([-1., -1., 0., 0., 1., 1.]), rtol=0, atol=0)

    def test_int8_persisted_unit_scale_rounds_like_rust_cache(self) -> None:
        # max/127 = exactly1, so BF16 persistence cannot obscure tie cases.
        source = torch.tensor([[127.0, 0.5, -0.5, 2.5, -2.5, 1.5, -1.5, 0.0]])
        result = rank.quantize_int8(source, 8, torch.device("cpu")).float()
        torch.testing.assert_close(result, torch.tensor([[127., 1., -1., 3., -3., 2., -2., 0.]]),
                                   rtol=0, atol=0)

    def test_persisted_zero_scale_and_search_epsilon_match_rust(self) -> None:
        grouped = torch.tensor([[[0.5, -0.5]]])
        torch.testing.assert_close(rank.rust_dequantize(grouped, torch.zeros((1, 1)), -8, 7),
                                   torch.zeros_like(grouped), rtol=0, atol=0)
        tiny = torch.full((1, 8), 2.0 ** -30)
        result, _ = rank.quantize_search_rmse(tiny, torch.ones(8), 8, torch.device("cpu"))
        torch.testing.assert_close(result.float(), torch.zeros_like(tiny), rtol=0, atol=0)

    def test_expected_error_retains_measured_router_exposure(self) -> None:
        rare = rank.expected_exposure_gain(0.1, 0.01, 1.0, 100)
        frequent = rank.expected_exposure_gain(0.1, 0.01, 10.0, 100)
        self.assertAlmostEqual(frequent, rare * 10)
        self.assertAlmostEqual(frequent, rank.expected_exposure_gain(0.1, 0.01, 20.0, 200))
        self.assertEqual(rank.expected_exposure_gain(0.01, 0.1, 1.0, 100), 0.0)
        for args in [(float("nan"), 0.1, 1.0, 100), (0.1, 0.01, -1.0, 100),
                     (0.1, 0.01, 1.0, 0)]:
            with self.assertRaises(RuntimeError):
                rank.expected_exposure_gain(*args)

    def _write_kic1(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        inputs = np.arange(1, 7, dtype="<f4").reshape(2, 3)
        down = np.arange(7, 11, dtype="<f4").reshape(2, 2)
        input_bytes = inputs.tobytes()
        down_bytes = down.tobytes()
        payload = input_bytes + down_bytes
        manifest = {
            "schema_version": rank.KIC_VERSION,
            "format": "Krasis expert INT4 activation moments",
            "objective": "router_weight_squared_diagonal_activation_second_moment_v1",
            "hidden_size": 3,
            "intermediate_size": 2,
            "routed_experts": 2,
            "topk": 1,
            "routed_layers": 1,
            "calibration_sha256": "11" * 32,
            "capture_corpora_sha256": ["22" * 32],
            "capture_artifact_sha256": ["33" * 32],
            "payload_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "layers": [
                {
                    "model_layer": 3,
                    "input_second_moment_f32": {"offset": 0, "len": len(input_bytes)},
                    "down_second_moment_f32": {
                        "offset": len(input_bytes),
                        "len": len(down_bytes),
                    },
                    "route_count_min": 1,
                    "route_count_max": 2,
                }
            ],
        }
        manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        payload_offset = 4096
        header = rank.KIC_HEADER.pack(
            rank.KIC_MAGIC,
            rank.KIC_VERSION,
            len(manifest_bytes),
            payload_offset,
            len(payload),
            hashlib.sha256(manifest_bytes).digest(),
        )
        path.write_bytes(header + manifest_bytes + bytes(payload_offset - len(header) - len(manifest_bytes)) + payload)
        return inputs, down

    def test_observed_tail_statistics_are_exact_and_deterministic(self) -> None:
        values = torch.arange(1, 201, dtype=torch.float32)
        first = rank.observed_statistics(values)
        second = rank.observed_statistics(values.flip(0))
        self.assertEqual(first, second)
        self.assertEqual(first["maximum"], 200.0)
        self.assertEqual(first["p99"], 198.0)
        self.assertEqual(first["cvar99"], 199.0)

    def test_ratio_order_uses_exact_integer_cross_multiplication_and_identity_tie_break(self) -> None:
        left = {
            "expected_gain_units": 10,
            "int4_bytes": 100,
            "int8_bytes": 200,
            "moe_layer": 0,
            "expert": 1,
            "model_layer": 3,
        }
        right = {
            "expected_gain_units": 20,
            "int4_bytes": 100,
            "int8_bytes": 300,
            "moe_layer": 0,
            "expert": 2,
            "model_layer": 3,
        }
        self.assertLess(rank.compare_score(left, right, "expected_gain_units"), 0)
        right["expert"] = 0
        self.assertGreater(rank.compare_score(left, right, "expected_gain_units"), 0)

    def test_atomic_write_never_leaves_a_temporary_publication(self) -> None:
        with tempfile.TemporaryDirectory(prefix="krasis-mixed-rank-") as directory:
            path = Path(directory) / "layer.json"
            rank.atomic_write(path, b"first")
            self.assertEqual(path.read_bytes(), b"first")
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_unique_tensor_name_fails_closed_on_missing_or_ambiguous_source(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expected one tensor"):
            rank.unique_tensor_name({}, ".weight")
        with self.assertRaisesRegex(RuntimeError, "expected one tensor"):
            rank.unique_tensor_name({"a.weight": "a", "b.weight": "b"}, ".weight")

    def test_kic1_moments_are_verified_and_used_without_recomputation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="krasis-mixed-kic-") as directory:
            path = Path(directory) / "calibration.kic"
            expected_inputs, expected_down = self._write_kic1(path)
            manifest, raw, payload_offset = rank.load_kic1(path)
            inputs, down = rank.kic1_layer_moments(
                manifest, raw, payload_offset, 3, 2, 3, 2
            )
            torch.testing.assert_close(inputs, torch.from_numpy(expected_inputs))
            torch.testing.assert_close(down, torch.from_numpy(expected_down))

    def test_kic1_corrupt_payload_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="krasis-mixed-kic-") as directory:
            path = Path(directory) / "calibration.kic"
            self._write_kic1(path)
            payload = bytearray(path.read_bytes())
            payload[-1] ^= 1
            path.write_bytes(payload)
            with self.assertRaisesRegex(RuntimeError, "payload SHA-256 mismatch"):
                rank.load_kic1(path)

    def test_sensitivity_layer_resume_requires_complete_immutable_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="krasis-mixed-layer-") as directory:
            audit_path = Path(directory) / "scale-audit-layer-000.bin"
            audit_path.write_bytes(b"audit")
            identity = rank.sensitivity_layer_identity(
                {
                    "checkpoint_sha256": "11" * 32,
                    "capture_artifact_sha256": ["22" * 32],
                },
                3,
                0,
            )
            value = {
                "format": rank.SENSITIVITY_LAYER_FORMAT,
                "schema_version": rank.SENSITIVITY_LAYER_VERSION,
                "identity": identity,
                "identity_sha256": hashlib.sha256(
                    rank.canonical_bytes(identity)
                ).hexdigest(),
                "audit_basename": audit_path.name,
                "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                "memory": {"process_vmhwm_bytes": 1},
                "source_precisions": ["F8_E4M3"],
                "candidates": [
                    {"model_layer": 3, "moe_layer": 0, "expert": expert}
                    for expert in range(2)
                ],
            }
            rank.validate_sensitivity_layer_shard(value, audit_path, identity, 2)
            stale = rank.sensitivity_layer_identity(
                {
                    "checkpoint_sha256": "33" * 32,
                    "capture_artifact_sha256": ["22" * 32],
                },
                3,
                0,
            )
            with self.assertRaisesRegex(RuntimeError, "immutable identity mismatch"):
                rank.validate_sensitivity_layer_shard(value, audit_path, stale, 2)

    def test_sensitivity_layer_resume_rejects_duplicate_or_incomplete_experts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="krasis-mixed-layer-") as directory:
            audit_path = Path(directory) / "scale-audit-layer-000.bin"
            audit_path.write_bytes(b"audit")
            identity = rank.sensitivity_layer_identity(
                {"checkpoint_sha256": "11" * 32}, 3, 0
            )
            value = {
                "format": rank.SENSITIVITY_LAYER_FORMAT,
                "schema_version": rank.SENSITIVITY_LAYER_VERSION,
                "identity": identity,
                "identity_sha256": hashlib.sha256(
                    rank.canonical_bytes(identity)
                ).hexdigest(),
                "audit_basename": audit_path.name,
                "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                "memory": {},
                "source_precisions": ["F8_E4M3"],
                "candidates": [
                    {"model_layer": 3, "moe_layer": 0, "expert": 0},
                    {"model_layer": 3, "moe_layer": 0, "expert": 0},
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "incomplete or duplicated"):
                rank.validate_sensitivity_layer_shard(value, audit_path, identity, 2)


if __name__ == "__main__":
    unittest.main()
