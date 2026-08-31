import unittest
from types import SimpleNamespace

from krasis import server


class StartupCalibrationProbeTests(unittest.TestCase):
    def test_portable_predicted_w1_probe_lengths_are_runtime_derived(self) -> None:
        self.assertEqual(
            server._portable_predicted_w1_probe_lengths(500, 39_920),
            [500, 4_467, 39_920],
        )

    def test_portable_predicted_w1_threshold_starts_at_first_measured_win(self) -> None:
        threshold, rows = server._select_portable_predicted_w1_threshold(
            [
                (500, [1.00, 1.01], [1.20, 1.21]),
                (5_000, [10.0, 10.1], [8.0, 8.1]),
                (40_000, [80.0, 80.5], [60.0, 61.0]),
            ]
        )

        self.assertEqual([row["verdict"] for row in rows], ["loss", "win", "win"])
        self.assertEqual(threshold, 5_000)

    def test_portable_predicted_w1_disables_when_no_probe_robustly_wins(self) -> None:
        threshold, rows = server._select_portable_predicted_w1_threshold(
            [
                (500, [1.00, 1.10], [1.05, 1.15]),
                (5_000, [10.0, 10.1], [11.0, 11.1]),
            ]
        )

        self.assertIsNone(threshold)
        self.assertEqual([row["verdict"] for row in rows], ["inconclusive", "loss"])

    def test_portable_predicted_w1_refuses_single_winning_extrapolation(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "only one robust winning probe"):
            server._select_portable_predicted_w1_threshold(
                [
                    (500, [1.0, 1.1], [1.2, 1.3]),
                    (5_000, [10.0, 10.1], [8.0, 8.1]),
                ]
            )

    def test_portable_predicted_w1_refuses_non_monotonic_win_region(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "non-monotonic"):
            server._select_portable_predicted_w1_threshold(
                [
                    (500, [1.0, 1.1], [0.8, 0.9]),
                    (5_000, [10.0, 10.1], [11.0, 11.1]),
                    (40_000, [80.0, 80.5], [60.0, 61.0]),
                ]
            )

    def test_long_calibration_falls_back_to_bounded_initial_probe_without_estimate(self) -> None:
        next_target, reason = server._next_startup_calibration_probe_target(
            short_tokens=500,
            default_long_tokens=39_920,
            observed_prefill_mins=[(500, 7_763)],
            target_floor_mb=1_200,
        )

        self.assertEqual(next_target, 4_000)
        self.assertIn("initial adaptive", reason)

    def test_long_calibration_model_estimate_cannot_skip_interior_probe(self) -> None:
        next_target, reason = server._next_startup_calibration_probe_target(
            short_tokens=500,
            default_long_tokens=39_920,
            observed_prefill_mins=[(500, 7_763)],
            target_floor_mb=1_200,
            estimated_prefill_mb_per_token=0.20,
        )

        self.assertEqual(next_target, 4_000)
        self.assertIn("bounded model-estimated", reason)

    def test_long_calibration_model_estimate_remains_bounded_after_first_probe(self) -> None:
        next_target, reason = server._next_startup_calibration_probe_target(
            short_tokens=500,
            default_long_tokens=39_920,
            observed_prefill_mins=[(500, 42_000), (4_000, 41_000)],
            target_floor_mb=1_200,
            estimated_prefill_mb_per_token=0.20,
        )

        self.assertEqual(next_target, 8_000)
        self.assertIn("model estimate", reason)

    def test_long_calibration_does_not_grow_when_short_probe_is_near_floor(self) -> None:
        next_target, reason = server._next_startup_calibration_probe_target(
            short_tokens=500,
            default_long_tokens=39_920,
            observed_prefill_mins=[(500, 1_650)],
            target_floor_mb=1_200,
        )

        self.assertIsNone(next_target)
        self.assertIn("short-probe headroom", reason)

    def test_long_calibration_does_not_fallback_when_estimate_has_no_reserve(self) -> None:
        next_target, reason = server._next_startup_calibration_probe_target(
            short_tokens=500,
            default_long_tokens=39_920,
            observed_prefill_mins=[(500, 2_300)],
            target_floor_mb=1_200,
            estimated_prefill_mb_per_token=0.20,
        )

        self.assertIsNone(next_target)
        self.assertIn("validation reserve", reason)

    def test_long_calibration_uses_runtime_derived_fail_closed_probe_near_guard(self) -> None:
        next_target, reason = server._next_startup_calibration_probe_target(
            short_tokens=500,
            default_long_tokens=39_920,
            observed_prefill_mins=[(500, 1_152)],
            target_floor_mb=1_200,
            estimated_prefill_mb_per_token=2.7,
            fail_closed_probe_tokens=4_000,
            runtime_safety_floor_mb=600,
        )

        self.assertEqual(next_target, 4_000)
        self.assertIn("runtime-derived fail-closed", reason)

    def test_long_calibration_rejects_fail_closed_probe_below_runtime_floor(self) -> None:
        next_target, reason = server._next_startup_calibration_probe_target(
            short_tokens=500,
            default_long_tokens=39_920,
            observed_prefill_mins=[(500, 599)],
            target_floor_mb=1_200,
            estimated_prefill_mb_per_token=2.7,
            fail_closed_probe_tokens=4_000,
            runtime_safety_floor_mb=600,
        )

        self.assertIsNone(next_target)
        self.assertIn("adaptive floor", reason)

    def test_long_calibration_continues_fail_closed_from_measured_long_probe(self) -> None:
        next_target, reason = server._next_startup_calibration_probe_target(
            short_tokens=500,
            default_long_tokens=39_920,
            observed_prefill_mins=[(500, 1_140), (4_000, 1_188)],
            target_floor_mb=1_200,
            estimated_prefill_mb_per_token=2.7,
            fail_closed_probe_tokens=8_000,
            runtime_safety_floor_mb=600,
        )

        self.assertEqual(next_target, 8_000)
        self.assertIn("runtime-derived fail-closed", reason)

    def test_long_calibration_stops_when_validation_floor_is_too_close(self) -> None:
        next_target, reason = server._next_startup_calibration_probe_target(
            short_tokens=500,
            default_long_tokens=39_920,
            observed_prefill_mins=[
                (500, 7_763),
                (4_000, 7_000),
                (8_000, 5_500),
                (16_000, 2_050),
                (20_000, 1_300),
            ],
            target_floor_mb=1_200,
        )

        self.assertIsNone(next_target)
        self.assertIn("validation reserve", reason)

    def test_long_calibration_uses_configured_safety_as_floor_source(self) -> None:
        self.assertEqual(server._startup_calibration_long_floor_mb(600), 1_200)
        self.assertEqual(server._startup_calibration_long_floor_mb(500), 1_000)

    def test_startup_vram_floor_is_fail_closed(self) -> None:
        server._require_startup_vram_floor("probe", 600, 600)
        with self.assertRaisesRegex(RuntimeError, "min_free=599 MB safety=600 MB"):
            server._require_startup_vram_floor("probe", 599, 600)

    def test_compact_kv_stage_exact_estimate_uses_model_dimensions(self) -> None:
        cfg = SimpleNamespace(
            is_gqa=True,
            num_hidden_layers=40,
            num_full_attention_layers=20,
            num_key_value_heads=8,
            gqa_head_dim=128,
        )
        model = SimpleNamespace(cfg=cfg)

        mb_per_token = server._startup_stage_exact_kv_mb_per_token(model, "k6v6")

        self.assertAlmostEqual(mb_per_token, (2 * 20 * 8 * 128) / (1024 * 1024))

    def test_prefill_growth_estimate_combines_scratch_and_stage_exact_kv(self) -> None:
        cfg = SimpleNamespace(
            is_gqa=True,
            num_hidden_layers=40,
            num_full_attention_layers=20,
            num_key_value_heads=8,
            gqa_head_dim=128,
        )
        model = SimpleNamespace(cfg=cfg)

        class Store:
            def prefill_scratch_reservation_mb(self, tokens: int) -> int:
                return 1_000 + tokens // 10

        estimate = server._startup_calibration_estimated_prefill_mb_per_token(
            model,
            Store(),
            "k4v4",
            500,
            10_500,
        )

        self.assertAlmostEqual(
            estimate,
            0.1 + (2 * 20 * 8 * 128) / (1024 * 1024),
        )


if __name__ == "__main__":
    unittest.main()
