import unittest
from types import SimpleNamespace

from krasis import server


class StartupCalibrationProbeTests(unittest.TestCase):
    def test_long_calibration_falls_back_to_bounded_initial_probe_without_estimate(self) -> None:
        next_target, reason = server._next_startup_calibration_probe_target(
            short_tokens=500,
            default_long_tokens=39_920,
            observed_prefill_mins=[(500, 7_763)],
            target_floor_mb=1_200,
        )

        self.assertEqual(next_target, 4_000)
        self.assertIn("initial adaptive", reason)

    def test_long_calibration_uses_model_estimate_for_first_probe(self) -> None:
        next_target, reason = server._next_startup_calibration_probe_target(
            short_tokens=500,
            default_long_tokens=39_920,
            observed_prefill_mins=[(500, 7_763)],
            target_floor_mb=1_200,
            estimated_prefill_mb_per_token=0.20,
        )

        self.assertEqual(next_target, 27_315)
        self.assertIn("model-estimated", reason)

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
