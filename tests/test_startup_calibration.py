import unittest

from krasis import server


class StartupCalibrationProbeTests(unittest.TestCase):
    def test_long_calibration_starts_from_short_probe_not_context_cap(self) -> None:
        next_target, reason = server._next_startup_calibration_probe_target(
            short_tokens=500,
            default_long_tokens=39_920,
            observed_prefill_mins=[(500, 7_763)],
            target_floor_mb=1_200,
        )

        self.assertEqual(next_target, 4_000)
        self.assertIn("initial adaptive", reason)

    def test_long_calibration_does_not_grow_when_short_probe_is_near_floor(self) -> None:
        next_target, reason = server._next_startup_calibration_probe_target(
            short_tokens=500,
            default_long_tokens=39_920,
            observed_prefill_mins=[(500, 1_650)],
            target_floor_mb=1_200,
        )

        self.assertIsNone(next_target)
        self.assertIn("short-probe headroom", reason)

    def test_long_calibration_stops_when_predicted_floor_is_too_close(self) -> None:
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
        self.assertIn("too close", reason)

    def test_long_calibration_uses_configured_safety_as_floor_source(self) -> None:
        self.assertEqual(server._startup_calibration_long_floor_mb(600), 1_200)
        self.assertEqual(server._startup_calibration_long_floor_mb(500), 1_000)


if __name__ == "__main__":
    unittest.main()
