import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class VramPressureSourceTests(unittest.TestCase):
    def test_monitor_latches_below_safety_pressure_until_hcs_drain(self) -> None:
        source = (ROOT / "src/vram_monitor.rs").read_text()
        section = source[
            source.index("if warn_enabled.load(Ordering::Relaxed) {"):
            source.index("let prev_min = dev.min_free_bytes.load(Ordering::Relaxed);")
        ]

        self.assertIn("mark_pressure(dev.device_id", section)
        self.assertNotIn("clear_pressure(dev.device_id)", section)
        self.assertIn("Do not clear pressure just because idle free recovered", section)

    def test_hcs_drain_uses_recorded_pressure_deficit_as_idle_headroom(self) -> None:
        source = (ROOT / "src/gpu_decode.rs").read_text()
        section = source[
            source.index("pub fn hcs_drain_vram_pressure"):
            source.index("/// Reload soft-tier HCS experts after prefill completes.")
        ]

        self.assertIn("let target_floor_mb = pending", section)
        self.assertIn("saturating_add(p.deficit_mb as usize)", section)
        self.assertIn("while final_free_mb < target_floor_mb", section)
        self.assertIn("if final_free_mb >= target_floor_mb", section)
        self.assertNotIn("final_free_mb >= target_floor_mb || final_free_mb >= safety_mb", section)
        self.assertIn("target_floor=", section)


if __name__ == "__main__":
    unittest.main()
