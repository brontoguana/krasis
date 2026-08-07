import unittest

from krasis.multi_gpu_planner import DeviceServiceProfile, optimize_contiguous_splits


def _profile(gpu, h2d_gbps, d2d_gbps, tail=1.01):
    probe = 16 * 1024 * 1024
    h2d_us = probe / (h2d_gbps * 1e9) * 1e6
    d2d_us = probe / (d2d_gbps * 1e9) * 1e6
    return DeviceServiceProfile(
        gpu_index=gpu,
        probe_bytes=probe,
        d2d_probe_bytes=probe,
        h2d_seconds_per_byte=1.0 / (h2d_gbps * 1e9),
        d2d_seconds_per_byte=1.0 / (d2d_gbps * 1e9),
        h2d_p50_us=h2d_us,
        h2d_p95_us=h2d_us * tail,
        d2d_p50_us=d2d_us,
        d2d_p95_us=d2d_us * tail,
    )


class MultiGpuPlannerTest(unittest.TestCase):
    def setUp(self):
        self.layers = [64 * 1024 * 1024] * 48
        self.moe = [True] * 48
        self.expert_bytes = 4 * 1024 * 1024

    def _budget(self, _gpu, start, end):
        return (end - start) * 64 * self.expert_bytes

    def test_matched_cards_preserve_existing_split(self):
        plan = optimize_contiguous_splits(
            preferred_splits=[24],
            layer_resident_bytes=self.layers,
            layer_is_moe=self.moe,
            profiles=[_profile(0, 25, 900), _profile(1, 25, 900)],
            hcs_budget_bytes=self._budget,
            expert_bytes=self.expert_bytes,
            experts_per_layer=64,
            experts_per_token=8,
            terminal_bytes=512 * 1024 * 1024,
        )
        self.assertEqual(plan.splits, (24,))
        self.assertFalse(plan.admitted)

    def test_mismatched_card_receives_fewer_serial_layers(self):
        plan = optimize_contiguous_splits(
            preferred_splits=[24],
            layer_resident_bytes=self.layers,
            layer_is_moe=self.moe,
            profiles=[_profile(0, 25, 1800), _profile(1, 25, 450)],
            hcs_budget_bytes=self._budget,
            expert_bytes=self.expert_bytes,
            experts_per_layer=64,
            experts_per_token=8,
            terminal_bytes=0,
        )
        self.assertTrue(plan.admitted)
        self.assertGreater(plan.splits[0], 24)
        self.assertLess(plan.predicted_seconds_per_token, plan.preferred_seconds_per_token)

    def test_small_predicted_change_is_rejected_inside_measured_tail(self):
        plan = optimize_contiguous_splits(
            preferred_splits=[24],
            layer_resident_bytes=self.layers,
            layer_is_moe=self.moe,
            profiles=[_profile(0, 25, 900, tail=1.10), _profile(1, 25, 850, tail=1.10)],
            hcs_budget_bytes=self._budget,
            expert_bytes=self.expert_bytes,
            experts_per_layer=64,
            experts_per_token=8,
            terminal_bytes=0,
        )
        self.assertEqual(plan.splits, (24,))
        self.assertFalse(plan.admitted)

    def test_three_device_plan_is_contiguous_and_respects_minimum(self):
        plan = optimize_contiguous_splits(
            preferred_splits=[16, 32],
            layer_resident_bytes=self.layers,
            layer_is_moe=self.moe,
            profiles=[_profile(0, 25, 1200), _profile(1, 25, 900), _profile(2, 25, 600)],
            hcs_budget_bytes=self._budget,
            expert_bytes=self.expert_bytes,
            experts_per_layer=64,
            experts_per_token=8,
            terminal_bytes=0,
        )
        self.assertEqual(len(plan.splits), 2)
        boundaries = (0, *plan.splits, len(self.layers))
        self.assertTrue(all(b - a >= 2 for a, b in zip(boundaries, boundaries[1:])))


if __name__ == "__main__":
    unittest.main()
