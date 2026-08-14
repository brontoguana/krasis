import unittest

from krasis.multi_gpu_planner import (
    DeviceServiceProfile,
    optimize_contiguous_splits,
    peer_plan_is_admissible,
    predict_peer_expert_plan,
)


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

    def test_peer_plan_uses_disjoint_heat_ranked_residents_and_overlap(self):
        profile = _profile(0, 20, 1_000)
        counts = {
            "0,0": 100,
            "0,1": 80,
            "0,2": 60,
            "0,3": 40,
        }
        plan = predict_peer_expert_plan(
            heatmap_counts=counts,
            total_decode_tokens=100,
            ranking=[(0, 0), (0, 1), (0, 2), (0, 3)],
            primary_capacity_experts=1,
            peer_capacity_experts=2,
            layer_resident_bytes=[16 * 1024],
            layer_is_moe=[True],
            primary_profile=profile,
            expert_bytes=self.expert_bytes,
            service_p95_us_by_routes=[20.0, 35.0, 50.0, 65.0],
            rtt_p95_us=25.0,
            terminal_bytes=0,
        )
        self.assertEqual(plan.primary_residents, ((0, 0),))
        self.assertEqual(plan.peer_residents, ((0, 1), (0, 2)))
        self.assertAlmostEqual(plan.cold_routes_before_per_token, 1.8)
        self.assertAlmostEqual(plan.captured_routes_per_token, 1.4)
        self.assertAlmostEqual(plan.cold_routes_after_per_token, 0.4)
        self.assertLess(
            plan.predicted_seconds_per_token,
            plan.predicted_primary_only_seconds_per_token,
        )

    def test_peer_plan_rejects_heatmap_without_runtime_denominator(self):
        with self.assertRaisesRegex(ValueError, "total_decode_tokens"):
            predict_peer_expert_plan(
                heatmap_counts={"0,0": 1},
                total_decode_tokens=0,
                ranking=[(0, 0)],
                primary_capacity_experts=0,
                peer_capacity_experts=1,
                layer_resident_bytes=[1],
                layer_is_moe=[True],
                primary_profile=_profile(0, 20, 1_000),
                expert_bytes=1,
                service_p95_us_by_routes=[1.0],
                rtt_p95_us=1.0,
                terminal_bytes=0,
            )

    def test_peer_admission_rejects_empty_disjoint_tier(self):
        profile = _profile(0, 20, 1_000)
        plan = predict_peer_expert_plan(
            heatmap_counts={"0,0": 100, "0,1": 80},
            total_decode_tokens=100,
            ranking=[(0, 0), (0, 1)],
            primary_capacity_experts=2,
            peer_capacity_experts=2,
            layer_resident_bytes=[16 * 1024],
            layer_is_moe=[True],
            primary_profile=profile,
            expert_bytes=self.expert_bytes,
            service_p95_us_by_routes=[20.0, 35.0],
            rtt_p95_us=25.0,
            terminal_bytes=0,
        )
        self.assertEqual(plan.peer_residents, ())
        self.assertEqual(plan.captured_routes_per_token, 0.0)
        self.assertFalse(
            peer_plan_is_admissible(
                peer_plan=plan,
                layer_split_seconds_per_token=(
                    plan.predicted_seconds_per_token * 2.0
                ),
                uncertainty_seconds=0.0,
                admitted_route_counts=[True, True],
            )
        )

    def test_peer_admission_accepts_faster_heat_capturing_tier(self):
        profile = _profile(0, 20, 1_000)
        plan = predict_peer_expert_plan(
            heatmap_counts={"0,0": 100, "0,1": 80, "0,2": 60},
            total_decode_tokens=100,
            ranking=[(0, 0), (0, 1), (0, 2)],
            primary_capacity_experts=1,
            peer_capacity_experts=2,
            layer_resident_bytes=[16 * 1024],
            layer_is_moe=[True],
            primary_profile=profile,
            expert_bytes=self.expert_bytes,
            service_p95_us_by_routes=[20.0, 35.0, 50.0],
            rtt_p95_us=25.0,
            terminal_bytes=0,
        )
        self.assertTrue(plan.peer_residents)
        self.assertGreater(plan.captured_routes_per_token, 0.0)
        self.assertTrue(
            peer_plan_is_admissible(
                peer_plan=plan,
                layer_split_seconds_per_token=(
                    plan.predicted_seconds_per_token * 2.0
                ),
                uncertainty_seconds=0.0,
                admitted_route_counts=[True, True, True],
            )
        )


if __name__ == "__main__":
    unittest.main()
