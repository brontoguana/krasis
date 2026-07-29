import unittest
from pathlib import Path

from tests.reference_test import timing_vram_safety_violation


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

        self.assertIn("let pressure_floor_mb = pending", section)
        self.assertIn("saturating_add(p.deficit_mb as usize)", section)
        self.assertIn("let soft_chunk_guard_mb =", section)
        self.assertIn(
            "let target_floor_mb = pressure_floor_mb.max("
            "safety_mb.saturating_add(soft_chunk_guard_mb));",
            section,
        )
        self.assertIn("while final_free_mb < target_floor_mb", section)
        self.assertIn("if final_free_mb >= target_floor_mb", section)
        self.assertNotIn("final_free_mb >= target_floor_mb || final_free_mb >= safety_mb", section)
        self.assertIn("target_floor=", section)

    def test_reference_runner_rejects_measured_low_below_margin(self) -> None:
        violation = timing_vram_safety_violation(
            {
                "safety_margin_mb": 600,
                "vram_low_water": [
                    {"device": 0, "min_free_mb": 552},
                    {"device": 1, "min_free_mb": 700},
                ],
            }
        )

        self.assertEqual(
            violation,
            {
                "device": 0,
                "min_free_mb": 552,
                "safety_margin_mb": 600,
                "deficit_mb": 48,
            },
        )

    def test_reference_runner_accepts_low_at_margin(self) -> None:
        self.assertIsNone(
            timing_vram_safety_violation(
                {
                    "safety_margin_mb": 600,
                    "vram_low_water": [{"device": 0, "min_free_mb": 600}],
                }
            )
        )

    def test_real_prefill_low_water_updates_native_runtime_reserve(self) -> None:
        source = (ROOT / "src/server.rs").read_text()
        call = "engine.update_measured_prefill_runtime_overhead_mb("

        self.assertGreaterEqual(source.count(call), 2)
        self.assertIn("engine.last_prepare_post_alloc_free_mb()", source)
        self.assertIn("crate::vram_monitor::current_request_lows()", source)

    def test_decode_timing_uses_copy_stream_events_for_graph_pcie_bandwidth(self) -> None:
        source = (ROOT / "src/gpu_decode.rs").read_text()

        self.assertNotIn("MoE is ~70% of total time", source)
        self.assertNotIn("Est PCIe BW:", source)
        self.assertIn("demand_dma: DecodeGraphEventSet", source)
        self.assertIn(".demand_dma\n                                    .record_start", source)
        self.assertIn(".demand_dma\n                                    .record_end", source)
        self.assertIn(
            "demand_dma_bytes as f64 / demand_dma_elapsed_s / 1e9",
            source,
        )
        self.assertIn("Demand H2D BW:", source)

    def test_gpu_route_split_launch_preserves_full_and_partitioned_weights(self) -> None:
        rust = (ROOT / "src/gpu_decode.rs").read_text()
        cuda = (ROOT / "src/cuda/decode_kernels.cu").read_text()

        self.assertNotIn(
            "KRASIS_SPLIT_EXPERT_LAUNCH is not supported together with "
            "KRASIS_GPU_ROUTE_SYNC\"",
            rust,
        )
        self.assertIn("let split_flag_idx = 2 + graph.max_experts_per_tok * 2;", rust)
        self.assertIn("batch_upload_ptrs_bytes + max_ept * 4 * 3", rust)
        self.assertIn("d_wts + (graph.max_experts_per_tok * 4) as u64", rust)
        self.assertIn(
            "d_upload_base + (ptr_stride * 4 + max_ept * 4 * 2) as u64",
            rust,
        )
        self.assertIn("hot_batch_count = topk - cold_count;", rust)
        self.assertIn("hot_batch_count = batch_count;", rust)
        self.assertIn(
            "KRASIS_SPLIT_EXPERT_LAUNCH is not supported for latent MoE layer",
            rust,
        )
        self.assertIn("if direct_w13_bf16 {", rust)
        self.assertIn("k.marlin_gemv_int4_v2_batched_bf16_out", rust)
        self.assertIn("k.relu2_w2_batched_coalesced.clone()", rust)
        self.assertIn("split_hot: DecodeGraphEventSet", rust)
        self.assertIn(".split_hot\n                        .record_start", rust)
        self.assertIn(".split_hot\n                        .record_end", rust)

        self.assertIn("float* split_weights_full = wts_arr + max_ept;", cuda)
        self.assertIn("float* split_weights_hot = split_weights_full + max_ept;", cuda)
        self.assertIn(
            "int split_expert_launch = mapped_cold_buf[2 + max_ept * 2] != 0;",
            cuda,
        )
        self.assertIn(
            "wts_arr[batch_count]  = split_expert_launch ? 0.0f : weight;",
            cuda,
        )
        self.assertIn("split_weights_full[batch_count] = weight;", cuda)
        self.assertIn("split_weights_hot[batch_count] = weight;", cuda)
        self.assertIn("split_weights_hot[batch_count] = 0.0f;", cuda)


if __name__ == "__main__":
    unittest.main()
