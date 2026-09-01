"""No-model contract tests for the Rust Manager launcher integration."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from krasis import launcher


class ManagerLauncherTests(unittest.TestCase):
    def test_validate_only_is_hidden_but_parseable(self):
        with mock.patch.object(
            sys,
            "argv",
            ["krasis", "--non-interactive", "--validate-only"],
        ):
            args = launcher.parse_args()
        self.assertTrue(args.validate_only)

    def test_manager_config_schema_is_complete_and_uuid_based(self):
        cfg = launcher.LauncherConfig()
        cfg.model_path = "/models/example"
        cfg.selected_gpu_specs = ["0"]
        fake = SimpleNamespace(
            cfg=cfg,
            selected_gpus=[{"index": 0, "uuid": "GPU-stable"}],
        )
        result = launcher._manager_config_dict(fake)
        self.assertEqual(result["gpu_uuids"], ["GPU-stable"])
        self.assertEqual(result["vram_safety_margin_mb"], 600)
        self.assertEqual(result["attention_quant"], "hqq6")
        self.assertEqual(result["vision_quant"], "int4")
        self.assertEqual(result["kv_dtype"], "k6v6")
        self.assertNotIn("gpu_expert_bits", result)
        self.assertNotIn("cpu_expert_bits", result)
        expected = {
            "model_path", "gpu_uuids", "host", "port", "attention_quant",
            "vision_quant",
            "hqq_cache_profile", "hqq_group_size", "hqq_auto_budget_pct",
            "hqq_sidecar_manifest", "kv_dtype", "kv_cache_mb",
            "max_context_tokens", "vram_safety_margin_mb", "layer_group_size",
            "expert_group_size", "gpu_expert_int4_calib", "shared_expert_quant",
            "dense_mlp_quant", "lm_head_quant", "krasis_threads", "hcs",
            "dynamic_hcs", "dynamic_hcs_tail_blocks", "hcs_host_cache_mode",
            "multi_gpu_mode", "dynamic_peer", "adaptive_cold_mass_pruning",
            "prefix_cache", "prefix_cache_ram_fraction", "enable_thinking",
            "gpu_prefill_threshold", "pp_partition", "heatmap_path", "gguf_path",
            "expert_compression", "expert_compression_sidecar",
            "expert_compression_pipeline", "dspark_mode", "ssh_tunnel",
            "ssh_key_path", "force_rebuild_cache", "force_rebuild_hqq_cache",
        }
        self.assertEqual(set(result), expected)

    def test_manager_config_keeps_mixed_attention_budget_separate_from_kv(self):
        cfg = launcher.LauncherConfig()
        cfg.model_path = "/models/example"
        cfg.attention_quant = "hqq46_auto"
        cfg.hqq_auto_budget_pct = 15.0
        cfg.kv_dtype = "k6v6"
        fake = SimpleNamespace(
            cfg=cfg,
            selected_gpus=[{"index": 0, "uuid": "GPU-stable"}],
        )
        result = launcher._manager_config_dict(fake)
        self.assertEqual(result["attention_quant"], "hqq46_auto")
        self.assertEqual(result["hqq_auto_budget_pct"], 15.0)
        self.assertEqual(result["kv_dtype"], "k6v6")
        self.assertNotIn(":", result["attention_quant"])

    def test_manager_cli_rejects_invalid_port_before_rust_start(self):
        with self.assertRaises(SystemExit) as caught:
            launcher._manager_main(["--port", "0", "--no-open"])
        self.assertEqual(caught.exception.code, 2)

    def test_manager_cli_propagates_explicit_lan_mode(self):
        with mock.patch("krasis.krasis.run_manager") as run_manager:
            launcher._manager_main(["--port", "8080", "--lan", "--no-open"])
        run_manager.assert_called_once_with(sys.executable, 8080, False, True)

    def test_manager_preload_budget_rejects_vram_and_ram_overcommit(self):
        vram = SimpleNamespace(
            budget_error=None,
            _compute_budget=lambda: {
                "over_budget": True,
                "worst_rank": 0,
                "ranks": [{"total_mb": 24576.0}],
                "gpu_vram_mb": 16384.0,
                "ram_total_mb": 1024.0,
                "total_ram_gb": 64.0,
            },
        )
        with self.assertRaisesRegex(ValueError, "permanent VRAM"):
            launcher._validate_manager_preload_budget(vram)

        host = SimpleNamespace(
            budget_error=None,
            _compute_budget=lambda: {
                "over_budget": False,
                "worst_rank": 0,
                "ranks": [{"total_mb": 8192.0}],
                "gpu_vram_mb": 16384.0,
                "ram_total_mb": 131072.0,
                "total_ram_gb": 64.0,
            },
        )
        with self.assertRaisesRegex(ValueError, "host cache"):
            launcher._validate_manager_preload_budget(host)


if __name__ == "__main__":
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit("Run this test through ./dev manager-test")
    unittest.main()
