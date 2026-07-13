import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from krasis import server


class ApprovedHeatmapAutoTest(unittest.TestCase):
    def test_route_signature_accepts_missing_layer_types(self):
        cfg = SimpleNamespace(
            model_type="qwen3_moe",
            num_hidden_layers=94,
            num_moe_layers=94,
            n_routed_experts=128,
            num_experts_per_tok=8,
            num_full_attention_layers=94,
            layer_types=None,
        )
        model = SimpleNamespace(cfg=cfg)
        args = SimpleNamespace(model_path=".")

        signature = server._heatmap_route_signature(model, args)

        self.assertEqual(
            signature["routing"]["layer_types_sha256"],
            server._sha256_jsonable([]),
        )

    def test_manifest_selection_matches_route_and_runtime_hashes(self):
        expected = {
            "route_signature": {"model": "example", "top_k": 8},
            "runtime_compat": {"attention_quant": "hqq4", "kv_dtype": "k4v4"},
        }
        route_hash = server._sha256_jsonable(expected["route_signature"])
        runtime_hash = server._sha256_jsonable(expected["runtime_compat"])
        manifest = {
            "artifacts": [
                {
                    "artifact_id": "wrong-runtime",
                    "status": "approved",
                    "priority": 1,
                    "route_signature_sha256": route_hash,
                    "validated_runtime_sha256s": ["not-the-runtime"],
                },
                {
                    "artifact_id": "right",
                    "status": "approved",
                    "priority": 5,
                    "route_signature_sha256": route_hash,
                    "validated_runtime_sha256s": [runtime_hash],
                },
            ],
        }

        entry = server._select_approved_heatmap_manifest_entry(manifest, expected)

        self.assertIsNotNone(entry)
        self.assertEqual(entry["artifact_id"], "right")

    def test_manifest_selection_can_use_explicit_runtime_compatibility_policy(self):
        canonical_runtime = {
            "attention_quant": "hqq8",
            "kv_dtype": "k4v4",
            "hqq_auto_budget_pct": None,
            "hqq46_auto_budget_mib": None,
            "gpu_expert_bits": 4,
            "cpu_expert_bits": 4,
        }
        current_runtime = {
            "attention_quant": "hqq46_auto",
            "kv_dtype": "k6v6",
            "hqq_auto_budget_pct": 10.0,
            "hqq46_auto_budget_mib": None,
            "gpu_expert_bits": 4,
            "cpu_expert_bits": 4,
        }
        expected = {
            "route_signature": {"model": "qwen36", "top_k": 8},
            "runtime_compat": current_runtime,
        }
        route_hash = server._sha256_jsonable(expected["route_signature"])
        manifest = {
            "artifacts": [
                {
                    "artifact_id": "qwen36-canonical-hqq8",
                    "status": "approved",
                    "priority": 1,
                    "route_signature_sha256": route_hash,
                    "validated_runtime_sha256s": [
                        server._sha256_jsonable(canonical_runtime),
                    ],
                    "validated_compatible_runtimes": [canonical_runtime],
                    "runtime_compatibility": {
                        "ignored_runtime_fields": [
                            "attention_quant",
                            "kv_dtype",
                            "hqq_auto_budget_pct",
                            "hqq46_auto_budget_mib",
                        ],
                        "accepted_attention_quants": [
                            "hqq4",
                            "hqq46_auto",
                            "hqq6",
                            "hqq68_auto",
                            "hqq8",
                        ],
                        "accepted_kv_dtypes": ["k4v4", "k6v6", "bf16"],
                    },
                },
            ],
        }

        entry = server._select_approved_heatmap_manifest_entry(manifest, expected)

        self.assertIsNotNone(entry)
        self.assertEqual(entry["artifact_id"], "qwen36-canonical-hqq8")

    def test_manifest_selection_prefers_matching_hqq_and_ignores_kv(self):
        route_signature = {"model": "step37", "top_k": 8}
        expected = {
            "route_signature": route_signature,
            "runtime_compat": {
                "attention_quant": "hqq6",
                "kv_dtype": "k6v6",
                "hqq_auto_budget_pct": None,
                "hqq46_auto_budget_mib": None,
                "gpu_expert_bits": 4,
                "cpu_expert_bits": 4,
            },
        }
        route_hash = server._sha256_jsonable(route_signature)

        def entry_for(attention_quant):
            runtime = {
                "attention_quant": attention_quant,
                "kv_dtype": "k4v4",
                "hqq_auto_budget_pct": None,
                "hqq46_auto_budget_mib": None,
                "gpu_expert_bits": 4,
                "cpu_expert_bits": 4,
            }
            return {
                "artifact_id": f"step37-{attention_quant}",
                "status": "approved",
                "priority": 10,
                "route_signature_sha256": route_hash,
                "validated_runtime_sha256s": [server._sha256_jsonable(runtime)],
                "validated_compatible_runtimes": [runtime],
                "runtime_compatibility": {
                    "ignored_runtime_fields": ["kv_dtype"],
                    "accepted_attention_quants": [attention_quant],
                    "accepted_kv_dtypes": ["k4v4", "k6v6", "bf16"],
                },
            }

        manifest = {
            "artifacts": [
                entry_for("hqq4"),
                entry_for("hqq6"),
                entry_for("hqq8"),
            ],
        }

        entry = server._select_approved_heatmap_manifest_entry(manifest, expected)

        self.assertIsNotNone(entry)
        self.assertEqual(entry["artifact_id"], "step37-hqq6")

    def test_approved_heatmap_validation_accepts_manifest_runtime_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical_runtime = {
                "attention_quant": "hqq8",
                "kv_dtype": "k4v4",
                "hqq_auto_budget_pct": None,
                "hqq46_auto_budget_mib": None,
                "gpu_expert_bits": 4,
            }
            current_runtime = {
                "attention_quant": "hqq6",
                "kv_dtype": "bf16",
                "hqq_auto_budget_pct": None,
                "hqq46_auto_budget_mib": None,
                "gpu_expert_bits": 4,
            }
            route_signature = {"model": "qwen36", "top_k": 8}
            heatmap = {
                "0,1": 42,
                "_metadata": {
                    "format": server.APPROVED_HEATMAP_FORMAT,
                    "format_version": server.APPROVED_HEATMAP_FORMAT_VERSION,
                    "route_signature": route_signature,
                    "validated_compatible_runtimes": [canonical_runtime],
                },
            }
            heatmap_path = root / "heatmap.json"
            heatmap_path.write_text(json.dumps(heatmap), encoding="utf-8")

            loaded = server._load_validated_heatmap(
                str(heatmap_path),
                {
                    "route_signature": route_signature,
                    "runtime_compat": current_runtime,
                    "runtime_compat_policy": {
                        "ignored_runtime_fields": [
                            "attention_quant",
                            "kv_dtype",
                            "hqq_auto_budget_pct",
                            "hqq46_auto_budget_mib",
                        ],
                        "accepted_attention_quants": ["hqq4", "hqq6", "hqq8"],
                        "accepted_kv_dtypes": ["k4v4", "k6v6", "bf16"],
                        "reason": "test canonical reuse",
                    },
                },
            )

            self.assertEqual(loaded["0,1"], 42)

    def test_verified_cache_downloads_file_url_and_checks_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            payload = b'{"_metadata": {"format": "test"}}\n'
            source.write_bytes(payload)
            expected_sha = hashlib.sha256(payload).hexdigest()
            entry = {
                "artifact_id": "artifact",
                "filename": "approved.json",
                "download_url": source.resolve().as_uri(),
                "sha256": expected_sha,
                "bytes": len(payload),
            }

            cached = server._verified_cached_approved_heatmap(str(root / "cache"), entry)

            self.assertTrue(Path(cached).is_file())
            self.assertEqual(Path(cached).read_bytes(), payload)
            self.assertIn(expected_sha[:16], Path(cached).name)

    def test_auto_mode_falls_back_when_listed_artifact_cannot_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = {
                "route_signature": {"model": "example", "top_k": 8},
                "runtime_compat": {"attention_quant": "hqq4", "kv_dtype": "k4v4"},
            }
            route_hash = server._sha256_jsonable(expected["route_signature"])
            runtime_hash = server._sha256_jsonable(expected["runtime_compat"])
            missing_artifact = root / "missing.json"
            manifest = {
                "format": server.APPROVED_HEATMAP_MANIFEST_FORMAT,
                "format_version": server.APPROVED_HEATMAP_MANIFEST_FORMAT_VERSION,
                "artifacts": [
                    {
                        "artifact_id": "missing",
                        "status": "approved",
                        "priority": 1,
                        "filename": "missing.json",
                        "download_url": missing_artifact.resolve().as_uri(),
                        "sha256": "0" * 64,
                        "bytes": 1,
                        "route_signature_sha256": route_hash,
                        "validated_runtime_sha256s": [runtime_hash],
                    },
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            args = SimpleNamespace(
                approved_heatmap_mode="auto",
                approved_heatmap_manifest_url=manifest_path.resolve().as_uri(),
            )

            warnings = []
            old_warn = server._warn
            try:
                server._warn = warnings.append
                heatmap_path, heatmap_data = server._try_load_auto_approved_heatmap(
                    str(root / "cache"),
                    expected,
                    args,
                )
            finally:
                server._warn = old_warn

            self.assertIsNone(heatmap_path)
            self.assertIsNone(heatmap_data)
            self.assertTrue(any("falling back to quick startup heatmap" in msg for msg in warnings))


if __name__ == "__main__":
    unittest.main()
