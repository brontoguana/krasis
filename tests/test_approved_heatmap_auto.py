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
