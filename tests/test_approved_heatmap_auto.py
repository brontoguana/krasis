import hashlib
import tempfile
import unittest
from pathlib import Path

from krasis import server


class ApprovedHeatmapAutoTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
