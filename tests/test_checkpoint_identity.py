import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from krasis import server
from krasis.checkpoint_identity import (
    CHECKPOINT_IDENTITY_ENV,
    checkpoint_identity,
    route_checkpoint_identity,
)
from krasis.config import cache_dir_for_model


def _write_model(root: Path, payloads: dict[str, bytes], revision: str | None = None) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"model_type":"test"}\n', encoding="utf-8")
    weight_map = {f"tensor.{index}": name for index, name in enumerate(sorted(payloads))}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}, sort_keys=True), encoding="utf-8"
    )
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
        if revision is not None:
            metadata = root / ".cache" / "huggingface" / "download" / f"{name}.metadata"
            metadata.parent.mkdir(parents=True, exist_ok=True)
            import hashlib
            completed_at = (root / name).stat().st_mtime
            metadata.write_text(
                f"{revision}\n{hashlib.sha256(payload).hexdigest()}\n{completed_at}\n",
                encoding="utf-8",
            )
    return root


class CheckpointIdentityTest(unittest.TestCase):
    def setUp(self):
        checkpoint_identity.cache_clear()
        os.environ.pop(CHECKPOINT_IDENTITY_ENV, None)

    def tearDown(self):
        checkpoint_identity.cache_clear()
        os.environ.pop(CHECKPOINT_IDENTITY_ENV, None)

    def test_hf_and_content_hash_identity_match_for_the_same_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            payloads = {"model-1.safetensors": b"first", "model-2.safetensors": b"second"}
            hf = _write_model(base / "hf" / "same", payloads, "a" * 40)
            local = _write_model(base / "local" / "same", payloads)

            with mock.patch.dict(os.environ, {"HOME": str(base / "home")}, clear=False):
                hf_identity = checkpoint_identity(str(hf))
                local_identity = checkpoint_identity(str(local))
                route_identity = route_checkpoint_identity(str(hf))

        self.assertEqual(hf_identity["source_kind"], "huggingface_snapshot")
        self.assertEqual(local_identity["source_kind"], "content_sha256")
        self.assertEqual(hf_identity["sha256"], local_identity["sha256"])
        self.assertEqual(route_identity["sha256"], hf_identity["sha256"])

    def test_same_basename_different_weights_have_different_cache_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            first = _write_model(base / "a" / "duplicate", {"model.safetensors": b"aaaa"})
            second = _write_model(base / "b" / "duplicate", {"model.safetensors": b"bbbb"})
            with mock.patch.dict(os.environ, {"HOME": str(base / "home")}, clear=False):
                first_cache = cache_dir_for_model(str(first))
                second_cache = cache_dir_for_model(str(second))

        self.assertNotEqual(first_cache, second_cache)
        self.assertIn("duplicate--", first_cache)
        parts = Path(first_cache).name.rsplit("--", 1)
        self.assertEqual(len(parts[1]), 64)

    def test_config_change_keeps_checkpoint_identity_but_changes_cache_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            model = _write_model(base / "model", {"model.safetensors": b"payload"})
            with mock.patch.dict(os.environ, {"HOME": str(base / "home")}, clear=False):
                identity_before = checkpoint_identity(str(model))["sha256"]
                cache_before = cache_dir_for_model(str(model))
                (model / "config.json").write_text(
                    '{"model_type":"test","variant":2}\n', encoding="utf-8"
                )
                cache_after = cache_dir_for_model(str(model))
                identity_after = checkpoint_identity(str(model))["sha256"]

        self.assertEqual(identity_before, identity_after)
        self.assertNotEqual(cache_before, cache_after)

    def test_same_size_weight_change_changes_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            model = _write_model(base / "model", {"model.safetensors": b"aaaa"})
            with mock.patch.dict(os.environ, {"HOME": str(base / "home")}, clear=False):
                before = checkpoint_identity(str(model))["sha256"]
                (model / "model.safetensors").write_bytes(b"bbbb")
                changed_mtime = (model / "model.safetensors").stat().st_mtime + 5
                os.utime(model / "model.safetensors", (changed_mtime, changed_mtime))
                checkpoint_identity.cache_clear()
                after = checkpoint_identity(str(model))["sha256"]
        self.assertNotEqual(before, after)

    def test_python_registers_identity_for_rust_weight_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            model = _write_model(base / "model", {"model.safetensors": b"payload"})
            with mock.patch.dict(
                os.environ,
                {CHECKPOINT_IDENTITY_ENV: "{}", "HOME": str(base / "home")},
                clear=False,
            ):
                identity = checkpoint_identity(str(model))
                mapping = json.loads(os.environ[CHECKPOINT_IDENTITY_ENV])
        self.assertEqual(mapping[str(model.resolve())], identity["sha256"])

    def test_identity_and_cache_namespace_match_cross_language_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            model = _write_model(
                base / "model", {"model.safetensors": b"payload"}, "a" * 40
            )
            with mock.patch.dict(os.environ, {"HOME": str(base / "home")}, clear=False):
                identity = checkpoint_identity(str(model))
                cache = Path(cache_dir_for_model(str(model))).name
        self.assertEqual(
            identity["sha256"],
            "4daab4e4e3a31165b08ea5172f8baf9cb2fe2934ce2311be03b7f2c755b4c833",
        )
        self.assertEqual(
            cache,
            "model--f8b7c37d8aa31473fa24ca46e340feba6e3393aeb0b0026059f83cf3d5a65743",
        )

    def test_route_signature_contains_only_source_independent_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            model = _write_model(base / "model", {"model.safetensors": b"payload"}, "b" * 40)
            cfg = SimpleNamespace(
                model_type="test",
                num_hidden_layers=1,
                num_moe_layers=1,
                n_routed_experts=2,
                num_experts_per_tok=1,
                num_full_attention_layers=1,
                layer_types=None,
                scoring_func="softmax",
                routed_scaling_factor=1.0,
                norm_topk_prob=True,
                use_moe_router_bias=False,
                need_fp32_gate=False,
                norm_bias_one=False,
            )
            args = SimpleNamespace(model_path=str(model), dspark_mode="off")
            with mock.patch.dict(os.environ, {"HOME": str(base / "home")}, clear=False):
                signature = server._heatmap_route_signature_from_cfg(cfg, args)

        checkpoint = signature["model"]["checkpoint_identity"]
        self.assertEqual(checkpoint["format"], "krasis_checkpoint_identity")
        self.assertNotIn("revision", checkpoint)
        self.assertNotIn("source_kind", checkpoint)

    def test_strong_runtime_does_not_select_legacy_route_entry(self):
        legacy_signature = {"model": {"model_name": "same"}}
        strong_signature = {
            "model": {
                "model_name": "same",
                "checkpoint_identity": {
                    "format": "krasis_checkpoint_identity",
                    "format_version": 1,
                    "sha256": "c" * 64,
                    "weight_file_count": 1,
                    "weight_bytes": 7,
                },
            }
        }
        runtime = {"attention_quant": "hqq4"}
        legacy_hash = server._sha256_jsonable(legacy_signature)
        strong_hash = server._sha256_jsonable(strong_signature)
        runtime_hash = server._sha256_jsonable(runtime)
        manifest = {
            "artifacts": [
                {
                    "artifact_id": "legacy",
                    "status": "approved",
                    "priority": 10,
                    "route_signature_sha256": legacy_hash,
                    "validated_runtime_sha256s": [runtime_hash],
                },
                {
                    "artifact_id": "strong",
                    "status": "approved",
                    "priority": 10,
                    "route_signature_sha256": strong_hash,
                    "validated_runtime_sha256s": [runtime_hash],
                },
            ]
        }

        selected = server._select_approved_heatmap_manifest_entry(
            manifest, {"route_signature": strong_signature, "runtime_compat": runtime}
        )
        self.assertEqual(selected["artifact_id"], "strong")

        legacy_selected = server._select_approved_heatmap_manifest_entry(
            manifest, {"route_signature": legacy_signature, "runtime_compat": runtime}
        )
        self.assertEqual(legacy_selected["artifact_id"], "legacy")


if __name__ == "__main__":
    raise SystemExit("Run through ./dev checkpoint-identity-test")
