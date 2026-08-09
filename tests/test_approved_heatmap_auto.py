import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from krasis import server


class ApprovedHeatmapAutoTest(unittest.TestCase):
    def test_peer_format_eligibility_accepts_only_production_int4(self):
        self.assertIsNone(server._peer_expert_format_error(4))
        self.assertEqual(
            server._peer_expert_format_error(8),
            "peer expert serving currently requires production INT4 experts",
        )
        self.assertEqual(
            server._peer_expert_format_error(3),
            "peer expert serving currently requires production INT4 experts",
        )

    def test_quick_heatmap_persists_measured_decode_token_denominator(self):
        class FakeStore:
            def hcs_init_collection(self, *args):
                return None

            def hcs_start_collecting(self):
                return None

            def rust_prefill_tokens(self, *args, **kwargs):
                return 7, 2, False

            def gpu_generate_batch(self, **kwargs):
                return [8, 9]

            def hcs_export_heatmap(self):
                return {"0,0": 3}

            def hcs_reset(self):
                return None

        model = SimpleNamespace(
            cfg=SimpleNamespace(
                num_hidden_layers=1,
                n_routed_experts=1,
                eos_token_id=99,
                extra_stop_token_ids=[],
            ),
            _gpu_decode_store=FakeStore(),
            server_cleanup=lambda: None,
        )
        args = SimpleNamespace()
        metadata = {
            "heatmap_build": {
                "decode_params": {
                    "temperature": 0.0,
                    "top_k": 0,
                    "top_p": 1.0,
                    "presence_penalty": 0.0,
                    "enable_thinking": False,
                    "mode": "benchmark",
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "quick.json"
            old_load = server._load_heatmap_prompts
            old_assert = server._assert_heatmap_prompts_are_held_out
            old_chat = server._chat_prompt_tokens
            old_meta = server._expected_heatmap_metadata
            try:
                server._load_heatmap_prompts = lambda path=None: ["one"]
                server._assert_heatmap_prompts_are_held_out = lambda prompts: None
                server._chat_prompt_tokens = lambda *args, **kwargs: [1, 2]
                server._expected_heatmap_metadata = lambda *args, **kwargs: metadata
                server._build_heatmap(model, str(out), args)
            finally:
                server._load_heatmap_prompts = old_load
                server._assert_heatmap_prompts_are_held_out = old_assert
                server._chat_prompt_tokens = old_chat
                server._expected_heatmap_metadata = old_meta

            artifact = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(
                artifact["_metadata"]["heatmap_build"]["total_decode_tokens"],
                3,
            )

    def test_quick_heatmap_validation_accepts_positive_measured_denominator(self):
        expected = {
            "format": server.HEATMAP_FORMAT,
            "format_version": server.HEATMAP_FORMAT_VERSION,
            "heatmap_build": {"prompt_count": 1},
        }
        actual = {
            **expected,
            "heatmap_build": {
                **expected["heatmap_build"],
                "total_decode_tokens": 257,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quick.json"
            path.write_text(
                json.dumps({"0,0": 1, "_metadata": actual}),
                encoding="utf-8",
            )
            loaded = server._load_validated_heatmap(str(path), expected)

        self.assertEqual(
            loaded["_metadata"]["heatmap_build"]["total_decode_tokens"],
            257,
        )

    def test_quick_heatmap_validation_rejects_invalid_measured_denominator(self):
        expected = {
            "format": server.HEATMAP_FORMAT,
            "format_version": server.HEATMAP_FORMAT_VERSION,
            "heatmap_build": {"prompt_count": 1},
        }
        actual = {
            **expected,
            "heatmap_build": {
                **expected["heatmap_build"],
                "total_decode_tokens": 0,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quick.json"
            path.write_text(
                json.dumps({"0,0": 1, "_metadata": actual}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "positive measured integer"):
                server._load_validated_heatmap(str(path), expected)

    def test_merge_heatmap_counts_keeps_cumulative_intervals_separate(self):
        cumulative = {"0,1": 3}

        merged = server._merge_heatmap_counts(
            cumulative,
            {"0,1": 4, "1,2": 5},
            "test interval",
        )

        self.assertEqual(merged, 9)
        self.assertEqual(cumulative, {"0,1": 7, "1,2": 5})

    def test_full_heatmap_ranking_orders_counts_then_fills_all_model_routes(self):
        model = SimpleNamespace(
            cfg=SimpleNamespace(n_routed_experts=3),
            layers=[
                SimpleNamespace(is_moe=True),
                SimpleNamespace(is_moe=False),
                SimpleNamespace(is_moe=True),
            ],
        )

        ranking = server._full_heatmap_ranking(
            model,
            {"2,1": 4, "0,2": 9, "0,0": 4},
        )

        self.assertEqual(ranking[:3], [(0, 2), (0, 0), (2, 1)])
        self.assertEqual(len(ranking), 6)
        self.assertEqual(set(ranking), {
            (0, 0), (0, 1), (0, 2),
            (2, 0), (2, 1), (2, 2),
        })

    def test_approved_builder_runs_only_first_fresh_prompt_without_residency(self):
        class FakeStore:
            def __init__(self):
                self.intervals = [{"0,1": 10}, {"0,0": 20}]
                self.pool_rankings = []
                self.reload_calls = []
                self.collection_inits = 0

            def set_vram_calibration(self, *args):
                return "calibrated"

            def hcs_init_collection(self, *args):
                self.collection_inits += 1

            def hcs_start_collecting(self):
                return None

            def hcs_export_heatmap(self):
                return self.intervals.pop(0)

            def rust_prefill_tokens(self, *args, **kwargs):
                return 7, 12, False

            def gpu_generate_batch(self, **kwargs):
                return [8, 9]

            def hcs_reset(self):
                return None

            def hcs_pool_init_tiered(self, ranking, **kwargs):
                self.pool_rankings.append(list(ranking))
                return "pool ready"

            def py_hcs_drain_vram_pressure(self, *args):
                return 0, 0.0, 700

            def py_hcs_reload_after_prefill(self, prompt_len):
                self.reload_calls.append(prompt_len)
                return 2, 1.0

        with tempfile.TemporaryDirectory() as tmp:
            store = FakeStore()
            model = SimpleNamespace(
                cfg=SimpleNamespace(
                    n_routed_experts=2,
                    num_hidden_layers=1,
                    eos_token_id=99,
                    extra_stop_token_ids=[],
                ),
                layers=[SimpleNamespace(is_moe=True)],
                _gpu_decode_store=store,
                server_cleanup=lambda: None,
            )
            args = SimpleNamespace(
                approved_heatmap_prompts=None,
                approved_heatmap_max_prompts=0,
                approved_heatmap_decode_tokens=2,
                approved_heatmap_checkpoint_every=0,
                approved_heatmap_residency_refresh_every=1,
                approved_heatmap_resume_from=None,
                approved_heatmap_bootstrap_from=None,
                heatmap_path=None,
                benchmark=False,
                benchmark_only=False,
                temperature=0.0,
                enable_thinking=False,
                residency_calibration={
                    "short_tokens": 8,
                    "long_tokens": 16,
                    "prefill_short_free_mb": 1000,
                    "prefill_long_free_mb": 900,
                    "decode_short_free_mb": 1100,
                    "decode_long_free_mb": 1000,
                    "baseline_free_mb": 1200,
                    "safety_margin_mb": 600,
                    "short_prefill_post_alloc_free_mb": 1050,
                    "long_prefill_post_alloc_free_mb": 950,
                    "decode_hcs_budget_mb": 500,
                },
            )
            out = Path(tmp) / "approved.json"
            old_load = server._load_heatmap_prompts
            old_assert = server._assert_heatmap_prompts_are_held_out
            old_chat = server._chat_prompt_tokens
            old_meta = server._approved_heatmap_metadata
            try:
                server._load_heatmap_prompts = lambda path=None: ["one", "two"]
                server._assert_heatmap_prompts_are_held_out = lambda prompts: None
                server._chat_prompt_tokens = lambda *args, **kwargs: [1, 2]
                server._approved_heatmap_metadata = (
                    lambda *args, **kwargs: {"format": server.APPROVED_HEATMAP_FORMAT}
                )
                server._build_approved_heatmap(
                    model,
                    str(out),
                    args,
                    args.residency_calibration,
                )
            finally:
                server._load_heatmap_prompts = old_load
                server._assert_heatmap_prompts_are_held_out = old_assert
                server._chat_prompt_tokens = old_chat
                server._approved_heatmap_metadata = old_meta

            artifact = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(store.collection_inits, 1)
            self.assertEqual(len(store.pool_rankings), 1)
            self.assertEqual(store.pool_rankings[0][0], (0, 1))
            self.assertEqual(store.reload_calls, [12])
            self.assertEqual(artifact["0,1"], 10)
            self.assertEqual(artifact["0,0"], 20)

    def test_approved_builder_bootstrap_ranking_does_not_enter_output_counts(self):
        class FakeStore:
            def __init__(self):
                self.pool_rankings = []

            def set_vram_calibration(self, *args):
                return "calibrated"

            def hcs_start_collecting(self):
                return None

            def hcs_export_heatmap(self):
                return {"0,1": 7}

            def rust_prefill_tokens(self, *args, **kwargs):
                return 7, 12, False

            def gpu_generate_batch(self, **kwargs):
                return []

            def hcs_reset(self):
                return None

            def hcs_pool_init_tiered(self, ranking, **kwargs):
                self.pool_rankings.append(list(ranking))
                return "pool ready"

            def py_hcs_drain_vram_pressure(self, *args):
                return 0, 0.0, 700

            def py_hcs_reload_after_prefill(self, prompt_len):
                return 0, 0.0

        with tempfile.TemporaryDirectory() as tmp:
            bootstrap = Path(tmp) / "bootstrap.json"
            bootstrap.write_text("{}", encoding="utf-8")
            store = FakeStore()
            model = SimpleNamespace(
                cfg=SimpleNamespace(
                    n_routed_experts=2,
                    num_hidden_layers=1,
                    eos_token_id=99,
                    extra_stop_token_ids=[],
                ),
                layers=[SimpleNamespace(is_moe=True)],
                _gpu_decode_store=store,
                server_cleanup=lambda: None,
            )
            args = SimpleNamespace(
                approved_heatmap_prompts=None,
                approved_heatmap_max_prompts=0,
                approved_heatmap_decode_tokens=1,
                approved_heatmap_checkpoint_every=0,
                approved_heatmap_residency_refresh_every=1,
                approved_heatmap_resume_from=None,
                approved_heatmap_bootstrap_from=str(bootstrap),
                heatmap_path=None,
                benchmark=False,
                benchmark_only=False,
                temperature=0.0,
                enable_thinking=False,
                residency_calibration={
                    "short_tokens": 8,
                    "long_tokens": 16,
                    "prefill_short_free_mb": 1000,
                    "prefill_long_free_mb": 900,
                    "decode_short_free_mb": 1100,
                    "decode_long_free_mb": 1000,
                    "baseline_free_mb": 1200,
                    "safety_margin_mb": 600,
                    "short_prefill_post_alloc_free_mb": 1050,
                    "long_prefill_post_alloc_free_mb": 950,
                    "decode_hcs_budget_mb": 500,
                },
            )
            out = Path(tmp) / "approved.json"
            old_load = server._load_heatmap_prompts
            old_assert = server._assert_heatmap_prompts_are_held_out
            old_chat = server._chat_prompt_tokens
            old_meta = server._approved_heatmap_metadata
            old_bootstrap = server._load_heatmap_residency_bootstrap
            try:
                server._load_heatmap_prompts = lambda path=None: ["one"]
                server._assert_heatmap_prompts_are_held_out = lambda prompts: None
                server._chat_prompt_tokens = lambda *args, **kwargs: [1, 2]
                server._approved_heatmap_metadata = (
                    lambda *args, **kwargs: {"format": server.APPROVED_HEATMAP_FORMAT}
                )
                server._load_heatmap_residency_bootstrap = (
                    lambda *args, **kwargs: ({"0,0": 100}, server.APPROVED_HEATMAP_FORMAT)
                )
                server._build_approved_heatmap(
                    model,
                    str(out),
                    args,
                    args.residency_calibration,
                )
                server._load_heatmap_residency_bootstrap = (
                    lambda *args, **kwargs: ({}, server.APPROVED_HEATMAP_FORMAT)
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "bootstrap contains no route counts",
                ):
                    server._build_approved_heatmap(
                        model,
                        str(Path(tmp) / "empty-bootstrap.json"),
                        args,
                        args.residency_calibration,
                    )
            finally:
                server._load_heatmap_prompts = old_load
                server._assert_heatmap_prompts_are_held_out = old_assert
                server._chat_prompt_tokens = old_chat
                server._approved_heatmap_metadata = old_meta
                server._load_heatmap_residency_bootstrap = old_bootstrap

            artifact = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(store.pool_rankings[0][0], (0, 0))
            self.assertNotIn("0,0", artifact)
            self.assertEqual(artifact["0,1"], 7)

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

    def test_marlin_digest_cache_is_bound_to_file_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experts_marlin.bin"
            path.write_bytes(b"exact-marlin-payload")
            digest = server._sha256_file(str(path))

            server._write_marlin_digest_cache(str(path), digest)

            cache = json.loads(
                Path(server._marlin_digest_cache_path(str(path))).read_text(encoding="utf-8")
            )
            stat = path.stat()
            self.assertEqual(cache["format"], "krasis_marlin_sha256_cache")
            self.assertEqual(cache["size"], stat.st_size)
            self.assertEqual(cache["mtime_ns"], stat.st_mtime_ns)
            self.assertEqual(cache["sha256"], digest)

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
