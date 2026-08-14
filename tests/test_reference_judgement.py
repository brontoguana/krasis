"""No-GPU contracts for reference-test verdict aggregation."""

import os
import tempfile
import unittest
from unittest.mock import patch

from tests.reference_test import (
    build_server_command,
    find_reference_data,
    judge_prompt,
    list_available_references,
)
from tests.validate_model import _resolve_reference_dir


class ReferenceJudgementTests(unittest.TestCase):
    def test_explicit_reference_output_dir_supports_detached_worktrees(self) -> None:
        with tempfile.TemporaryDirectory() as ref_root:
            model_dir = os.path.join(ref_root, "Qwen3-Coder-Next")
            os.makedirs(model_dir)
            reference_path = os.path.join(model_dir, "greedy_reference.json")
            with open(reference_path, "w", encoding="utf-8") as handle:
                handle.write("{}")

            with patch.dict(
                os.environ,
                {"KRASIS_REFERENCE_OUTPUT_DIR": ref_root},
                clear=False,
            ):
                self.assertEqual(
                    find_reference_data("Qwen3-Coder-Next", "/detached/krasis"),
                    reference_path,
                )
                self.assertEqual(
                    list_available_references("/detached/krasis"),
                    ["Qwen3-Coder-Next"],
                )

    def test_invalid_explicit_reference_output_dir_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            sibling_model_dir = os.path.join(
                temp_root,
                "krasis-internal",
                "reference-outputs",
                "output",
                "Qwen3-Coder-Next",
            )
            os.makedirs(sibling_model_dir)
            with open(
                os.path.join(sibling_model_dir, "greedy_reference.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write("{}")

            missing_root = os.path.join(temp_root, "missing-reference-root")
            with patch.dict(
                os.environ,
                {"KRASIS_REFERENCE_OUTPUT_DIR": missing_root},
                clear=False,
            ):
                self.assertIsNone(
                    find_reference_data(
                        "Qwen3-Coder-Next",
                        os.path.join(temp_root, "krasis"),
                    )
                )
                self.assertEqual(
                    list_available_references(os.path.join(temp_root, "krasis")),
                    [],
                )
                self.assertIsNone(
                    _resolve_reference_dir("Qwen3-Coder-Next")
                )

    def test_validate_model_uses_explicit_reference_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as ref_root:
            model_dir = os.path.join(ref_root, "Qwen3-Coder-Next-prefill")
            os.makedirs(model_dir)
            with open(
                os.path.join(model_dir, "greedy_reference.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write("{}")

            with patch.dict(
                os.environ,
                {"KRASIS_REFERENCE_OUTPUT_DIR": ref_root},
                clear=False,
            ):
                self.assertEqual(
                    os.fspath(_resolve_reference_dir("Qwen3-Coder-Next")),
                    os.path.realpath(model_dir),
                )

    def test_reference_server_command_preserves_selected_gpu_override(self) -> None:
        command = build_server_command("/env/python", "model.conf", "1,3")
        self.assertEqual(
            command,
            [
                "/env/python",
                "-m",
                "krasis.server",
                "--config",
                "model.conf",
                "--test-endpoints",
                "--selected-gpus",
                "1,3",
            ],
        )

    def test_multitoken_prefill_pass_cannot_hide_decode_failure(self) -> None:
        verdict = judge_prompt(
            {
                "first_match": False,
                "match_run": 0,
                "ref_tokens_count": 50,
                "containment_rate": 0.08,
            },
            {"prefill_total": 1, "prefill_containment": 1.0},
            has_linear_attention=True,
        )

        self.assertEqual(verdict, "FAIL")

    def test_multitoken_pass_requires_prefill_and_decode_to_pass(self) -> None:
        verdict = judge_prompt(
            {
                "first_match": True,
                "match_run": 16,
                "ref_tokens_count": 50,
                "containment_rate": 0.60,
            },
            {"prefill_total": 4, "prefill_containment": 0.80},
            has_linear_attention=True,
        )

        self.assertEqual(verdict, "PASS")

    def test_multitoken_result_keeps_worse_prefill_verdict(self) -> None:
        verdict = judge_prompt(
            {
                "first_match": True,
                "match_run": 16,
                "ref_tokens_count": 50,
                "containment_rate": 0.60,
            },
            {"prefill_total": 4, "prefill_containment": 0.70},
            has_linear_attention=True,
        )

        self.assertEqual(verdict, "WARN")

    def test_first_token_artifact_preserves_prefill_primary_semantics(self) -> None:
        verdict = judge_prompt(
            {
                "first_match": False,
                "match_run": 0,
                "ref_tokens_count": 1,
                "containment_rate": 0.0,
            },
            {"prefill_total": 1, "prefill_containment": 1.0},
            has_linear_attention=True,
        )

        self.assertEqual(verdict, "PASS")

    def test_decode_only_fallback_keeps_existing_sequence_gate(self) -> None:
        verdict = judge_prompt(
            {
                "first_match": True,
                "match_run": 5,
                "ref_tokens_count": 10,
                "containment_rate": 0.75,
            },
            {},
            has_linear_attention=False,
        )

        self.assertEqual(verdict, "WARN")


if __name__ == "__main__":
    unittest.main()
