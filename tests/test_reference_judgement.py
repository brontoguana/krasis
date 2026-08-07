"""No-GPU contracts for reference-test verdict aggregation."""

import unittest

from tests.reference_test import build_server_command, judge_prompt


class ReferenceJudgementTests(unittest.TestCase):
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
