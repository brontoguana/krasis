"""Focused contracts for the live session-cache acceptance calculations."""

from __future__ import annotations

import unittest

from tests.test_session_cache_live import (
    _completion_trace,
    _cross_height_is_within_baseline,
    _distribution_drift,
    _measured_variation_envelope,
    _same_height_control_envelope,
    _sequence_drift,
)


class SessionCacheGateTests(unittest.TestCase):
    def test_completion_trace_uses_prefill_distribution_for_first_token(self) -> None:
        response = {
            "krasis_debug": {
                "completion_token_ids": [7, 8],
                "completion_decode_trace": [
                    {"token_id": 7, "top_k": []},
                    {"token_id": 8, "top_k": [{"token_id": 8, "log_prob": -0.2}]},
                ],
                "first_token_logits": {
                    "top_logits_before_logprob": [
                        {"token_id": 7, "logprob": -0.1},
                    ],
                },
            },
        }
        trace = _completion_trace(response)
        self.assertEqual(trace[0]["top_k"][0]["token_id"], 7)
        self.assertEqual(trace[1]["top_k"][0]["token_id"], 8)

    def test_sequence_drift_reports_first_divergent_token(self) -> None:
        drift = _sequence_drift([10, 20, 30, 40], [10, 20, 31, 40])
        self.assertEqual(drift["matched_prefix_tokens"], 2)
        self.assertEqual(drift["positional_matches"], 3)
        self.assertEqual(drift["first_divergent_token_index"], 2)
        self.assertFalse(drift["token_identical"])

    def test_distribution_drift_uses_only_shared_token_scores(self) -> None:
        drift = _distribution_drift(
            [
                {"token_id": 1, "log_prob": -0.1},
                {"token_id": 2, "log_prob": -1.0},
            ],
            [
                {"token_id": 2, "logprob": -1.25},
                {"token_id": 3, "logprob": -2.0},
            ],
        )
        self.assertTrue(drift["top_token_changed"])
        self.assertEqual(drift["top_k_overlap"], 1)
        self.assertAlmostEqual(drift["max_shared_log_prob_delta"], 0.25)

    def test_cross_height_drift_must_stay_inside_measured_envelope(self) -> None:
        baseline = {
            "max_shared_log_prob_delta": 0.25,
            "minimum_top_k_overlap": 7,
        }
        accepted = {
            "token_identical": False,
            "decision_step_distribution": {
                "max_shared_log_prob_delta": 0.20,
                "top_k_overlap": 8,
            },
        }
        worse_delta = {
            "token_identical": False,
            "decision_step_distribution": {
                "max_shared_log_prob_delta": 0.30,
                "top_k_overlap": 8,
            },
        }
        worse_overlap = {
            "token_identical": False,
            "decision_step_distribution": {
                "max_shared_log_prob_delta": 0.20,
                "top_k_overlap": 6,
            },
        }
        self.assertTrue(_cross_height_is_within_baseline(accepted, baseline))
        self.assertFalse(_cross_height_is_within_baseline(worse_delta, baseline))
        self.assertFalse(_cross_height_is_within_baseline(worse_overlap, baseline))

    def test_matched_step_tail_drift_does_not_replace_divergence_metric(self) -> None:
        cross_height = {
            "token_identical": False,
            "max_shared_log_prob_delta": 9.0,
            "minimum_top_k_overlap": 2,
            "decision_step_distribution": {
                "max_shared_log_prob_delta": 0.2,
                "top_k_overlap": 9,
            },
        }
        baseline = {
            "max_shared_log_prob_delta": 0.25,
            "minimum_top_k_overlap": 7,
        }
        self.assertTrue(_cross_height_is_within_baseline(cross_height, baseline))

    def test_token_identity_always_passes_cross_height_measurement(self) -> None:
        self.assertTrue(
            _cross_height_is_within_baseline(
                {"token_identical": True},
                {"max_shared_log_prob_delta": 0.0, "minimum_top_k_overlap": 0},
            )
        )

    def test_same_height_full_variation_becomes_measured_envelope(self) -> None:
        envelope = _measured_variation_envelope({
            "max_shared_log_prob_delta": 1.75,
            "minimum_top_k_overlap": 6,
        })
        self.assertEqual(
            envelope["source"],
            "cache_disabled_same_height_full_prefill_repeat",
        )
        self.assertEqual(envelope["max_shared_log_prob_delta"], 1.75)
        self.assertEqual(envelope["minimum_top_k_overlap"], 6)

    def test_same_height_control_envelope_detects_nondeterminism(self) -> None:
        def response(tokens: list[int]) -> dict:
            return {
                "krasis_debug": {
                    "completion_token_ids": tokens,
                    "completion_decode_trace": [
                        {
                            "token_id": token,
                            "top_k": [{"token_id": token, "log_prob": -0.1}],
                        }
                        for token in tokens
                    ],
                    "first_token_logits": {
                        "top_logits_before_logprob": [
                            {"token_id": tokens[0], "logprob": -0.1},
                        ],
                    },
                },
            }

        envelope = _same_height_control_envelope([
            response([1, 2, 3]),
            response([1, 2, 4]),
            response([1, 2, 3]),
        ])
        self.assertFalse(envelope["token_identical"])
        self.assertEqual(envelope["samples"], 3)
        self.assertEqual(envelope["unique_token_sequences"], 2)


if __name__ == "__main__":
    unittest.main()
