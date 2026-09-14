import unittest

from tests.adq_compare_witness import _metrics


def _row(selected: int, selected_logprob: float, extra: int) -> dict:
    return {
        "token_id": selected,
        "log_prob": selected_logprob,
        "top_k": [
            {"token_id": selected, "log_prob": selected_logprob},
            {"token_id": extra, "log_prob": selected_logprob - 1.0},
        ],
    }


class AdqWitnessComparisonTests(unittest.TestCase):
    def test_logits_after_first_divergence_are_not_compared(self) -> None:
        reference = {
            "token_ids": [10, 20, 30, 40],
            "per_token_data": [
                _row(10, -0.1, 99),
                _row(20, -0.2, 98),
                _row(30, -0.3, 97),
                _row(40, -0.4, 96),
            ],
        }
        actual = {
            "token_ids": [10, 21, 30, 40],
            "per_token_data": [
                _row(10, -0.15, 99),
                {
                    "token_id": 21,
                    "log_prob": -0.2,
                    "top_k": [
                        {"token_id": 21, "log_prob": -0.2},
                        {"token_id": 20, "log_prob": -0.25},
                    ],
                },
                _row(30, -9.0, 30),
                _row(40, -9.0, 40),
            ],
            "timing": {
                "safety_margin_mb": 600,
                "vram_low_water": [{"min_free_mb": 700}],
            },
        }
        result = _metrics(reference, actual)
        self.assertEqual(result["exact_prefix_tokens"], 1)
        self.assertEqual(result["identical_context_logit_steps"], 2)
        self.assertEqual(result["witness_top10_total"], 2)
        self.assertEqual(result["witness_top10_hits"], 2)
        self.assertAlmostEqual(result["selected_logprob_abs_delta_max"], 0.05)

    def test_first_token_distribution_is_reported_separately(self) -> None:
        reference = {"token_ids": [7], "per_token_data": [_row(7, -0.4, 8)]}
        actual = {
            "token_ids": [7],
            "per_token_data": [_row(7, -0.1, 8)],
            "timing": {
                "safety_margin_mb": 600,
                "vram_low_water": [{"min_free_mb": 600}],
            },
        }
        result = _metrics(reference, actual)
        self.assertAlmostEqual(result["first_token_selected_logprob_abs_delta"], 0.3)
        self.assertEqual(result["first_token_top10_overlap"], 2)
        self.assertTrue(result["vram_safe"])


if __name__ == "__main__":
    unittest.main()
