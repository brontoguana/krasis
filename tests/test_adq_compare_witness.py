import unittest

from tests.adq_compare_witness import _build_request, _metrics


def _row(selected: int, selected_logprob: float, extra: int) -> dict:
    return {
        "token_id": selected,
        "log_prob": selected_logprob,
        "top_k": [
            {"token_id": selected, "log_prob": selected_logprob},
            {"token_id": extra, "log_prob": selected_logprob - 1.0},
        ],
    }


def _prefill_reference(position: int, selected: int, selected_logprob: float) -> dict:
    return {
        "positions": [
            {
                "position": position,
                "top_k": [
                    {"token_id": selected, "log_prob": selected_logprob, "logit": 4.0},
                    {"token_id": 99, "log_prob": selected_logprob - 1.0, "logit": 3.0},
                ],
            }
        ]
    }


def _prefill_actual(prompt_len: int, selected: int, selected_logprob: float) -> dict:
    return {
        "prompt_len": prompt_len,
        "first_token_logits": {
            "available": True,
            "nan_count": 0,
            "pos_inf_count": 0,
            "neg_inf_count": 0,
            "top_logits_before_logprob": [
                {"token_id": selected, "logprob": selected_logprob, "raw_logit": 4.0},
                {"token_id": 98, "logprob": selected_logprob - 1.0, "raw_logit": 3.0},
            ],
        },
    }


class AdqWitnessComparisonTests(unittest.TestCase):
    def test_trace_request_is_explicit_and_fail_closed(self) -> None:
        request = _build_request(
            [1, 2, 3],
            4,
            [5],
            debug_reference_trace=True,
            debug_prefill_device_trace_all_layers=True,
            debug_prefill_device_trace_layer=7,
            debug_prefill_device_trace_full_pre_out_proj=True,
            debug_prefill_device_trace_rows=[2],
            debug_prefill_device_trace_dims=[0, 9],
            debug_prefill_device_trace_experts=[11],
            debug_prefill_device_trace_local_scan_token=2,
        )
        self.assertEqual(request["input_token_ids"], [1, 2, 3])
        self.assertTrue(request["debug_prompt_trace"])
        self.assertTrue(request["debug_reference_trace"])
        self.assertTrue(request["debug_prefill_device_trace"])
        self.assertTrue(request["debug_prefill_device_trace_all_layers"])
        self.assertEqual(request["debug_prefill_device_trace_layer"], 7)
        self.assertTrue(request["debug_prefill_device_trace_full_pre_out_proj"])
        self.assertEqual(request["debug_prefill_device_trace_rows"], [2])
        self.assertEqual(request["debug_prefill_device_trace_dims"], [0, 9])
        self.assertEqual(request["debug_prefill_device_trace_experts"], [11])
        self.assertEqual(request["debug_prefill_device_trace_local_scan_token"], 2)

    def test_default_request_does_not_enable_device_trace(self) -> None:
        request = _build_request([1], 1, [])
        self.assertTrue(request["debug_prompt_trace"])
        self.assertNotIn("debug_reference_trace", request)
        self.assertNotIn("debug_prefill_device_trace", request)

    def test_logits_after_first_divergence_are_not_compared(self) -> None:
        reference = {
            "input_token_ids": [1, 2, 3],
            "token_ids": [10, 20, 30, 40],
            "prefill_logits": _prefill_reference(2, 10, -0.1),
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
            "debug_prompt_trace": _prefill_actual(3, 10, -0.15),
        }
        result = _metrics(reference, actual)
        self.assertEqual(result["exact_prefix_tokens"], 1)
        self.assertEqual(result["identical_context_logit_steps"], 2)
        self.assertEqual(result["witness_top10_total"], 2)
        self.assertEqual(result["witness_top10_hits"], 2)
        self.assertAlmostEqual(result["selected_logprob_abs_delta_max"], 0.05)

    def test_first_token_distribution_is_reported_separately(self) -> None:
        reference = {"token_ids": [7], "per_token_data": [_row(7, -0.4, 8)]}
        reference["input_token_ids"] = [1, 2]
        reference["prefill_logits"] = _prefill_reference(1, 7, -0.4)
        actual = {
            "token_ids": [7],
            "per_token_data": [_row(7, -0.1, 8)],
            "timing": {
                "safety_margin_mb": 600,
                "vram_low_water": [{"min_free_mb": 600}],
            },
            "debug_prompt_trace": _prefill_actual(2, 7, -0.1),
        }
        result = _metrics(reference, actual)
        self.assertAlmostEqual(result["first_token_selected_logprob_abs_delta"], 0.3)
        self.assertEqual(result["first_token_top10_overlap"], 2)
        self.assertTrue(result["vram_safe"])
        self.assertTrue(result["prefill"]["evidence_complete"])
        self.assertEqual(result["prefill"]["top10_overlap"], 1)
        self.assertAlmostEqual(result["prefill"]["selected_logprob_abs_delta"], 0.3)

    def test_vram_failure_does_not_erase_complete_numerical_evidence(self) -> None:
        reference = {
            "input_token_ids": [1, 2],
            "token_ids": [7],
            "per_token_data": [_row(7, -0.4, 8)],
            "prefill_logits": _prefill_reference(1, 7, -0.4),
        }
        actual = {
            "token_ids": [7],
            "per_token_data": [_row(7, -0.1, 8)],
            "debug_prompt_trace": _prefill_actual(2, 7, -0.1),
            "timing": {
                "safety_margin_mb": 600,
                "vram_low_water": [{"min_free_mb": 599}],
            },
        }
        result = _metrics(reference, actual)
        self.assertTrue(result["numerical_evidence_complete"])
        self.assertFalse(result["vram_safe"])

    def test_missing_prefill_distribution_fails_numerical_evidence_closed(self) -> None:
        reference = {
            "input_token_ids": [1, 2],
            "token_ids": [7],
            "per_token_data": [_row(7, -0.4, 8)],
            "prefill_logits": _prefill_reference(1, 7, -0.4),
        }
        actual = {
            "token_ids": [7],
            "per_token_data": [_row(7, -0.1, 8)],
            "timing": {
                "safety_margin_mb": 600,
                "vram_low_water": [{"min_free_mb": 700}],
            },
        }
        result = _metrics(reference, actual)
        self.assertFalse(result["numerical_evidence_complete"])
        self.assertTrue(result["vram_safe"])


if __name__ == "__main__":
    unittest.main()
