import unittest

from tests.adq_build_witness_inputs import _fit_exact_chat_input, _model_context_limit


class _FakeBackend:
    @staticmethod
    def decode(values, **_kwargs):
        return "".join(chr(value) for value in values)


class _FakeTokenizer:
    tokenizer = _FakeBackend()

    def apply_chat_template(self, messages, **kwargs):
        self.last_kwargs = kwargs
        content = messages[0]["content"]
        return [1, 2] + [ord(value) for value in content] + [3, 4, 5]


class _BoundaryMergeTokenizer(_FakeTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        content = messages[0]["content"]
        alignment_present = " x" in content
        logical_length = len(content.replace(" x", ""))
        length = 5 + 2 * logical_length + int(alignment_present)
        return list(range(length))


class AdqWitnessInputBuilderTests(unittest.TestCase):
    def test_context_limit_supports_nested_multimodal_config(self) -> None:
        self.assertEqual(
            _model_context_limit(
                {"text_config": {"max_position_embeddings": 1_048_576}}
            ),
            1_048_576,
        )

    def test_exact_length_uses_checkpoint_native_renderer(self) -> None:
        tokenizer = _FakeTokenizer()
        values, alignment = _fit_exact_chat_input(
            tokenizer,
            list(range(65, 91)) * 4,
            64,
            " QUERY",
        )
        self.assertEqual(len(values), 64)
        self.assertEqual(alignment, "")
        self.assertEqual(
            tokenizer.last_kwargs,
            {"add_generation_prompt": True, "enable_thinking": False},
        )

    def test_target_must_exceed_fixed_chat_framing(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not larger than"):
            _fit_exact_chat_input(_FakeTokenizer(), [65, 66], 5, " QUERY")

    def test_alignment_closes_a_boundary_merge_length_gap(self) -> None:
        values, alignment = _fit_exact_chat_input(
            _BoundaryMergeTokenizer(),
            list(range(65, 91)),
            20,
            "Q",
        )
        self.assertEqual(len(values), 20)
        self.assertTrue(alignment)


if __name__ == "__main__":
    unittest.main()
