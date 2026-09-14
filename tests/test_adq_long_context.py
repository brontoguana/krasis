import unittest
from pathlib import Path

from tokenizers import Tokenizer

from tests.adq_long_context_live import (
    EXPECTED,
    QUERY_FORMAT,
    TARGET_TOKENS,
    _sha256_ids,
    _short_isolation_ids,
    build_case,
    build_witness_input_artifact,
    select_smallest_qualifying_turn,
)


class AdqLongContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = Tokenizer.from_file(
            "/home/main/.krasis/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/tokenizer.json"
        )

    def test_case_is_exact_and_places_each_distinct_record(self) -> None:
        base = [100] * TARGET_TOKENS
        ids, records = build_case(base, self.tokenizer)
        self.assertEqual(len(ids), TARGET_TOKENS)
        self.assertEqual(len(records), 3)
        self.assertEqual(len({row["absolute_token_position"] for row in records}), 3)
        decoded_tail = self.tokenizer.decode(ids[-200:])
        self.assertIn(QUERY_FORMAT, decoded_tail)
        self.assertNotIn(EXPECTED, decoded_tail)

    def test_shorter_diagnostic_scales_record_positions(self) -> None:
        target = 131_072
        ids, records = build_case([100] * TARGET_TOKENS, self.tokenizer, target)
        self.assertEqual(len(ids), target)
        self.assertEqual(
            [row["absolute_token_position"] for row in records],
            [int(target * 0.03), int(target * 0.50), int(target * 0.95)],
        )

    def test_smallest_qualifying_staged_source_is_selected(self) -> None:
        source = {
            "conversations": [
                {"turns": [{"input_token_ids": [1] * 131_072}]},
                {"turns": [{"input_token_ids": [2] * 262_144}]},
                {"turns": [{"input_token_ids": [3] * 500_000}]},
            ]
        }
        selected = select_smallest_qualifying_turn(source, 131_072)
        self.assertEqual(len(selected["input_token_ids"]), 131_072)
        selected = select_smallest_qualifying_turn(source, 200_000)
        self.assertEqual(len(selected["input_token_ids"]), 262_144)
        with self.assertRaisesRegex(ValueError, "at least 600,000 tokens"):
            select_smallest_qualifying_turn(source, 600_000)

    def test_witness_input_preserves_retrieval_and_isolation_ids(self) -> None:
        retrieval_ids = [0, 128803, 271, 1]
        isolation_ids = _short_isolation_ids(self.tokenizer)
        artifact = build_witness_input_artifact(
            retrieval_ids, isolation_ids, Path("frozen-long-context.json")
        )
        turns = [row["turns"][0] for row in artifact["conversations"]]
        self.assertEqual(turns[0]["input_token_ids"], retrieval_ids)
        self.assertEqual(turns[0]["input_sha256"], _sha256_ids(retrieval_ids))
        self.assertEqual(turns[1]["input_token_ids"], isolation_ids)
        self.assertEqual(turns[1]["input_sha256"], _sha256_ids(isolation_ids))
        self.assertEqual(len(isolation_ids), 20)


if __name__ == "__main__":
    unittest.main()
