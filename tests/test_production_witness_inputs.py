import tempfile
import unittest
from pathlib import Path

from tests.step37_witness_inputs import _iter_prompt_turns, _resolve_prompt


class ProductionWitnessInputTests(unittest.TestCase):
    def test_resolves_relative_prompt_file_and_exact_byte_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompt = root / "prompt.md"
            prompt.write_bytes(b"0123456789")
            contract = root / "contract.json"
            turn = {"prompt_file": "prompt.md", "prompt_max_bytes": 6}
            self.assertEqual(_resolve_prompt(turn, contract), "012345")

    def test_invalid_utf8_byte_boundary_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompt = root / "prompt.md"
            prompt.write_text("A£B", encoding="utf-8")
            contract = root / "contract.json"
            with self.assertRaisesRegex(ValueError, "not valid UTF-8"):
                _resolve_prompt(
                    {"prompt_file": "prompt.md", "prompt_max_bytes": 2},
                    contract,
                )

    def test_embedded_prompt_still_supported(self) -> None:
        source = {"conversations": [{"turns": [{"prompt": "hello"}]}]}
        rows = list(_iter_prompt_turns(source, Path("contract.json")))
        self.assertEqual(rows, [(0, 0, "hello")])


if __name__ == "__main__":
    unittest.main()
