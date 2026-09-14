import unittest
from pathlib import Path

from tokenizers import Tokenizer

from tests.adq_session_soak_live import _render


TOKENIZER = "/home/main/.krasis/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/tokenizer.json"


class AdqSessionSoakTests(unittest.TestCase):
    def test_live_entry_point_adds_repository_root_before_tests_import(self) -> None:
        source = (Path(__file__).with_name("adq_session_soak_live.py")).read_text(
            encoding="utf-8"
        )
        path_init = 'sys.path.insert(0, str(Path(__file__).resolve().parents[1]))'
        self.assertLess(source.index(path_init), source.index("from tests.adq_build_witness_inputs"))

    def test_renderer_matches_bundled_deepseek_v4_contract(self) -> None:
        rendered = _render([{"role": "user", "content": "hello"}])
        self.assertEqual(
            rendered,
            "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>",
        )

    def test_multiturn_renderer_includes_closed_history(self) -> None:
        rendered = _render(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "second"},
            ]
        )
        self.assertEqual(
            rendered,
            "<｜begin▁of▁sentence｜><｜User｜>first"
            "<｜Assistant｜></think>answer<｜end▁of▁sentence｜>"
            "<｜User｜>second<｜Assistant｜></think>",
        )

    def test_renderer_token_count_uses_real_checkpoint_tokenizer(self) -> None:
        tokenizer = Tokenizer.from_file(TOKENIZER)
        rendered = _render([{"role": "user", "content": "hello"}])
        self.assertEqual(
            len(tokenizer.encode(rendered, add_special_tokens=False).ids),
            5,
        )

if __name__ == "__main__":
    unittest.main()
