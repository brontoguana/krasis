import unittest
from pathlib import Path

from krasis.tokenizer import Tokenizer
from tests.adq_session_soak_live import _chat_ids, _token_count


DEEPSEEK_MODEL = "/home/main/.krasis/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
GLM_MODEL = "/home/main/.krasis/models/GLM-5.3-Flash"


class AdqSessionSoakTests(unittest.TestCase):
    def test_live_entry_point_adds_repository_root_before_tests_import(self) -> None:
        source = (Path(__file__).with_name("adq_session_soak_live.py")).read_text(
            encoding="utf-8"
        )
        path_init = 'sys.path.insert(0, str(Path(__file__).resolve().parents[1]))'
        self.assertLess(source.index(path_init), source.index("from tests.adq_build_witness_inputs"))

    def test_checkpoint_native_chat_ids_cover_deepseek_and_glm(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        for model_dir in (DEEPSEEK_MODEL, GLM_MODEL):
            with self.subTest(model_dir=model_dir):
                tokenizer = Tokenizer(model_dir)
                ids = _chat_ids(tokenizer, messages)
                self.assertEqual(ids, tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, enable_thinking=False
                ))
                self.assertGreater(len(ids), 0)

    def test_multiturn_uses_checkpoint_native_contract(self) -> None:
        messages = (
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "second"},
            ]
        )
        deepseek = _chat_ids(Tokenizer(DEEPSEEK_MODEL), messages)
        glm = _chat_ids(Tokenizer(GLM_MODEL), messages)
        self.assertNotEqual(deepseek, glm)
        self.assertGreater(len(deepseek), 0)
        self.assertGreater(len(glm), 0)

    def test_renderer_token_count_uses_real_checkpoint_tokenizer(self) -> None:
        for model_dir in (DEEPSEEK_MODEL, GLM_MODEL):
            with self.subTest(model_dir=model_dir):
                tokenizer = Tokenizer(model_dir)
                messages = [{"role": "user", "content": "hello"}]
                self.assertEqual(_token_count(tokenizer, messages), len(_chat_ids(tokenizer, messages)))

if __name__ == "__main__":
    unittest.main()
