import os
import tempfile
import unittest
from unittest.mock import patch

from krasis.tokenizer import _load_hf_tokenizer


class TokenizerLoaderTests(unittest.TestCase):
    def test_ordinary_checkpoint_uses_auto_tokenizer(self):
        sentinel = object()
        with patch("krasis.tokenizer.AutoTokenizer.from_pretrained", return_value=sentinel) as load:
            result = _load_hf_tokenizer("/model", {}, {"trust_remote_code": True})

        self.assertIs(result, sentinel)
        load.assert_called_once_with("/model", trust_remote_code=True)

    def test_tokenizers_backend_uses_explicit_fast_compatibility_class(self):
        sentinel = object()
        with tempfile.TemporaryDirectory() as model_path:
            tokenizer_file = os.path.join(model_path, "tokenizer.json")
            with open(tokenizer_file, "wb") as handle:
                handle.write(b"{}")

            cfg = {
                "backend": "tokenizers",
                "tokenizer_class": "TokenizersBackend",
            }
            with (
                patch("krasis.tokenizer.AutoTokenizer.from_pretrained") as auto_load,
                patch(
                    "krasis.tokenizer.PreTrainedTokenizerFast.from_pretrained",
                    return_value=sentinel,
                ) as fast_load,
            ):
                result = _load_hf_tokenizer(
                    model_path, cfg, {"extra_special_tokens": {}}
                )

        self.assertIs(result, sentinel)
        auto_load.assert_not_called()
        fast_load.assert_called_once_with(
            model_path, extra_special_tokens={}
        )

    def test_tokenizers_backend_requires_exact_backend(self):
        cfg = {
            "backend": "unknown",
            "tokenizer_class": "TokenizersBackend",
        }
        with self.assertRaisesRegex(ValueError, "only the explicit 'tokenizers' backend"):
            _load_hf_tokenizer("/model", cfg, {})

    def test_tokenizers_backend_requires_local_tokenizer_file(self):
        cfg = {
            "backend": "tokenizers",
            "tokenizer_class": "TokenizersBackend",
        }
        with tempfile.TemporaryDirectory() as model_path:
            with self.assertRaisesRegex(FileNotFoundError, "tokenizer.json is missing"):
                _load_hf_tokenizer(model_path, cfg, {})


if __name__ == "__main__":
    unittest.main()
