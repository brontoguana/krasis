import os
import tempfile
import unittest
from types import SimpleNamespace

from krasis.hf_downloader import (
    candidate_from_info,
    destination_for_repo,
    parse_hf_repo_id,
    selected_download_files,
    validate_local_model,
)


class HFDownloaderTests(unittest.TestCase):
    def test_parse_hf_repo_id_accepts_urls_and_repo_ids(self):
        self.assertEqual(parse_hf_repo_id("Qwen/Qwen3-Coder-Next"), "Qwen/Qwen3-Coder-Next")
        self.assertEqual(
            parse_hf_repo_id("https://huggingface.co/Qwen/Qwen3-Coder-Next/tree/main"),
            "Qwen/Qwen3-Coder-Next",
        )
        self.assertEqual(
            parse_hf_repo_id("https://huggingface.co/Qwen/Qwen3-Coder-Next/blob/main/config.json"),
            "Qwen/Qwen3-Coder-Next",
        )

    def test_parse_hf_repo_id_rejects_non_model_values(self):
        with self.assertRaises(ValueError):
            parse_hf_repo_id("https://example.com/Qwen/Qwen3-Coder-Next")
        with self.assertRaises(ValueError):
            parse_hf_repo_id("Qwen")

    def test_candidate_estimates_int4_payload_from_safetensors_metadata(self):
        info = SimpleNamespace(
            id="Qwen/Test",
            pipeline_tag="text-generation",
            tags=["transformers", "safetensors", "qwen3_moe"],
            gated=False,
            private=False,
            downloads=10,
            likes=2,
            last_modified=None,
            safetensors=SimpleNamespace(total=1_000_000, parameters={"BF16": 1_000_000}),
            siblings=[],
        )
        candidate = candidate_from_info(info, include_files=False)
        self.assertTrue(candidate.has_safetensors)
        self.assertEqual(candidate.int4_payload_bytes, 500_000)
        self.assertIn("qwen3_moe", candidate.summary)
        self.assertTrue(candidate.is_krasis_candidate)

    def test_candidate_filters_non_native_or_unknown_results(self):
        no_safetensors = candidate_from_info(
            SimpleNamespace(
                id="Org/BinOnly",
                pipeline_tag="text-generation",
                tags=["transformers"],
                gated=False,
                private=False,
                downloads=0,
                likes=0,
                last_modified=None,
                safetensors=None,
                siblings=[],
            ),
            include_files=False,
        )
        gguf = candidate_from_info(
            SimpleNamespace(
                id="Org/GGUF",
                pipeline_tag="text-generation",
                tags=["transformers", "gguf", "safetensors"],
                gated=False,
                private=False,
                downloads=0,
                likes=0,
                last_modified=None,
                safetensors=SimpleNamespace(total=1000, parameters={}),
                siblings=[],
            ),
            include_files=False,
        )
        unknown_task = candidate_from_info(
            SimpleNamespace(
                id="Org/Embedding",
                pipeline_tag="feature-extraction",
                tags=["safetensors"],
                gated=False,
                private=False,
                downloads=0,
                likes=0,
                last_modified=None,
                safetensors=SimpleNamespace(total=1000, parameters={}),
                siblings=[],
            ),
            include_files=False,
        )
        self.assertFalse(no_safetensors.is_krasis_candidate)
        self.assertFalse(gguf.is_krasis_candidate)
        self.assertFalse(unknown_task.is_krasis_candidate)

    def test_candidate_filters_quantized_conversion_results(self):
        for repo_id, tags in (
            ("mlx-community/Qwen-4bit", ["mlx", "safetensors"]),
            ("Org/Model-FP8", ["transformers", "safetensors", "fp8"]),
            ("Org/Model-LoRA", ["transformers", "safetensors", "lora"]),
            ("Org/Model-AWQ", ["transformers", "safetensors", "base_model:quantized:Org/Base"]),
        ):
            candidate = candidate_from_info(
                SimpleNamespace(
                    id=repo_id,
                    pipeline_tag="text-generation",
                    tags=tags,
                    gated=False,
                    private=False,
                    downloads=0,
                    likes=0,
                    last_modified=None,
                    safetensors=SimpleNamespace(total=1000, parameters={}),
                    siblings=[],
                ),
                include_files=False,
            )
            self.assertFalse(candidate.is_krasis_candidate, repo_id)

    def test_selected_download_files_skips_non_krasis_artifacts(self):
        files = [
            SimpleNamespace(rfilename="config.json", size=100, lfs=None),
            SimpleNamespace(rfilename="model-00001.safetensors", size=1000, lfs=None),
            SimpleNamespace(rfilename="model.gguf", size=2000, lfs=None),
            SimpleNamespace(rfilename="pytorch_model.bin", size=3000, lfs=None),
        ]
        selected = selected_download_files(SimpleNamespace(siblings=files))
        self.assertEqual([f.rfilename for f in selected], ["config.json", "model-00001.safetensors"])

    def test_destination_for_repo_preserves_org_repo_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                destination_for_repo(tmp, "Qwen/Qwen3-Coder-Next"),
                os.path.join(tmp, "Qwen", "Qwen3-Coder-Next"),
            )

    def test_validate_local_model_reports_missing_required_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            issues = validate_local_model(tmp)
            self.assertIn("missing config.json", issues)
            self.assertIn("missing .safetensors weights", issues)
            self.assertIn("missing tokenizer files", issues)


if __name__ == "__main__":
    unittest.main()
