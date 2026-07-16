import os
import sys
import tempfile
import unittest
from types import ModuleType
from types import SimpleNamespace

import krasis.hf_downloader as hf_downloader
from krasis.hf_downloader import (
    candidate_from_info,
    destination_for_supported_model,
    destination_for_repo,
    download_model,
    get_supported_model_details,
    hf_login,
    parse_hf_repo_id,
    selected_download_files,
    supported_models,
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
            pipeline_tag="image-text-to-text",
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
            ("z-lab/Qwen3.6-35B-A3B-DFlash", ["transformers", "safetensors", "dflash", "draft-model"]),
            ("deepseek-ai/DeepSeek-OCR", ["transformers", "safetensors", "vision-language", "ocr"]),
            ("tiny-random/minimax-m2.5", ["transformers", "safetensors"]),
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

    def test_supported_model_catalog_is_exact_and_pinned(self):
        models = supported_models()
        self.assertEqual(
            [m.repo_id for m in models],
            [
                "Qwen/Qwen3-Coder-Next",
                "stepfun-ai/Step-3.7-Flash",
                "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
                "Qwen/Qwen3.6-35B-A3B",
                "deepreinforce-ai/Ornith-1.0-35B",
                "Qwen/Qwen3.5-35B-A3B",
                "Qwen/Qwen3.5-122B-A10B",
                "Qwen/Qwen3-235B-A22B",
                "google/gemma-4-26b-a4b-it",
            ],
        )
        self.assertEqual(len({m.key for m in models}), len(models))
        self.assertEqual(len({m.local_dir_name for m in models}), len(models))
        repo_root = os.path.dirname(os.path.dirname(__file__))
        for model in models:
            self.assertRegex(model.revision, r"^[0-9a-f]{40}$")
            self.assertTrue(
                os.path.exists(os.path.join(repo_root, model.recommended_config)),
                model.recommended_config,
            )

    def test_destination_for_supported_model_uses_krasis_model_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = supported_models()[0]
            self.assertEqual(
                destination_for_supported_model(tmp, model),
                os.path.join(tmp, "Qwen3-Coder-Next"),
            )

    def test_get_supported_model_details_uses_pinned_revision(self):
        calls = []

        class FakeApi:
            def model_info(self, repo_id, *, revision=None, files_metadata=False, token=None):
                calls.append((repo_id, revision, files_metadata, token))
                return SimpleNamespace(
                    id=repo_id,
                    pipeline_tag="text-generation",
                    tags=["transformers", "safetensors"],
                    gated=False,
                    private=False,
                    downloads=1,
                    likes=0,
                    last_modified=None,
                    safetensors=SimpleNamespace(total=1000, parameters={}),
                    siblings=[
                        SimpleNamespace(rfilename="config.json", size=100, lfs=None),
                        SimpleNamespace(rfilename="model.safetensors", size=1000, lfs=None),
                        SimpleNamespace(rfilename="model.gguf", size=2000, lfs=None),
                    ],
                )

        original = hf_downloader._require_hf
        hf_downloader._require_hf = lambda: (FakeApi, lambda: "hf_test", None, None, None, None, None)
        try:
            candidate = get_supported_model_details("qcn")
        finally:
            hf_downloader._require_hf = original

        spec = supported_models()[0]
        self.assertEqual(calls, [(spec.repo_id, spec.revision, True, True)])
        self.assertTrue(candidate.supported_download)
        self.assertTrue(candidate.is_krasis_candidate)
        self.assertEqual(candidate.local_dir_name, spec.local_dir_name)
        self.assertEqual(candidate.revision, spec.revision)
        self.assertEqual(candidate.compatibility, "supported by Krasis")

    def test_download_model_passes_revision_to_snapshot_download(self):
        calls = []

        def snapshot_download(**kwargs):
            calls.append(kwargs)
            return kwargs["local_dir"]

        original = hf_downloader._require_hf
        hf_downloader._require_hf = lambda: (None, lambda: None, None, snapshot_download, None, None, None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                result = download_model("Org/Model", tmp, revision="abc123", max_workers=3)
        finally:
            hf_downloader._require_hf = original

        self.assertEqual(result, tmp)
        self.assertEqual(calls[0]["repo_id"], "Org/Model")
        self.assertEqual(calls[0]["revision"], "abc123")
        self.assertEqual(calls[0]["max_workers"], 3)

    def test_validate_local_model_reports_missing_required_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            issues = validate_local_model(tmp)
            self.assertIn("missing config.json", issues)
            self.assertIn("missing .safetensors weights", issues)
            self.assertIn("missing tokenizer files", issues)

    def test_require_hf_does_not_require_removed_hffolder_symbol(self):
        fake_hub = ModuleType("huggingface_hub")
        fake_hub.__path__ = []
        fake_hub.HfApi = object
        fake_hub.get_token = lambda: None
        fake_hub.login = lambda **_kwargs: None
        fake_hub.snapshot_download = lambda **_kwargs: None

        fake_errors = ModuleType("huggingface_hub.errors")
        fake_errors.GatedRepoError = type("GatedRepoError", (Exception,), {})
        fake_errors.HfHubHTTPError = type("HfHubHTTPError", (Exception,), {})
        fake_errors.RepositoryNotFoundError = type("RepositoryNotFoundError", (Exception,), {})

        names = ("huggingface_hub", "huggingface_hub.errors")
        original = {name: sys.modules.get(name) for name in names}
        sys.modules["huggingface_hub"] = fake_hub
        sys.modules["huggingface_hub.errors"] = fake_errors
        try:
            result = hf_downloader._require_hf()
        finally:
            for name, module in original.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertIs(result[0], object)
        self.assertIs(result[1], fake_hub.get_token)
        self.assertIs(result[2], fake_hub.login)
        self.assertIs(result[3], fake_hub.snapshot_download)

    def test_hf_login_supports_older_hub_signature_without_new_session(self):
        calls = []

        class FakeApi:
            def whoami(self, token):
                calls.append(("whoami", token))
                return {"name": "tester"}

        def get_token():
            return "hf_test"

        def old_login(token=None, *, add_to_git_credential=False):
            calls.append(("login", token, add_to_git_credential))

        original = hf_downloader._require_hf
        hf_downloader._require_hf = lambda: (FakeApi, get_token, old_login, None, None, None, None)
        try:
            result = hf_login(" hf_test ")
        finally:
            hf_downloader._require_hf = original

        self.assertEqual(result, {"logged_in": True, "user": "tester"})
        self.assertEqual(calls, [("whoami", "hf_test"), ("login", "hf_test", False)])

    def test_hf_login_saves_token_when_cache_is_empty(self):
        calls = []

        class FakeApi:
            def whoami(self, token):
                calls.append(("whoami", token))
                return {"name": "tester"}

        def get_token():
            return None

        def new_login(token=None, *, add_to_git_credential=False, new_session=True):
            calls.append(("login", token, add_to_git_credential, new_session))

        original = hf_downloader._require_hf
        original_save = hf_downloader._save_hf_token
        hf_downloader._require_hf = lambda: (FakeApi, get_token, new_login, None, None, None, None)
        hf_downloader._save_hf_token = lambda token: calls.append(("save_token", token))
        try:
            result = hf_login(" hf_saved ")
        finally:
            hf_downloader._require_hf = original
            hf_downloader._save_hf_token = original_save

        self.assertEqual(result, {"logged_in": True, "user": "tester"})
        self.assertEqual(
            calls,
            [
                ("whoami", "hf_saved"),
                ("login", "hf_saved", False, True),
                ("save_token", "hf_saved"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
