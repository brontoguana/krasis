"""Hugging Face model search and download helpers for the Krasis launcher."""

from __future__ import annotations

import fnmatch
import inspect
import os
from pathlib import Path
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse


KRASIS_HF_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "*.model",
]

KRASIS_HF_IGNORE_PATTERNS = [
    "*.bin",
    "*.gguf",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.onnx",
    "*.msgpack",
    "optimizer.*",
    "training_args.bin",
    "runs/*",
    "checkpoints/*",
]

UNSUPPORTED_REPO_TAGS = {
    "gguf",
    "mlx",
    "onnx",
    "coreml",
    "tflite",
    "openvino",
    "awq",
    "gptq",
    "exl2",
    "fp8",
    "fp4",
    "nvfp4",
    "compressed-tensors",
    "modelopt",
    "model optimizer",
    "vllm",
    "quantized",
    "4-bit",
    "8-bit",
    "lora",
    "qlora",
    "peft",
    "adapter",
    "adapter-transformers",
    "dflash",
    "draft-model",
    "speculative-decoding",
    "mtp",
    "vision-language",
    "ocr",
    "image-to-text",
    "visual-question-answering",
    "text-to-speech",
    "automatic-speech-recognition",
    "audio",
    "video",
}

UNSUPPORTED_REPO_NAME_MARKERS = (
    "gguf",
    "mlx",
    "awq",
    "gptq",
    "exl2",
    "fp8",
    "fp4",
    "nvfp4",
    "modelopt",
    "compressed-tensors",
    "dflash",
    "ocr",
    "-vl",
    "_vl",
    "vision",
    "tiny-random",
)

ALLOWED_PIPELINE_TAGS = {
    "",
    "text-generation",
    "conversational",
    # HF tags some Qwen text/MoE repos this way even when the downloadable
    # payload is a native Transformers safetensors model Krasis can inspect.
    "image-text-to-text",
}


@dataclass
class SupportedHFModel:
    key: str
    display_name: str
    repo_id: str
    local_dir_name: str
    revision: str
    recommended_config: str
    notes: str


SUPPORTED_HF_MODELS: Sequence[SupportedHFModel] = (
    SupportedHFModel(
        key="qcn",
        display_name="Qwen3-Coder-Next",
        repo_id="Qwen/Qwen3-Coder-Next",
        local_dir_name="Qwen3-Coder-Next",
        revision="a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb",
        recommended_config="tests/qcn-k4v4-hqq4-int4-benchmark.conf",
        notes="Current production coding model.",
    ),
    SupportedHFModel(
        key="step37",
        display_name="Step-3.7-Flash",
        repo_id="stepfun-ai/Step-3.7-Flash",
        local_dir_name="Step-3.7-Flash",
        revision="5f6244077ac62e04eec3f320501ff8c2b293373a",
        recommended_config="tests/step37-flash-4-4-hqq4-k4v4-a16.conf",
        notes="Validated StepFun sparse MoE target with HQQ4 attention and k4v4 KV.",
    ),
    SupportedHFModel(
        key="dsv4",
        display_name="DeepSeek-V4-Flash",
        repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
        local_dir_name="deepseek-ai/DeepSeek-V4-Flash-0731",
        revision="9e165c30e2704aec5d9d593cce3eebd58bbef1cb",
        recommended_config="testconfigs/deepseek-v4-flash-0731-4-4-a16.conf",
        notes=(
            "Quality-gated DeepSeek-V4-Flash-0731 target with INT4 experts; "
            "fresh launcher selections default to HQQ6 attention and Native cache."
        ),
    ),
    SupportedHFModel(
        key="nemotron-nano",
        display_name="NVIDIA Nemotron-3 Nano 30B A3B",
        repo_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        local_dir_name="NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        revision="cbd3fa9f933d55ef16a84236559f4ee2a0526848",
        recommended_config="tests/nemotron-nano-4-4-hqq4-k4v4-a16.conf",
        notes="Validated Nemotron-H Nano target with HQQ4 attention and k4v4 KV.",
    ),
    SupportedHFModel(
        key="nemotron-super",
        display_name="NVIDIA Nemotron-3 Super 120B A12B",
        repo_id="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        local_dir_name="NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        revision="d51eab0d1f979ebc26b546e634a04f450d99158e",
        recommended_config="tests/nemotron-super-4-4-hqq4-k4v4-a16.conf",
        notes="Validated Nemotron-H Super target with HQQ4 attention and k4v4 KV.",
    ),
    SupportedHFModel(
        key="qwen36-35b",
        display_name="Qwen3.6-35B-A3B",
        repo_id="Qwen/Qwen3.6-35B-A3B",
        local_dir_name="Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        recommended_config="tests/qwen36-35b-5090-hqq4-k4v4-benchmark.conf",
        notes="Qwen 35B-class MoE target with canonical HQQ8 heatmap reused across HQQ/KV profiles.",
    ),
    SupportedHFModel(
        key="ornith35",
        display_name="Ornith-1.0-35B",
        repo_id="deepreinforce-ai/Ornith-1.0-35B",
        local_dir_name="deepreinforce-ai/Ornith-1.0-35B",
        revision="5df2ed3f675c7beaa490328cc70bb573b65fb660",
        recommended_config="tests/ornith35-stats-hqq6-k6v6.conf",
        notes="Validated Ornith 35B-class Qwen3.5-MoE target with HQQ6 attention and k6v6 KV.",
    ),
    SupportedHFModel(
        key="ornith397",
        display_name="Ornith-1.0-397B",
        repo_id="deepreinforce-ai/Ornith-1.0-397B",
        local_dir_name="deepreinforce-ai/Ornith-1.0-397B",
        revision="5e3e761811e804c295c1d3c0ce68b21da6154209",
        recommended_config="tests/ornith397-stats-hqq6-k6v6.conf",
        notes="Validated Ornith 397B-class Qwen3.5-MoE target with HQQ6 attention and k6v6 KV.",
    ),
    SupportedHFModel(
        key="qwen35-35b",
        display_name="Qwen3.5-35B-A3B",
        repo_id="Qwen/Qwen3.5-35B-A3B",
        local_dir_name="Qwen3.5-35B-A3B",
        revision="b1fc3d59ae0ab1e4279e04a8dd0fc4dc361fc2b6",
        recommended_config="tests/q35b-4-4-hqq6-k6v6-diagnostic.conf",
        notes="Validated Qwen 35B-class MoE target.",
    ),
    SupportedHFModel(
        key="qwen35-122b",
        display_name="Qwen3.5-122B-A10B",
        repo_id="Qwen/Qwen3.5-122B-A10B",
        local_dir_name="Qwen3.5-122B-A10B",
        revision="b000b2eb18a7f4cdf3153c4215842da339e09d99",
        recommended_config="tests/q122b-k4v4-hqq6-int4-benchmark.conf",
        notes="Large Qwen MoE target for multi-GPU/high-memory runs.",
    ),
    SupportedHFModel(
        key="qwen3-235b",
        display_name="Qwen3-235B-A22B",
        repo_id="Qwen/Qwen3-235B-A22B",
        local_dir_name="Qwen3-235B-A22B",
        revision="8efa61729e24bd65b1d152b5ab5409052aa80e65",
        recommended_config="tests/q235-k4v4-hqq6-int4-benchmark.conf",
        notes="Large Qwen MoE target.",
    ),
    SupportedHFModel(
        key="gemma4-26b-a4b-it",
        display_name="Gemma4 26B A4B IT",
        repo_id="google/gemma-4-26b-a4b-it",
        local_dir_name="gemma-4-26b-a4b-it",
        revision="6e6f6edea8c52db2094dca3086e4b963a0034dfc",
        recommended_config="tests/gemma-4-4-k6v6-a16.conf",
        notes="Gemma4 text plus lazy image path; non-ring k6v6 is the validated fast text mode.",
    ),
)


@dataclass
class HFModelCandidate:
    repo_id: str
    pipeline_tag: str
    tags: List[str]
    gated: Any
    private: bool
    downloads: int
    likes: int
    last_modified: str
    safetensors_params: int
    safetensors_total_bytes: int
    selected_bytes: int
    selected_file_count: int
    selected_files: List[str]
    safetensors_file_count: int
    summary: str
    display_name: str = ""
    local_dir_name: str = ""
    revision: str = ""
    recommended_config: str = ""
    support_notes: str = ""
    supported_download: bool = False

    @property
    def has_safetensors(self) -> bool:
        return self.safetensors_params > 0 or self.safetensors_file_count > 0

    @property
    def int4_payload_bytes(self) -> int:
        if self.safetensors_params > 0:
            return int(self.safetensors_params * 0.5)
        if self.safetensors_total_bytes > 0:
            return int(self.safetensors_total_bytes / 4)
        if self.selected_bytes > 0:
            return int(self.selected_bytes / 4)
        return 0

    @property
    def compatibility(self) -> str:
        if self.supported_download:
            if not self.has_safetensors:
                return "unsupported: selected Krasis model has no safetensors metadata"
            return "supported by Krasis"
        if not self.has_safetensors:
            return "unsupported: no safetensors metadata"
        tags = {t.lower() for t in self.tags}
        if "transformers" not in tags:
            return "unsupported: not a Transformers repo"
        if tags.intersection(UNSUPPORTED_REPO_TAGS):
            return "unsupported: non-native or quantized repo"
        if any(tag.startswith("base_model:quantized:") for tag in tags):
            return "unsupported: quantized conversion repo"
        repo_lower = self.repo_id.lower()
        if any(marker in repo_lower for marker in UNSUPPORTED_REPO_NAME_MARKERS):
            return "unsupported: non-native or quantized repo"
        if self.pipeline_tag not in ALLOWED_PIPELINE_TAGS:
            return "unknown task"
        return "likely"

    @property
    def is_krasis_candidate(self) -> bool:
        return self.compatibility in ("likely", "supported by Krasis")


def _require_hf():
    try:
        from huggingface_hub import HfApi, get_token, login, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for the model downloader. "
            "Install a current Krasis wheel or run ./dev build."
        ) from exc
    try:
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    except ImportError:
        try:
            from huggingface_hub.utils import HfHubHTTPError
            from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
        except ImportError as exc:
            raise RuntimeError(
                "The installed huggingface_hub package is too old for the model downloader. "
                "Upgrade Hugging Face Hub in the Krasis environment."
            ) from exc
    return HfApi, get_token, login, snapshot_download, GatedRepoError, HfHubHTTPError, RepositoryNotFoundError


def supported_models() -> List[SupportedHFModel]:
    return list(SUPPORTED_HF_MODELS)


def supported_model_spec(value: str) -> SupportedHFModel:
    clean = value.strip().lower()
    for spec in SUPPORTED_HF_MODELS:
        if clean in {
            spec.key.lower(),
            spec.display_name.lower(),
            spec.repo_id.lower(),
            spec.local_dir_name.lower(),
        }:
            return spec
    raise ValueError(f"unsupported Krasis model: {value}")


def _apply_supported_spec(candidate: HFModelCandidate, spec: SupportedHFModel) -> HFModelCandidate:
    candidate.display_name = spec.display_name
    candidate.local_dir_name = spec.local_dir_name
    candidate.revision = spec.revision
    candidate.recommended_config = spec.recommended_config
    candidate.support_notes = spec.notes
    candidate.supported_download = True
    return candidate


def _save_hf_token(token: str) -> None:
    try:
        from huggingface_hub import HfFolder

        HfFolder.save_token(token)
        return
    except (ImportError, AttributeError):
        pass

    try:
        from huggingface_hub._login import _save_token, _set_active_token

        token_name = "krasis"
        _save_token(token=token, token_name=token_name)
        _set_active_token(token_name=token_name, add_to_git_credential=False)
        return
    except Exception:
        pass

    try:
        from huggingface_hub.constants import HF_TOKEN_PATH

        token_path = Path(HF_TOKEN_PATH)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(token, encoding="utf-8")
    except Exception as exc:
        raise RuntimeError("Hugging Face token was validated but could not be saved.") from exc


def parse_hf_repo_id(value: str) -> str:
    """Parse a Hugging Face model URL or repo id into `namespace/name`."""
    text = value.strip()
    if not text:
        raise ValueError("empty Hugging Face repo")

    if "://" in text:
        parsed = urlparse(text)
        host = parsed.netloc.lower()
        if host not in ("huggingface.co", "www.huggingface.co"):
            raise ValueError("not a huggingface.co URL")
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            raise ValueError("URL does not include a model repo id")
        if parts[0] in ("models", "spaces", "datasets"):
            parts = parts[1:]
        repo_parts: List[str] = []
        for part in parts:
            if part in ("tree", "blob", "resolve"):
                break
            repo_parts.append(part)
        if len(repo_parts) < 2:
            raise ValueError("URL does not include a model repo id")
        return "/".join(repo_parts[:2])

    if text.startswith("huggingface.co/"):
        return parse_hf_repo_id(f"https://{text}")

    text = text.strip("/")
    if len(text.split("/")) != 2:
        raise ValueError("use a Hugging Face repo id like Qwen/Qwen3-Coder-Next")
    return text


def _token_arg() -> Optional[bool]:
    _HfApi, get_token, _login, _snapshot_download, *_ = _require_hf()
    return True if get_token() else None


def hf_auth_status() -> Dict[str, Any]:
    HfApi, get_token, _login, _snapshot_download, *_ = _require_hf()
    token = get_token()
    if not token:
        return {"logged_in": False, "user": ""}
    try:
        info = HfApi().whoami(token=True)
        return {"logged_in": True, "user": str(info.get("name") or info.get("email") or "authenticated")}
    except Exception as exc:
        return {"logged_in": True, "user": "", "error": str(exc)}


def hf_login(token: str) -> Dict[str, Any]:
    HfApi, get_token, login, _snapshot_download, *_ = _require_hf()
    clean_token = token.strip()
    info = HfApi().whoami(token=clean_token)
    kwargs: Dict[str, Any] = {"token": clean_token, "add_to_git_credential": False}
    try:
        params = inspect.signature(login).parameters
    except (TypeError, ValueError):
        params = {}
    if "new_session" in params:
        kwargs["new_session"] = True
    login(**kwargs)
    if get_token() != clean_token:
        _save_hf_token(clean_token)
    return {"logged_in": True, "user": str(info.get("name") or info.get("email") or "authenticated")}


def _safetensors_param_total(info: Any) -> int:
    st = getattr(info, "safetensors", None)
    if st is None:
        return 0
    total = int(getattr(st, "total", 0) or 0)
    if total > 0:
        return total
    params = getattr(st, "parameters", None) or {}
    try:
        return int(sum(int(v) for v in params.values()))
    except Exception:
        return 0


def _file_size(sibling: Any) -> int:
    size = getattr(sibling, "size", None)
    if isinstance(size, int) and size > 0:
        return size
    lfs = getattr(sibling, "lfs", None)
    lfs_size = getattr(lfs, "size", None)
    if isinstance(lfs_size, int) and lfs_size > 0:
        return lfs_size
    return 0


def _matches_any(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


def selected_download_files(info: Any) -> List[Any]:
    files = []
    for sibling in getattr(info, "siblings", None) or []:
        name = getattr(sibling, "rfilename", "")
        if not name:
            continue
        if _matches_any(name, KRASIS_HF_IGNORE_PATTERNS):
            continue
        if _matches_any(name, KRASIS_HF_ALLOW_PATTERNS):
            files.append(sibling)
    return files


def _format_last_modified(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if value:
        return str(value)[:10]
    return "unknown"


def _summary_from_info(info: Any) -> str:
    tags = [str(t) for t in (getattr(info, "tags", None) or [])]
    useful = []
    skip_prefixes = ("license:", "arxiv:", "base_model:", "region:")
    for tag in tags:
        low = tag.lower()
        if low in ("transformers", "safetensors", "text-generation", "conversational"):
            continue
        if low.startswith(skip_prefixes):
            continue
        useful.append(tag)
        if len(useful) >= 4:
            break
    pipeline = getattr(info, "pipeline_tag", None) or "model"
    if useful:
        return f"{pipeline}; " + ", ".join(useful)
    return str(pipeline)


def candidate_from_info(info: Any, *, include_files: bool = False) -> HFModelCandidate:
    selected = selected_download_files(info) if include_files else []
    selected_bytes = sum(_file_size(s) for s in selected)
    safetensors_files = [s for s in selected if str(getattr(s, "rfilename", "")).endswith(".safetensors")]
    safetensors_bytes = sum(_file_size(s) for s in safetensors_files)
    params = _safetensors_param_total(info)
    if selected_bytes == 0 and params > 0:
        selected_bytes = params * 2
    if safetensors_bytes == 0 and params > 0:
        safetensors_bytes = params * 2
    return HFModelCandidate(
        repo_id=str(getattr(info, "id", "") or getattr(info, "modelId", "")),
        pipeline_tag=str(getattr(info, "pipeline_tag", "") or ""),
        tags=[str(t) for t in (getattr(info, "tags", None) or [])],
        gated=getattr(info, "gated", False),
        private=bool(getattr(info, "private", False) or False),
        downloads=int(getattr(info, "downloads", 0) or 0),
        likes=int(getattr(info, "likes", 0) or 0),
        last_modified=_format_last_modified(getattr(info, "last_modified", None)),
        safetensors_params=params,
        safetensors_total_bytes=safetensors_bytes,
        selected_bytes=selected_bytes,
        selected_file_count=len(selected),
        selected_files=[str(getattr(s, "rfilename", "")) for s in selected],
        safetensors_file_count=len(safetensors_files),
        summary=_summary_from_info(info),
    )


def search_models(query: str, *, limit: int = 20) -> List[HFModelCandidate]:
    HfApi, *_ = _require_hf()
    api = HfApi()
    token = _token_arg()
    search_text = query.strip()
    direct_candidate: Optional[HFModelCandidate] = None
    try:
        direct_repo_id = parse_hf_repo_id(search_text)
    except ValueError:
        direct_repo_id = ""
    if direct_repo_id:
        search_text = direct_repo_id.split("/", 1)[1]
        try:
            direct_info = api.model_info(direct_repo_id, token=token)
            direct_candidate = candidate_from_info(direct_info, include_files=False)
        except Exception:
            direct_candidate = None

    raw_limit = max(limit * 5, 100)
    models = list(api.list_models(
        search=search_text,
        filter="safetensors",
        limit=raw_limit,
        expand=[
            "downloads",
            "gated",
            "lastModified",
            "likes",
            "pipeline_tag",
            "private",
            "safetensors",
            "tags",
        ],
        token=token,
    ))
    direct_candidates = []
    candidates = []
    seen = set()
    if direct_candidate and direct_candidate.is_krasis_candidate:
        direct_candidates.append(direct_candidate)
        seen.add(direct_candidate.repo_id)
    for info in models:
        candidate = candidate_from_info(info, include_files=False)
        if candidate.repo_id in seen or not candidate.is_krasis_candidate:
            continue
        candidates.append(candidate)
        seen.add(candidate.repo_id)
    candidates.sort(key=lambda c: (-c.downloads, c.repo_id.lower()))
    return (direct_candidates + candidates)[:limit]


def get_model_details(repo_or_url: str) -> HFModelCandidate:
    HfApi, *_ = _require_hf()
    repo_id = parse_hf_repo_id(repo_or_url)
    info = HfApi().model_info(repo_id, files_metadata=True, token=_token_arg())
    return candidate_from_info(info, include_files=True)


def get_supported_model_details(key_or_repo: str) -> HFModelCandidate:
    HfApi, *_ = _require_hf()
    spec = supported_model_spec(key_or_repo)
    info = HfApi().model_info(
        spec.repo_id,
        revision=spec.revision,
        files_metadata=True,
        token=_token_arg(),
    )
    return _apply_supported_spec(candidate_from_info(info, include_files=True), spec)


def destination_for_repo(models_dir: str, repo_id: str) -> str:
    org, name = repo_id.split("/", 1)
    return os.path.join(models_dir, org, name)


def destination_for_supported_model(models_dir: str, model: Any) -> str:
    local_dir_name = getattr(model, "local_dir_name", "")
    if not local_dir_name:
        local_dir_name = supported_model_spec(str(getattr(model, "repo_id", model))).local_dir_name
    return os.path.join(models_dir, local_dir_name)


def format_bytes(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    if unit in ("B", "KB", "MB"):
        return f"{value:.0f} {unit}"
    return f"{value:.1f} {unit}"


def download_model(repo_id: str, local_dir: str, *, revision: Optional[str] = None, max_workers: int = 8) -> str:
    _HfApi, _get_token, _login, snapshot_download, *_ = _require_hf()
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass
    os.makedirs(local_dir, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=KRASIS_HF_ALLOW_PATTERNS,
        ignore_patterns=KRASIS_HF_IGNORE_PATTERNS,
        max_workers=max_workers,
        revision=revision,
        token=_token_arg(),
    )


def count_selected_local_bytes(local_dir: str, selected_names: Iterable[str]) -> int:
    total = 0
    selected = set(selected_names)
    for name in selected:
        path = os.path.join(local_dir, name)
        if os.path.isfile(path):
            try:
                total += os.path.getsize(path)
            except OSError:
                pass
    return total


def validate_local_model(local_dir: str) -> List[str]:
    issues: List[str] = []
    if not os.path.isfile(os.path.join(local_dir, "config.json")):
        issues.append("missing config.json")
    has_safetensors = False
    for root, _dirs, files in os.walk(local_dir):
        if any(name.endswith(".safetensors") for name in files):
            has_safetensors = True
            break
    if not has_safetensors:
        issues.append("missing .safetensors weights")
    tokenizer_files = ("tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt")
    if not any(os.path.isfile(os.path.join(local_dir, name)) for name in tokenizer_files):
        issues.append("missing tokenizer files")
    return issues
