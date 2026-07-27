"""HF tokenizer wrapper with chat template support."""

import json
import logging
import os
from typing import Dict, List, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def _load_hf_tokenizer(model_path: str, cfg: dict, tokenizer_kwargs: dict):
    """Load the tokenizer declared by the checkpoint without changing its backend."""
    tokenizer_class = cfg.get("tokenizer_class")
    if tokenizer_class != "TokenizersBackend":
        return AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)

    backend = cfg.get("backend")
    if backend != "tokenizers":
        raise ValueError(
            "Tokenizer config declares tokenizer_class=TokenizersBackend but "
            f"backend={backend!r}; only the explicit 'tokenizers' backend is supported"
        )

    tokenizer_file = os.path.join(model_path, "tokenizer.json")
    if not os.path.isfile(tokenizer_file):
        raise FileNotFoundError(
            "Tokenizer config declares the TokenizersBackend class but the "
            f"required local tokenizer.json is missing: {tokenizer_file}"
        )

    logger.info(
        "Loading tokenizer_class=TokenizersBackend through the equivalent "
        "PreTrainedTokenizerFast compatibility class"
    )
    return PreTrainedTokenizerFast.from_pretrained(
        model_path, **tokenizer_kwargs
    )


class Tokenizer:
    """Wraps HuggingFace tokenizer with chat template formatting."""

    # DeepSeek-V2/V2-Lite chat format (plain text User:/Assistant:)
    _DEEPSEEK_CHAT_TEMPLATE = (
        "{% if not add_generation_prompt is defined %}"
        "{% set add_generation_prompt = false %}{% endif %}"
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ 'User: ' + message['content'] + '\n\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ 'Assistant: ' + message['content'] + eos_token }}"
        "{% elif message['role'] == 'system' %}"
        "{{ message['content'] + '\n\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ 'Assistant:' }}"
        "{% endif %}"
    )

    # DeepSeek-VL2 chat format (special <|User|>/<|Assistant|> tokens)
    _DEEPSEEK_VL2_CHAT_TEMPLATE = (
        "{% if not add_generation_prompt is defined %}"
        "{% set add_generation_prompt = false %}{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<|User|>' + message['content'] + '\n\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|Assistant|>' + message['content'] + eos_token }}"
        "{% elif message['role'] == 'system' %}"
        "{{ message['content'] + '\n\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|Assistant|>' }}"
        "{% endif %}"
    )

    def __init__(self, model_path: str):
        tokenizer_kwargs = {"trust_remote_code": True}
        tokenizer_config = os.path.join(model_path, "tokenizer_config.json")
        cfg = {}
        try:
            with open(tokenizer_config, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if isinstance(cfg.get("extra_special_tokens"), list):
                tokenizer_kwargs["extra_special_tokens"] = {}
                logger.info(
                    "Tokenizer config extra_special_tokens is a list; overriding to empty dict for text-only loading"
                )
        except FileNotFoundError:
            pass
        self.tokenizer = _load_hf_tokenizer(
            model_path, cfg, tokenizer_kwargs
        )
        if not getattr(self.tokenizer, "chat_template", None):
            template_path = os.path.join(model_path, "chat_template.jinja")
            if os.path.isfile(template_path):
                with open(template_path, "r", encoding="utf-8") as f:
                    self.tokenizer.chat_template = f.read()
                logger.info("Loaded chat template from %s", template_path)
        # Set fallback chat template if model doesn't have one
        if not getattr(self.tokenizer, "chat_template", None):
            # Detect VL2-style models: they have <|User|> as a special token
            vocab = self.tokenizer.get_vocab()
            if "<|User|>" in vocab and "<|Assistant|>" in vocab:
                self.tokenizer.chat_template = self._DEEPSEEK_VL2_CHAT_TEMPLATE
                logger.info("No chat template found — using DeepSeek-VL2 format (<|User|>/<|Assistant|> tokens)")
            else:
                self.tokenizer.chat_template = self._DEEPSEEK_CHAT_TEMPLATE
                logger.info("No chat template found — using DeepSeek format fallback")
        template = getattr(self.tokenizer, "chat_template", "") or ""
        self._template_has_thinking = "<think>" in template and "</think>" in template
        logger.info(
            "Tokenizer loaded: vocab=%d, eos=%d, bos=%d",
            self.tokenizer.vocab_size,
            self.tokenizer.eos_token_id or -1,
            self.tokenizer.bos_token_id or -1,
        )

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.tokenizer.bos_token_id

    @property
    def chat_template_supports_enable_thinking(self) -> bool:
        template = getattr(self.tokenizer, "chat_template", "") or ""
        return "enable_thinking" in template

    @staticmethod
    def _close_initial_think_block(rendered: str) -> str:
        suffix = "<|im_start|>assistant\n<think>\n"
        if rendered.endswith(suffix):
            return rendered + "\n</think>\n\n"
        return rendered

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> List[int]:
        """Format messages using the model's chat template and tokenize."""
        kwargs = dict(kwargs)
        requested_enable_thinking = kwargs.get("enable_thinking")
        if requested_enable_thinking is not None and not self.chat_template_supports_enable_thinking:
            if requested_enable_thinking and not self._template_has_thinking:
                raise ValueError(
                    "Model chat template does not support enable_thinking; "
                    "refuse to silently ignore an explicit thinking request"
                )
            kwargs.pop("enable_thinking", None)
        force_close_think = (
            requested_enable_thinking is False
            and self._template_has_thinking
            and add_generation_prompt
        )
        try:
            result = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=not force_close_think,
                **kwargs,
            )
        except (TypeError, Exception) as e:
            # Some templates don't accept extra kwargs (enable_thinking, etc.)
            if kwargs:
                logger.debug("Chat template failed with kwargs %s, retrying without: %s", list(kwargs), e)
                result = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                    tokenize=not force_close_think,
                )
            else:
                raise
        if force_close_think:
            return self.tokenizer.encode(
                self._close_initial_think_block(result),
                add_special_tokens=False,
            )
        return self._ensure_int_list(result)

    def _ensure_int_list(self, result) -> List[int]:
        """Force result to List[int] regardless of what transformers returned."""
        if isinstance(result, str):
            return self.tokenizer.encode(result, add_special_tokens=False)
        # transformers v5: apply_chat_template returns BatchEncoding
        # (has __getitem__ for "input_ids" but may not pass isinstance dict)
        try:
            ids = result["input_ids"]
            # Could be nested: [[1,2,3]] for batch or [1,2,3]
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            return [int(t) for t in ids]
        except (KeyError, TypeError, IndexError):
            pass
        if isinstance(result, list):
            if result and isinstance(result[0], str):
                return self.tokenizer.encode(
                    "".join(result), add_special_tokens=False
                )
            return [int(t) for t in result]
        # numpy array or tensor
        return [int(t) for t in result]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode_incremental(self, token_id: int) -> str:
        """Decode a single token. Returns the text piece."""
        return self.tokenizer.decode([token_id], skip_special_tokens=False)
