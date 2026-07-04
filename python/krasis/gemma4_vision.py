"""Minimal Gemma4 image encoder for Krasis multimodal prefill.

This mirrors the upstream Transformers Gemma4 vision tower and image soft-token
projection, but keeps only the image path needed to produce text-model
``inputs_embeds``. Audio/video and the language model stay outside this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


@dataclass
class Gemma4VisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    hidden_activation: str = "gelu_pytorch_tanh"
    attention_dropout: float = 0.0
    patch_size: int = 16
    pooling_kernel_size: int = 3
    position_embedding_size: int = 10240
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1.0e-6
    rope_theta: float = 100.0
    standardize: bool = True
    use_clipped_linears: bool = False
    default_output_length: int = 280

    @classmethod
    def from_dict(cls, raw: dict) -> "Gemma4VisionConfig":
        rope = raw.get("rope_parameters") or {}
        return cls(
            hidden_size=int(raw.get("hidden_size", cls.hidden_size)),
            intermediate_size=int(raw.get("intermediate_size", cls.intermediate_size)),
            num_hidden_layers=int(raw.get("num_hidden_layers", cls.num_hidden_layers)),
            num_attention_heads=int(raw.get("num_attention_heads", cls.num_attention_heads)),
            num_key_value_heads=int(raw.get("num_key_value_heads", cls.num_key_value_heads)),
            head_dim=int(raw.get("head_dim", raw.get("hidden_size", cls.hidden_size) // raw.get("num_attention_heads", cls.num_attention_heads))),
            hidden_activation=str(raw.get("hidden_activation", cls.hidden_activation)),
            attention_dropout=float(raw.get("attention_dropout", cls.attention_dropout)),
            patch_size=int(raw.get("patch_size", cls.patch_size)),
            pooling_kernel_size=int(raw.get("pooling_kernel_size", cls.pooling_kernel_size)),
            position_embedding_size=int(raw.get("position_embedding_size", cls.position_embedding_size)),
            max_position_embeddings=int(raw.get("max_position_embeddings", cls.max_position_embeddings)),
            rms_norm_eps=float(raw.get("rms_norm_eps", cls.rms_norm_eps)),
            rope_theta=float(rope.get("rope_theta", raw.get("rope_theta", cls.rope_theta))),
            standardize=bool(raw.get("standardize", cls.standardize)),
            use_clipped_linears=bool(raw.get("use_clipped_linears", cls.use_clipped_linears)),
            default_output_length=int(raw.get("default_output_length", cls.default_output_length)),
        )


class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-6, with_scale: bool = True):
        super().__init__()
        self.eps = float(eps)
        self.with_scale = bool(with_scale)
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(int(dim)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = hidden_states.float() * torch.pow(
            hidden_states.float().pow(2).mean(-1, keepdim=True) + self.eps,
            -0.5,
        )
        if self.with_scale:
            normed = normed * self.weight.float()
        return normed.to(dtype=hidden_states.dtype)


class Gemma4ClippableLinear(nn.Module):
    def __init__(self, config: Gemma4VisionConfig, in_features: int, out_features: int):
        super().__init__()
        self.use_clipped_linears = bool(config.use_clipped_linears)
        self.linear = nn.Linear(int(in_features), int(out_features), bias=False)
        if self.use_clipped_linears:
            self.register_buffer("input_min", torch.tensor(-float("inf")))
            self.register_buffer("input_max", torch.tensor(float("inf")))
            self.register_buffer("output_min", torch.tensor(-float("inf")))
            self.register_buffer("output_max", torch.tensor(float("inf")))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)
        hidden_states = self.linear(hidden_states)
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)
        return hidden_states


class Gemma4VisionPatchEmbedder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.position_embedding_size = config.position_embedding_size
        self.input_proj = nn.Linear(3 * self.patch_size * self.patch_size, self.hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, self.position_embedding_size, self.hidden_size)
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        pixel_values = 2 * (pixel_values - 0.5)
        weight = getattr(self.input_proj, "weight", None)
        if weight is not None and weight.dtype.is_floating_point:
            pixel_values = pixel_values.to(weight.dtype)
        else:
            pixel_values = pixel_values.to(torch.bfloat16)
        hidden_states = self.input_proj(pixel_values)
        clamped_positions = pixel_position_ids.clamp(min=0)
        x_emb = F.embedding(clamped_positions[..., 0], self.position_embedding_table[0])
        y_emb = F.embedding(clamped_positions[..., 1], self.position_embedding_table[1])
        pos = x_emb + y_emb
        pos = torch.where(padding_positions.unsqueeze(-1), 0.0, pos)
        return hidden_states + pos


class Gemma4VisionPooler(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.root_hidden_size = self.hidden_size ** 0.5

    def _avg_pool_by_positions(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_seq_len = int(hidden_states.shape[1])
        k = int((input_seq_len // int(length)) ** 0.5)
        k_squared = k * k
        if k_squared * int(length) != input_seq_len:
            raise ValueError(
                f"Cannot pool {tuple(hidden_states.shape)} to {length}: k={k} is incompatible"
            )
        clamped_positions = pixel_position_ids.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = F.one_hot(kernel_idxs.long(), int(length)).float() / float(k_squared)
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(output_length) > int(hidden_states.shape[1]):
            raise ValueError(
                f"Cannot output {output_length} soft tokens from {hidden_states.shape[1]} patches"
            )
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        if int(hidden_states.shape[1]) != int(output_length):
            hidden_states, padding_positions = self._avg_pool_by_positions(
                hidden_states,
                pixel_position_ids,
                int(output_length),
            )
        hidden_states = hidden_states.float() * self.root_hidden_size
        return hidden_states, padding_positions


class Gemma4VisionRotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        spatial_dim = int(config.head_dim) // 2
        inv_freq = 1.0 / (
            float(config.rope_theta)
            ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32) / float(spatial_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        all_cos = []
        all_sin = []
        inv = self.inv_freq.to(device=x.device)
        for dim in range(2):
            pos = position_ids[:, :, dim].to(device=x.device, dtype=torch.float32)
            freqs = pos[:, :, None] * inv[None, None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
            all_cos.append(emb.cos())
            all_sin.append(emb.sin())
        return torch.cat(all_cos, dim=-1).to(x.dtype), torch.cat(all_sin, dim=-1).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (x * cos) + (_rotate_half(x) * sin)


def _apply_multidimensional_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    ndim = int(position_ids.shape[-1])
    channels = int(x.shape[-1])
    per_dim = 2 * (channels // (2 * ndim))
    if per_dim <= 0:
        raise ValueError(f"Invalid Gemma4 RoPE dimensions: channels={channels}, ndim={ndim}")
    parts = []
    for x_part, c_part, s_part in zip(
        torch.split(x, [per_dim] * ndim, dim=-1),
        torch.split(cos, [per_dim] * ndim, dim=-1),
        torch.split(sin, [per_dim] * ndim, dim=-1),
    ):
        parts.append(_apply_rope(x_part, c_part, s_part))
    return torch.cat(parts, dim=-1)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if int(n_rep) == 1:
        return hidden_states
    bsz, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, int(n_rep), seq_len, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * int(n_rep), seq_len, head_dim)


class Gemma4VisionAttention(nn.Module):
    def __init__(self, config: Gemma4VisionConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.head_dim = int(config.head_dim)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_proj = Gemma4ClippableLinear(config, config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = Gemma4ClippableLinear(config, config.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = Gemma4ClippableLinear(config, config.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = Gemma4ClippableLinear(config, self.num_heads * self.head_dim, config.hidden_size)
        self.q_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len = hidden_states.shape[:2]
        cos, sin = position_embeddings
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        q = _apply_multidimensional_rope(q, cos, sin, position_ids).transpose(1, 2)

        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        k = self.k_norm(k)
        k = _apply_multidimensional_rope(k, cos, sin, position_ids).transpose(1, 2)

        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_norm(v).transpose(1, 2)
        k = _repeat_kv(k, self.num_key_value_groups)
        v = _repeat_kv(v, self.num_key_value_groups)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
            scale=1.0,
        )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class Gemma4VisionMLP(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.gate_proj = Gemma4ClippableLinear(config, config.hidden_size, config.intermediate_size)
        self.up_proj = Gemma4ClippableLinear(config, config.hidden_size, config.intermediate_size)
        self.down_proj = Gemma4ClippableLinear(config, config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class Gemma4VisionEncoderLayer(nn.Module):
    def __init__(self, config: Gemma4VisionConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Gemma4VisionAttention(config, layer_idx)
        self.mlp = Gemma4VisionMLP(config)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask, position_ids)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class Gemma4VisionEncoder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.rotary_emb = Gemma4VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [Gemma4VisionEncoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, pixel_position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, attention_mask, pixel_position_ids)
        return hidden_states


class Gemma4VisionModel(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedder = Gemma4VisionPatchEmbedder(config)
        self.encoder = Gemma4VisionEncoder(config)
        self.pooler = Gemma4VisionPooler(config)
        if config.standardize:
            self.register_buffer("std_bias", torch.empty(config.hidden_size))
            self.register_buffer("std_scale", torch.empty(config.hidden_size))

    def forward(self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor) -> torch.Tensor:
        pooling = int(self.config.pooling_kernel_size)
        output_length = int(pixel_values.shape[-2]) // (pooling * pooling)
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        hidden_states = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
        hidden_states = self.encoder(hidden_states, ~padding_positions, pixel_position_ids)
        hidden_states, pooler_mask = self.pooler(
            hidden_states,
            pixel_position_ids,
            padding_positions,
            output_length,
        )
        hidden_states = hidden_states[pooler_mask]
        if self.config.standardize:
            hidden_states = (hidden_states - self.std_bias.float()) * self.std_scale.float()
        return hidden_states.to(dtype=pixel_values.dtype)


class Gemma4MultimodalEmbedder(nn.Module):
    def __init__(self, vision_config: Gemma4VisionConfig, text_hidden_size: int):
        super().__init__()
        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            vision_config.hidden_size,
            eps=vision_config.rms_norm_eps,
            with_scale=False,
        )
        self.embedding_projection = nn.Linear(vision_config.hidden_size, int(text_hidden_size), bias=False)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return self.embedding_projection(self.embedding_pre_projection_norm(inputs_embeds))


def _target_size(height: int, width: int, patch_size: int, max_patches: int, pooling_kernel_size: int) -> tuple[int, int]:
    target_px = int(max_patches) * int(patch_size) * int(patch_size)
    factor = math.sqrt(float(target_px) / float(height * width))
    side_mult = int(pooling_kernel_size) * int(patch_size)
    target_h = int(math.floor((factor * height) / side_mult)) * side_mult
    target_w = int(math.floor((factor * width) / side_mult)) * side_mult
    max_side = (int(max_patches) // (int(pooling_kernel_size) ** 2)) * side_mult
    if target_h == 0 and target_w == 0:
        raise ValueError("Gemma4 image resize produced a 0x0 target")
    if target_h == 0:
        target_h = side_mult
        target_w = min(int(math.floor(width / height)) * side_mult, max_side)
    elif target_w == 0:
        target_w = side_mult
        target_h = min(int(math.floor(height / width)) * side_mult, max_side)
    return target_h, target_w


def _patchify(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    channels, height, width = image.shape
    patch_h = height // patch_size
    patch_w = width // patch_size
    patches = image.reshape(channels, patch_h, patch_size, patch_w, patch_size)
    patches = patches.permute(1, 3, 2, 4, 0)
    return patches.reshape(patch_h * patch_w, patch_size * patch_size * channels)


class Gemma4ImagePreprocessor:
    def __init__(
        self,
        *,
        patch_size: int = 16,
        pooling_kernel_size: int = 3,
        max_soft_tokens: int = 280,
        rescale_factor: float = 1.0 / 255.0,
    ):
        self.patch_size = int(patch_size)
        self.pooling_kernel_size = int(pooling_kernel_size)
        self.max_soft_tokens = int(max_soft_tokens)
        self.rescale_factor = float(rescale_factor)
        self.max_patches = self.max_soft_tokens * self.pooling_kernel_size * self.pooling_kernel_size

    def __call__(self, images: Iterable[Image.Image]) -> dict:
        pixel_values = []
        position_ids = []
        num_soft_tokens = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise TypeError(f"Gemma4 image must be a PIL Image, got {type(image)!r}")
            image = image.convert("RGB")
            target_h, target_w = _target_size(
                image.height,
                image.width,
                self.patch_size,
                self.max_patches,
                self.pooling_kernel_size,
            )
            if (target_w, target_h) != image.size:
                image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
            arr = np.asarray(image, dtype=np.float32) * self.rescale_factor
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            patches = _patchify(tensor, self.patch_size)
            patch_h = target_h // self.patch_size
            patch_w = target_w // self.patch_size
            soft = int(patches.shape[0]) // (self.pooling_kernel_size * self.pooling_kernel_size)
            num_soft_tokens.append(soft)

            xs, ys = torch.meshgrid(torch.arange(patch_w), torch.arange(patch_h), indexing="xy")
            pos = torch.stack((xs, ys), dim=-1).reshape(-1, 2).to(torch.long)
            pad = self.max_patches - int(patches.shape[0])
            if pad < 0:
                raise ValueError(f"Gemma4 image produced {patches.shape[0]} patches > {self.max_patches}")
            if pad:
                patches = F.pad(patches, (0, 0, 0, pad))
                pos = F.pad(pos, (0, 0, 0, pad), value=-1)
            pixel_values.append(patches)
            position_ids.append(pos)
        return {
            "pixel_values": torch.stack(pixel_values, dim=0),
            "image_position_ids": torch.stack(position_ids, dim=0),
            "num_soft_tokens_per_image": num_soft_tokens,
        }


__all__ = [
    "Gemma4ImagePreprocessor",
    "Gemma4MultimodalEmbedder",
    "Gemma4VisionConfig",
    "Gemma4VisionModel",
]
