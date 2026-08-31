"""GLM-5.3 image encoder used by Krasis multimodal prefill.

This is the image-only subset of the upstream Transformers GLM-5.3 vision
implementation.  It deliberately excludes video and the language model: the
result is a sequence of text-width image embeddings consumed by the existing
Rust prefill path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class Glm53VisionConfig:
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    out_hidden_size: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    projection_intermediate_size: int
    rms_norm_eps: float
    swiglu_limit: float
    attention_bias: bool
    hidden_act: str

    @classmethod
    def from_dict(cls, raw: dict) -> "Glm53VisionConfig":
        required = (
            "depth",
            "hidden_size",
            "intermediate_size",
            "num_heads",
            "out_hidden_size",
            "patch_size",
            "temporal_patch_size",
            "spatial_merge_size",
            "projection_intermediate_size",
            "rms_norm_eps",
            "swiglu_limit",
            "attention_bias",
            "hidden_act",
        )
        missing = [name for name in required if name not in raw]
        if missing:
            raise ValueError(f"GLM-5.3 vision config is missing {missing}")
        cfg = cls(
            depth=int(raw["depth"]),
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_heads=int(raw["num_heads"]),
            out_hidden_size=int(raw["out_hidden_size"]),
            patch_size=int(raw["patch_size"]),
            temporal_patch_size=int(raw["temporal_patch_size"]),
            spatial_merge_size=int(raw["spatial_merge_size"]),
            projection_intermediate_size=int(raw["projection_intermediate_size"]),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            swiglu_limit=float(raw["swiglu_limit"]),
            attention_bias=bool(raw["attention_bias"]),
            hidden_act=str(raw["hidden_act"]),
        )
        if cfg.hidden_size <= 0 or cfg.num_heads <= 0 or cfg.hidden_size % cfg.num_heads:
            raise ValueError(
                "GLM-5.3 vision hidden_size must be positive and divisible by num_heads"
            )
        if cfg.out_hidden_size <= 0 or cfg.spatial_merge_size <= 0:
            raise ValueError("GLM-5.3 vision output/merge dimensions must be positive")
        if cfg.hidden_act != "silu":
            raise ValueError(f"Unsupported GLM-5.3 vision activation {cfg.hidden_act!r}")
        return cfg


def glm53_smart_resize(
    height: int,
    width: int,
    *,
    temporal_factor: int,
    factor: int,
    min_image_tokens: int,
    max_image_tokens: int,
) -> tuple[int, int]:
    if height <= 0 or width <= 0 or temporal_factor <= 0 or factor <= 0:
        raise ValueError("GLM-5.3 image dimensions and alignment factors must be positive")
    if min_image_tokens <= 0 or max_image_tokens < min_image_tokens:
        raise ValueError("GLM-5.3 image token limits are invalid")

    pixels_per_token = temporal_factor * factor * factor
    min_pixels = min_image_tokens * pixels_per_token
    max_pixels = max_image_tokens * pixels_per_token

    def align(value: int) -> int:
        return math.ceil(value / factor) * factor

    aligned_frames = temporal_factor
    aligned_height = align(height)
    aligned_width = align(width)
    aligned_pixels = aligned_frames * aligned_height * aligned_width
    if aligned_pixels < min_pixels:
        scale = math.sqrt(min_pixels / (temporal_factor * height * width))
        aligned_height = align(max(1, math.ceil(height * scale)))
        aligned_width = align(max(1, math.ceil(width * scale)))
        aligned_pixels = aligned_frames * aligned_height * aligned_width

    if aligned_pixels > max_pixels:
        if max_pixels < aligned_frames * factor * factor:
            raise ValueError(
                f"max image budget {max_pixels} cannot fit one aligned GLM-5.3 patch"
            )
        low, high = 1, height
        best_height, best_width = factor, factor
        while low <= high:
            content_height = (low + high) // 2
            content_width = max(1, math.floor(width * content_height / height))
            candidate_height = align(content_height)
            candidate_width = align(content_width)
            if aligned_frames * candidate_height * candidate_width <= max_pixels:
                best_height, best_width = candidate_height, candidate_width
                low = content_height + 1
            else:
                high = content_height - 1
        aligned_height, aligned_width = best_height, best_width

    return aligned_height, aligned_width


class Glm53ImagePreprocessor:
    def __init__(
        self,
        *,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        min_image_tokens: int,
        max_image_tokens: int,
        image_mean: list[float],
        image_std: list[float],
    ):
        self.patch_size = int(patch_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.merge_size = int(merge_size)
        self.min_image_tokens = int(min_image_tokens)
        self.max_image_tokens = int(max_image_tokens)
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)

    @classmethod
    def from_checkpoint_config(
        cls,
        raw: dict,
        vision_config: Glm53VisionConfig,
    ) -> "Glm53ImagePreprocessor":
        """Build from checkpoint-owned metadata, rejecting incomplete contracts."""
        required = (
            "do_rescale",
            "patch_expand_factor",
            "merge_size",
            "image_mean",
            "image_std",
            "temporal_patch_size",
            "patch_size",
            "min_image_tokens",
            "max_image_tokens",
            "image_processor_type",
        )
        missing = [name for name in required if name not in raw]
        if missing:
            raise ValueError(f"GLM-5.3 image processor config is missing {missing}")
        if raw["image_processor_type"] != "Glm5NextImageProcessor":
            raise ValueError(
                "Unsupported GLM-5.3 image processor type "
                f"{raw['image_processor_type']!r}"
            )
        if raw["do_rescale"] is not True or int(raw["patch_expand_factor"]) != 1:
            raise ValueError(
                "GLM-5.3 image preprocessing requires checkpoint-declared "
                "do_rescale=true and patch_expand_factor=1"
            )

        patch_size = int(raw["patch_size"])
        temporal_patch_size = int(raw["temporal_patch_size"])
        merge_size = int(raw["merge_size"])
        checkpoint_geometry = (patch_size, temporal_patch_size, merge_size)
        tower_geometry = (
            vision_config.patch_size,
            vision_config.temporal_patch_size,
            vision_config.spatial_merge_size,
        )
        if checkpoint_geometry != tower_geometry:
            raise ValueError(
                "GLM-5.3 image processor/tower geometry mismatch: "
                f"processor={checkpoint_geometry} tower={tower_geometry}"
            )

        image_mean = [float(value) for value in raw["image_mean"]]
        image_std = [float(value) for value in raw["image_std"]]
        if (
            len(image_mean) != 3
            or len(image_std) != 3
            or not all(math.isfinite(value) for value in image_mean + image_std)
            or not all(value > 0 for value in image_std)
        ):
            raise ValueError(
                "GLM-5.3 image normalization must contain three finite means "
                "and three positive finite standard deviations"
            )

        return cls(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            min_image_tokens=int(raw["min_image_tokens"]),
            max_image_tokens=int(raw["max_image_tokens"]),
            image_mean=image_mean,
            image_std=image_std,
        )

    def _one(self, image: Image.Image) -> tuple[torch.Tensor, tuple[int, int, int]]:
        image = image.convert("RGB")
        height, width = int(image.height), int(image.width)
        factor = self.patch_size * self.merge_size
        target_height, target_width = glm53_smart_resize(
            height,
            width,
            temporal_factor=self.temporal_patch_size,
            factor=factor,
            min_image_tokens=self.min_image_tokens,
            max_image_tokens=self.max_image_tokens,
        )
        scale = min(target_height / height, target_width / width)
        pixels_per_token = self.temporal_patch_size * factor * factor
        if self.temporal_patch_size * height * width >= pixels_per_token * self.min_image_tokens:
            scale = min(1.0, scale)
        content_height = max(1, min(target_height, math.floor(height * scale)))
        content_width = max(1, min(target_width, math.floor(width * scale)))

        pixels = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)
        if (content_height, content_width) != (height, width):
            pixels = F.interpolate(
                pixels.unsqueeze(0).to(torch.float32),
                size=(content_height, content_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).squeeze(0).clamp_(0, 255)
        pixels = F.pad(
            pixels,
            (0, target_width - content_width, 0, target_height - content_height),
            value=0,
        ).to(torch.float32)
        pixels = (pixels / 255.0 - self.image_mean) / self.image_std

        grid_h = target_height // self.patch_size
        grid_w = target_width // self.patch_size
        if grid_h % self.merge_size or grid_w % self.merge_size:
            raise ValueError(
                f"GLM-5.3 image grid {(grid_h, grid_w)} is not divisible by merge {self.merge_size}"
            )
        patches = pixels.unsqueeze(0).reshape(
            1,
            3,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
        patches = (
            patches.unsqueeze(6)
            .expand(-1, -1, -1, -1, -1, -1, self.temporal_patch_size, -1, -1)
            .reshape(grid_h * grid_w, 3 * self.temporal_patch_size * self.patch_size**2)
            .contiguous()
        )
        return patches, (1, grid_h, grid_w)

    def __call__(self, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        if not images:
            raise ValueError("GLM-5.3 image preprocessing requires at least one image")
        processed = [self._one(image) for image in images]
        return {
            "pixel_values": torch.cat([item[0] for item in processed], dim=0),
            "image_grid_thw": torch.tensor([item[1] for item in processed], dtype=torch.long),
        }


class Glm53VisionRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        dtype = hidden.dtype
        value = hidden.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * self.weight.float()).to(dtype)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Glm53VisionAttention(nn.Module):
    def __init__(self, config: Glm53VisionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.attention_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Glm53VisionRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Glm53VisionRMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        segment_lengths: list[int],
    ) -> torch.Tensor:
        seq_len = int(hidden.shape[0])
        q, k, v = (
            self.qkv(hidden)
            .reshape(seq_len, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = position_embeddings
        cos = cos.unsqueeze(1).float()
        sin = sin.unsqueeze(1).float()
        q_dtype, k_dtype = q.dtype, k.dtype
        q = (q.float() * cos + _rotate_half(q.float()) * sin).to(q_dtype)
        k = (k.float() * cos + _rotate_half(k.float()) * sin).to(k_dtype)

        outputs = []
        offset = 0
        for length in segment_lengths:
            end = offset + int(length)
            qs = q[offset:end].transpose(0, 1).unsqueeze(0)
            ks = k[offset:end].transpose(0, 1).unsqueeze(0)
            vs = v[offset:end].transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(qs, ks, vs, is_causal=False)
            outputs.append(out.squeeze(0).transpose(0, 1))
            offset = end
        if offset != seq_len:
            raise ValueError(f"GLM-5.3 vision segments cover {offset} tokens, expected {seq_len}")
        return self.proj(torch.cat(outputs, dim=0).reshape(seq_len, -1).contiguous())


class Glm53VisionMLP(nn.Module):
    def __init__(self, config: Glm53VisionConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.attention_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.attention_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.attention_bias)
        self.limit = config.swiglu_limit

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden).clamp(max=self.limit)
        up = self.up_proj(hidden).clamp(min=-self.limit, max=self.limit)
        return self.down_proj(F.silu(gate) * up)


class Glm53VisionBlock(nn.Module):
    def __init__(self, config: Glm53VisionConfig):
        super().__init__()
        self.norm1 = Glm53VisionRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = Glm53VisionRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = Glm53VisionAttention(config)
        self.mlp = Glm53VisionMLP(config)

    def forward(self, hidden, position_embeddings, segment_lengths):
        hidden = hidden + self.attn(self.norm1(hidden), position_embeddings, segment_lengths)
        return hidden + self.mlp(self.norm2(hidden))


class Glm53VisionPatchMerger(nn.Module):
    def __init__(self, config: Glm53VisionConfig):
        super().__init__()
        width = config.out_hidden_size
        context = config.projection_intermediate_size
        self.proj = nn.Linear(width, width, bias=False)
        self.post_projection_norm = nn.LayerNorm(width)
        self.gate_proj = nn.Linear(width, context, bias=False)
        self.up_proj = nn.Linear(width, context, bias=False)
        self.down_proj = nn.Linear(context, width, bias=False)
        self.limit = config.swiglu_limit

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.post_projection_norm(self.proj(hidden)))
        gate = self.gate_proj(hidden).clamp(max=self.limit)
        up = self.up_proj(hidden).clamp(min=-self.limit, max=self.limit)
        return self.down_proj(F.silu(gate) * up)


class Glm53VisionModel(nn.Module):
    def __init__(self, config: Glm53VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv3d(
            3,
            config.hidden_size,
            kernel_size=(config.temporal_patch_size, config.patch_size, config.patch_size),
            stride=(config.temporal_patch_size, config.patch_size, config.patch_size),
        )
        self.blocks = nn.ModuleList([Glm53VisionBlock(config) for _ in range(config.depth)])
        self.post_layernorm = Glm53VisionRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.downsample = nn.Conv2d(
            config.hidden_size,
            config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.merger = Glm53VisionPatchMerger(config)
        rotary_dim = (config.hidden_size // config.num_heads) // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        self.register_buffer("rotary_inv_freq", inv_freq, persistent=False)

    def _position_ids(self, grid_thw: torch.Tensor) -> torch.Tensor:
        values = []
        merge = self.config.spatial_merge_size
        for temporal, height, width in grid_thw.tolist():
            hpos, wpos = torch.meshgrid(
                torch.arange(height, device=grid_thw.device),
                torch.arange(width, device=grid_thw.device),
                indexing="ij",
            )
            shape = (height // merge, merge, width // merge, merge)
            hpos = hpos.reshape(shape).transpose(1, 2).flatten()
            wpos = wpos.reshape(shape).transpose(1, 2).flatten()
            values.append(torch.stack((hpos, wpos), dim=-1).repeat(int(temporal), 1))
        return torch.cat(values, dim=0)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        pixel_values = pixel_values.reshape(
            -1, 3, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size
        )
        hidden = self.patch_embed.proj(pixel_values.to(self.patch_embed.proj.weight.dtype))
        hidden = hidden.reshape(-1, cfg.hidden_size)

        positions = self._position_ids(grid_thw)
        rotary = (positions.unsqueeze(-1) * self.rotary_inv_freq).flatten(1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (rotary.cos(), rotary.sin())
        segment_lengths = [int(t * h * w) for t, h, w in grid_thw.tolist()]
        for block in self.blocks:
            hidden = block(hidden, position_embeddings, segment_lengths)

        hidden = self.post_layernorm(hidden)
        merge = cfg.spatial_merge_size
        hidden = hidden.reshape(-1, merge, merge, cfg.hidden_size).permute(0, 3, 1, 2)
        hidden = self.downsample(hidden).reshape(-1, cfg.out_hidden_size)
        return self.merger(hidden)
