"""DeepSeek-V4-Flash-Vision image tower and checkpoint-native preprocessing."""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn


IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
COMPRESS_PAD_TO = 4
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"


@dataclass(frozen=True)
class DeepseekV4ImageInput:
    start: int
    patches: torch.Tensor
    n_vit_h: int
    n_vit_w: int
    types: torch.Tensor
    perm: torch.Tensor


def _grid_tokens(best_height, best_width, patch_size, downsample_ratio):
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def _solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    ratio = height / width
    max_w_float = math.sqrt((max_n_token - 2) / ratio + 0.25) - 0.5
    max_h_float = max_w_float * ratio
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        if max_w <= 1:
            raise ValueError("DeepSeek-V4 vision token budget cannot fit image geometry")
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = _grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def _safe_resize(height, width, best_height, best_width, cfg):
    max_n_token = cfg.vision_max_n_token - (COMPRESS_PAD_TO - 1)
    n_llm_h, n_llm_w, num_tokens = _grid_tokens(
        best_height,
        best_width,
        cfg.vision_patch_size,
        cfg.vision_downsample_ratio,
    )
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = _solve_resize_ratio(
            height,
            width,
            cfg.vision_patch_size,
            cfg.vision_downsample_ratio,
            budget,
        )
        budget -= 1
        if budget <= 2:
            raise ValueError("DeepSeek-V4 vision token budget could not be satisfied")
    return n_llm_h, n_llm_w, best_height, best_width


def _build_image_block(n_llm_h: int, n_llm_w: int, start_pos: int):
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
        + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    order = (
        torch.arange(rows * row_len)
        .view(rows // 2, 2, row_len)
        .transpose(1, 2)
        .reshape(-1)
    )
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(
        n_llm_h * n_llm_w
    ).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_START]),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_END]),
        ]
    )
    return types, perm


class DeepseekV4ImagePreprocessor:
    def __init__(self, cfg):
        self.cfg = cfg

    def prepare(self, image: Image.Image, start_pos: int) -> DeepseekV4ImageInput:
        cfg = self.cfg
        patch = cfg.vision_patch_size
        image = image.convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("DeepSeek-V4 image has invalid dimensions")
        if width > height * cfg.vision_max_wh_ratio:
            width = int(height * cfg.vision_max_wh_ratio)
        if width * height < cfg.vision_min_pixels:
            ratio = (cfg.vision_min_pixels / (width * height)) ** 0.5
            width = int(width * ratio)
            height = int(height * ratio)
        best_width = math.ceil(width / patch) * patch
        best_height = math.ceil(height / patch) * patch
        n_llm_h, n_llm_w, best_height, best_width = _safe_resize(
            height, width, best_height, best_width, cfg
        )
        n_vit_h, n_vit_w = best_height // patch, best_width // patch
        if image.width >= cfg.vision_max_wh_ratio * image.height:
            image = image.resize((best_width, best_height))
        else:
            image = ImageOps.pad(
                image, (best_width, best_height), color=(127, 127, 127)
            )
        pixels = (
            torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
            .permute(2, 0, 1)
            .div_(255.0)
        )
        pixels = ((pixels - 0.5) / 0.5).to(torch.bfloat16)
        patches = (
            pixels.reshape(3, n_vit_h, patch, n_vit_w, patch)
            .permute(1, 3, 0, 2, 4)
            .reshape(n_vit_h * n_vit_w, 3, patch, patch)
        )
        types, perm = _build_image_block(n_llm_h, n_llm_w, start_pos)
        return DeepseekV4ImageInput(
            start_pos, patches, n_vit_h, n_vit_w, types, perm
        )


def _vision_cos_sin(n_h, n_w, dim, theta, device):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    hpos = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w)
    freqs = (
        torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    ).flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def _apply_rotary(x, cos, sin):
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class DeepseekV4VisionRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        return (self.weight * xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)).to(dtype)


class DeepseekV4VisionAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.vision_n_heads
        self.head_dim = cfg.vision_dim // cfg.vision_n_heads
        self.wqkv = nn.Linear(cfg.vision_dim, 3 * cfg.vision_dim)
        self.wo = nn.Linear(cfg.vision_dim, cfg.vision_dim)

    def forward(self, x, cos, sin):
        n = x.shape[0]
        q, k, v = (
            t.view(n, self.n_heads, self.head_dim)
            for t in self.wqkv(x).chunk(3, dim=-1)
        )
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        )
        return self.wo(out.transpose(0, 1).reshape(n, -1))


class DeepseekV4VisionMlp(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.vision_dim, 2 * cfg.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(cfg.vision_inter_dim, cfg.vision_dim, bias=False)

    def forward(self, x):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class DeepseekV4VisionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = DeepseekV4VisionRMSNorm(cfg.vision_dim)
        self.attn = DeepseekV4VisionAttention(cfg)
        self.norm2 = DeepseekV4VisionRMSNorm(cfg.vision_dim)
        self.mlp = DeepseekV4VisionMlp(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class DeepseekV4VisionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.rope_dim = cfg.vision_dim // cfg.vision_n_heads // 2
        self.rope_theta = cfg.vision_rope_theta
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Linear(
            3 * cfg.vision_patch_size**2, cfg.vision_dim
        )
        self.blocks = nn.ModuleList(
            [DeepseekV4VisionBlock(cfg) for _ in range(cfg.vision_n_layers)]
        )
        self.norm = DeepseekV4VisionRMSNorm(cfg.vision_dim)

    def forward(self, patches, n_h, n_w):
        x = self.patch_embed.proj(patches.flatten(1))
        cos, sin = _vision_cos_sin(
            n_h, n_w, self.rope_dim, self.rope_theta, x.device
        )
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class DeepseekV4VisionAligner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.downsample_ratio = cfg.vision_downsample_ratio
        in_dim = cfg.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, cfg.hidden_size)
        self.w2 = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x, n_h, n_w):
        ratio = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % ratio, 0, -n_h % ratio))
        x = F.unfold(x.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))
