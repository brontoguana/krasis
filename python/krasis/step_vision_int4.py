"""Weight-only INT4 modules for lazy vision validation.

This is intentionally a conservative correctness path: weights stay packed as
INT4 plus BF16 scales while resident, and each module dequantizes only its own
weight for the current forward. It proves the memory/quality tradeoff before a
future fused CUDA kernel path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StepVisionInt4Stats:
    modules: int = 0
    source_weight_bytes: int = 0
    quantized_weight_bytes: int = 0
    bf16_bias_bytes: int = 0

    def add(self, other: "StepVisionInt4Stats") -> None:
        self.modules += int(other.modules)
        self.source_weight_bytes += int(other.source_weight_bytes)
        self.quantized_weight_bytes += int(other.quantized_weight_bytes)
        self.bf16_bias_bytes += int(other.bf16_bias_bytes)


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _validate_group_size(group_size: int) -> int:
    group_size = int(group_size)
    if group_size not in (32, 64, 128):
        raise ValueError(f"Vision INT4 group size must be 32, 64, or 128, got {group_size}")
    return group_size


def _quantize_int4_weight(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Quantize rows of a dense/flattened weight tensor to signed symmetric INT4.

    Returns packed uint8 nibbles and BF16 scales. Shapes are generic and padded
    to group boundaries so this works for linears and convolutions alike.
    """
    group_size = _validate_group_size(group_size)
    if weight.ndim < 2:
        raise ValueError(f"INT4 weight must be at least rank-2, got shape {tuple(weight.shape)}")

    rows = int(weight.shape[0])
    cols = int(weight.numel() // rows)
    groups = _ceil_div(cols, group_size)
    padded_cols = groups * group_size

    flat = weight.detach().to(device="cpu", dtype=torch.float32).reshape(rows, cols)
    if padded_cols != cols:
        flat = F.pad(flat, (0, padded_cols - cols))
    grouped = flat.reshape(rows, groups, group_size)
    scales = grouped.abs().amax(dim=2).clamp_min(1.0e-8) / 7.0
    q = torch.round(grouped / scales.unsqueeze(-1)).clamp(-8, 7).to(torch.int16) + 8
    q = q.reshape(rows, padded_cols).to(torch.uint8)

    packed_cols = _ceil_div(padded_cols, 2)
    if packed_cols * 2 != padded_cols:
        q = F.pad(q, (0, packed_cols * 2 - padded_cols))
    q_pairs = q.reshape(rows, packed_cols, 2).to(torch.uint8)
    packed = (q_pairs[:, :, 0] | (q_pairs[:, :, 1] << 4)).contiguous()
    return packed, scales.to(torch.bfloat16).contiguous(), cols, padded_cols


def _dequant_int4_weight(
    packed: torch.Tensor,
    scales: torch.Tensor,
    rows: int,
    cols: int,
    padded_cols: int,
    group_size: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    q = torch.stack((low, high), dim=-1).reshape(rows, -1)[:, :padded_cols]
    q = q.to(torch.int16) - 8
    grouped = q.reshape(rows, padded_cols // group_size, group_size).to(torch.float32)
    weight = grouped * scales.to(torch.float32).unsqueeze(-1)
    weight = weight.reshape(rows, padded_cols)[:, :cols].to(dtype=dtype)
    return weight.reshape(shape).contiguous()


class StepVisionInt4Linear(nn.Module):
    def __init__(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        in_features: int,
        out_features: int,
        cols: int,
        padded_cols: int,
        group_size: int,
        source_weight_bytes: int,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.cols = int(cols)
        self.padded_cols = int(padded_cols)
        self.group_size = _validate_group_size(group_size)
        self.source_weight_bytes = int(source_weight_bytes)
        self.register_buffer("packed_weight", packed.contiguous(), persistent=False)
        self.register_buffer("weight_scales", scales.contiguous(), persistent=False)
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().to(torch.bfloat16).contiguous(), persistent=False)

    @classmethod
    def from_weight_bias(
        cls,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        group_size: int,
    ) -> "StepVisionInt4Linear":
        out_features = int(weight.shape[0])
        in_features = int(weight.numel() // out_features)
        packed, scales, cols, padded_cols = _quantize_int4_weight(weight, group_size)
        return cls(
            packed,
            scales,
            bias,
            in_features=in_features,
            out_features=out_features,
            cols=cols,
            padded_cols=padded_cols,
            group_size=group_size,
            source_weight_bytes=int(weight.numel() * weight.element_size()),
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, group_size: int) -> "StepVisionInt4Linear":
        return cls.from_weight_bias(linear.weight, linear.bias, group_size=group_size)

    def quantized_weight_bytes(self) -> int:
        return int(self.packed_weight.numel() * self.packed_weight.element_size()) + int(
            self.weight_scales.numel() * self.weight_scales.element_size()
        )

    def bf16_bias_bytes(self) -> int:
        if self.bias is None:
            return 0
        return int(self.bias.numel() * self.bias.element_size())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = _dequant_int4_weight(
            self.packed_weight,
            self.weight_scales,
            self.out_features,
            self.cols,
            self.padded_cols,
            self.group_size,
            (self.out_features, self.in_features),
            x.dtype,
        )
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None and self.bias.dtype != x.dtype else self.bias
        return F.linear(x, weight, bias)


class StepVisionInt4Conv2d(nn.Module):
    def __init__(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        original_shape: tuple[int, int, int, int],
        cols: int,
        padded_cols: int,
        group_size: int,
        stride,
        padding,
        dilation,
        groups: int,
        source_weight_bytes: int,
    ):
        super().__init__()
        self.original_shape = tuple(int(x) for x in original_shape)
        self.out_channels = int(original_shape[0])
        self.cols = int(cols)
        self.padded_cols = int(padded_cols)
        self.group_size = _validate_group_size(group_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = int(groups)
        self.source_weight_bytes = int(source_weight_bytes)
        self.register_buffer("packed_weight", packed.contiguous(), persistent=False)
        self.register_buffer("weight_scales", scales.contiguous(), persistent=False)
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().to(torch.bfloat16).contiguous(), persistent=False)

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, *, group_size: int) -> "StepVisionInt4Conv2d":
        packed, scales, cols, padded_cols = _quantize_int4_weight(conv.weight, group_size)
        return cls(
            packed,
            scales,
            conv.bias,
            original_shape=tuple(conv.weight.shape),
            cols=cols,
            padded_cols=padded_cols,
            group_size=group_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            source_weight_bytes=int(conv.weight.numel() * conv.weight.element_size()),
        )

    def quantized_weight_bytes(self) -> int:
        return int(self.packed_weight.numel() * self.packed_weight.element_size()) + int(
            self.weight_scales.numel() * self.weight_scales.element_size()
        )

    def bf16_bias_bytes(self) -> int:
        if self.bias is None:
            return 0
        return int(self.bias.numel() * self.bias.element_size())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = _dequant_int4_weight(
            self.packed_weight,
            self.weight_scales,
            self.out_channels,
            self.cols,
            self.padded_cols,
            self.group_size,
            self.original_shape,
            x.dtype,
        )
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None and self.bias.dtype != x.dtype else self.bias
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class StepVisionInt4Attention(nn.Module):
    def __init__(
        self,
        in_proj: StepVisionInt4Linear,
        out_proj: StepVisionInt4Linear,
        in_proj_bias: Optional[torch.Tensor],
        *,
        num_heads: int,
        head_dim: int,
        scale: float,
        rope: Optional[nn.Module],
    ):
        super().__init__()
        self.in_proj = in_proj
        self.out_proj = out_proj
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.scale = float(scale)
        self.rope = rope
        if in_proj_bias is None:
            self.in_proj_bias = None
        else:
            self.register_buffer("in_proj_bias", in_proj_bias.detach().to(torch.bfloat16).contiguous(), persistent=False)

    @classmethod
    def from_attention(cls, attention: nn.Module, *, group_size: int) -> "StepVisionInt4Attention":
        in_proj = StepVisionInt4Linear.from_weight_bias(
            attention.in_proj_weight,
            None,
            group_size=group_size,
        )
        out_proj = StepVisionInt4Linear.from_linear(attention.out_proj, group_size=group_size)
        return cls(
            in_proj,
            out_proj,
            attention.in_proj_bias,
            num_heads=int(attention.num_heads),
            head_dim=int(attention.head_dim),
            scale=float(attention.scale),
            rope=attention.rope,
        )

    def forward(self, hidden_states: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        qkv = self.in_proj(hidden_states)
        if self.in_proj_bias is not None:
            bias = (
                self.in_proj_bias.to(dtype=qkv.dtype)
                if self.in_proj_bias.dtype != qkv.dtype
                else self.in_proj_bias
            )
            qkv = qkv + bias
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, k, grid_hw=grid_hw)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=self.scale)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(attn_output)


def _collect_stats(module: nn.Module) -> StepVisionInt4Stats:
    stats = StepVisionInt4Stats()
    for child in module.modules():
        if isinstance(child, (StepVisionInt4Linear, StepVisionInt4Conv2d)):
            stats.modules += 1
            stats.source_weight_bytes += int(child.source_weight_bytes)
            stats.quantized_weight_bytes += int(child.quantized_weight_bytes())
            stats.bf16_bias_bytes += int(child.bf16_bias_bytes())
    return stats


def quantize_vision_modules_int4(
    *modules: nn.Module,
    group_size: int = 128,
) -> tuple[tuple[nn.Module, ...], StepVisionInt4Stats]:
    group_size = _validate_group_size(group_size)

    def replace_children(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if (
                hasattr(child, "in_proj_weight")
                and hasattr(child, "in_proj_bias")
                and hasattr(child, "out_proj")
                and hasattr(child, "num_heads")
                and hasattr(child, "head_dim")
            ):
                setattr(module, name, StepVisionInt4Attention.from_attention(child, group_size=group_size))
            elif isinstance(child, nn.Linear):
                setattr(module, name, StepVisionInt4Linear.from_linear(child, group_size=group_size))
            elif isinstance(child, nn.Conv2d):
                setattr(module, name, StepVisionInt4Conv2d.from_conv2d(child, group_size=group_size))
            else:
                replace_children(child)

    quantized = []
    for module in modules:
        replace_children(module)
        quantized.append(module)

    stats = StepVisionInt4Stats()
    for module in quantized:
        stats.add(_collect_stats(module))
    return tuple(quantized), stats


def quantize_step_vision_modules_int4(
    vision: nn.Module,
    projector: nn.Module,
    *,
    group_size: int = 128,
) -> tuple[nn.Module, nn.Module, StepVisionInt4Stats]:
    if isinstance(projector, nn.Linear):
        projector = StepVisionInt4Linear.from_linear(projector, group_size=group_size)

    (vision,), stats = quantize_vision_modules_int4(vision, group_size=group_size)
    stats.add(_collect_stats(projector))
    return vision, projector, stats
