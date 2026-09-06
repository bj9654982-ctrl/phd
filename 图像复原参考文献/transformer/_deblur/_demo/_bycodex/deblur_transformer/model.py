from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm over channels at every spatial location."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class TransposedSelfAttention(nn.Module):
    """Restormer-style multi-head channel attention with global spatial context."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by heads ({heads})")
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dw = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        channels_per_head = channels // self.heads

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, self.heads, channels_per_head, height * width)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (q @ k.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        output = attention @ v
        output = output.reshape(batch, channels, height, width)
        return self.project(output)


class GatedFeedForward(nn.Module):
    """Gated depth-wise convolutional feed-forward network."""

    def __init__(self, channels: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden = max(8, int(channels * expansion))
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=False)
        self.depthwise = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            padding=1,
            groups=hidden * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, gate = self.depthwise(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(first) * gate)


class TransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int, expansion: float = 2.0) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attention = TransposedSelfAttention(channels, heads)
        self.norm2 = LayerNorm2d(channels)
        self.feed_forward = GatedFeedForward(channels, expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


def make_stage(channels: int, depth: int, heads: int, expansion: float) -> nn.Sequential:
    return nn.Sequential(
        *(TransformerBlock(channels, heads, expansion) for _ in range(depth))
    )


class DeblurTransformer(nn.Module):
    """A compact three-scale Transformer U-Net for RGB image deblurring."""

    def __init__(
        self,
        embed_dim: int = 24,
        depths: Iterable[int] = (2, 2, 4),
        heads: Iterable[int] = (1, 2, 4),
        expansion: float = 2.0,
    ) -> None:
        super().__init__()
        depths = tuple(depths)
        heads = tuple(heads)
        if len(depths) != 3 or len(heads) != 3:
            raise ValueError("depths and heads must each contain exactly three values")

        dim1, dim2, dim3 = embed_dim, embed_dim * 2, embed_dim * 4
        self.model_config = {
            "embed_dim": embed_dim,
            "depths": depths,
            "heads": heads,
            "expansion": expansion,
        }

        self.input_projection = nn.Conv2d(3, dim1, kernel_size=3, padding=1, bias=False)
        self.encoder1 = make_stage(dim1, depths[0], heads[0], expansion)
        self.down1 = nn.Conv2d(dim1, dim2, kernel_size=3, stride=2, padding=1, bias=False)
        self.encoder2 = make_stage(dim2, depths[1], heads[1], expansion)
        self.down2 = nn.Conv2d(dim2, dim3, kernel_size=3, stride=2, padding=1, bias=False)
        self.bottleneck = make_stage(dim3, depths[2], heads[2], expansion)

        self.up2 = nn.ConvTranspose2d(dim3, dim2, kernel_size=2, stride=2, bias=False)
        self.reduce2 = nn.Conv2d(dim2 * 2, dim2, kernel_size=1, bias=False)
        self.decoder2 = make_stage(dim2, depths[1], heads[1], expansion)
        self.up1 = nn.ConvTranspose2d(dim2, dim1, kernel_size=2, stride=2, bias=False)
        self.reduce1 = nn.Conv2d(dim1 * 2, dim1, kernel_size=1, bias=False)
        self.decoder1 = make_stage(dim1, depths[0], heads[0], expansion)
        self.output_projection = nn.Conv2d(dim1, 3, kernel_size=3, padding=1, bias=False)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int = 4) -> tuple[torch.Tensor, int, int]:
        height, width = x.shape[-2:]
        pad_height = (multiple - height % multiple) % multiple
        pad_width = (multiple - width % multiple) % multiple
        if pad_height or pad_width:
            mode = "reflect" if height > pad_height and width > pad_width else "replicate"
            x = F.pad(x, (0, pad_width, 0, pad_height), mode=mode)
        return x, height, width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, original_height, original_width = self._pad_to_multiple(x)
        residual = x

        enc1 = self.encoder1(self.input_projection(x))
        enc2 = self.encoder2(self.down1(enc1))
        latent = self.bottleneck(self.down2(enc2))

        dec2 = self.up2(latent)
        dec2 = self.decoder2(self.reduce2(torch.cat((dec2, enc2), dim=1)))
        dec1 = self.up1(dec2)
        dec1 = self.decoder1(self.reduce1(torch.cat((dec1, enc1), dim=1)))

        restored = self.output_projection(dec1) + residual
        restored = restored[..., :original_height, :original_width]
        return restored.clamp(0.0, 1.0)
