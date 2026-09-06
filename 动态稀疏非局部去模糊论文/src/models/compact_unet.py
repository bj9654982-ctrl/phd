"""A compact residual U-Net used as the controlled deblurring baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.body(x)


def make_stage(channels: int, blocks: int) -> nn.Sequential:
    return nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))


class CompactDeblurUNet(nn.Module):
    """Two-level U-Net with residual prediction and no CPU-specific branch."""

    def __init__(
        self,
        width: int = 24,
        encoder_blocks: tuple[int, int] = (2, 2),
        bottleneck_blocks: int = 4,
        decoder_blocks: tuple[int, int] = (2, 2),
        bottleneck_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if width < 8:
            raise ValueError("width must be at least 8")

        c1, c2, c3 = width, width * 2, width * 4
        self.stem = nn.Conv2d(3, c1, 3, padding=1)
        self.encoder1 = make_stage(c1, encoder_blocks[0])
        self.down1 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.encoder2 = make_stage(c2, encoder_blocks[1])
        self.down2 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)

        self.bottleneck = make_stage(c3, bottleneck_blocks)
        self.bottleneck_module = bottleneck_module

        self.up2 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)
        self.fuse2 = nn.Conv2d(c2 * 2, c2, 1)
        self.decoder2 = make_stage(c2, decoder_blocks[1])
        self.up1 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)
        self.fuse1 = nn.Conv2d(c1 * 2, c1, 1)
        self.decoder1 = make_stage(c1, decoder_blocks[0])
        self.head = nn.Conv2d(c1, 3, 3, padding=1)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int = 4) -> tuple[torch.Tensor, int, int]:
        height, width = x.shape[-2:]
        pad_h = (multiple - height % multiple) % multiple
        pad_w = (multiple - width % multiple) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, height, width

    def forward(
        self,
        blurry: torch.Tensor,
        return_geometry: bool = False,
        geometry_override: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if blurry.ndim != 4 or blurry.shape[1] != 3:
            raise ValueError(f"expected BCHW RGB input, got {tuple(blurry.shape)}")

        padded, height, width = self._pad_to_multiple(blurry)
        e1 = self.encoder1(self.stem(padded))
        e2 = self.encoder2(self.down1(e1))
        latent = self.bottleneck(self.down2(e2))
        geometry = None
        if self.bottleneck_module is not None:
            if return_geometry:
                latent, geometry = self.bottleneck_module(
                    latent,
                    geometry_source=padded,
                    geometry_override=geometry_override,
                    return_geometry=True,
                )
            else:
                latent = self.bottleneck_module(
                    latent,
                    geometry_source=padded,
                    geometry_override=geometry_override,
                )
        elif geometry_override is not None:
            raise RuntimeError("geometry_override requires a geometry-aware bottleneck module")

        d2 = self.decoder2(self.fuse2(torch.cat((self.up2(latent), e2), dim=1)))
        d1 = self.decoder1(self.fuse1(torch.cat((self.up1(d2), e1), dim=1)))
        restored = padded + self.head(d1)
        restored = restored[..., :height, :width]
        if not return_geometry:
            return restored
        if geometry is None:
            raise RuntimeError("return_geometry requires a geometry-aware bottleneck module")
        return restored, geometry

    def predict_geometry(self, blurry: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run only the encoder and geometry-aware bottleneck."""
        if self.bottleneck_module is None:
            raise RuntimeError("predict_geometry requires a geometry-aware bottleneck module")
        padded, _, _ = self._pad_to_multiple(blurry)
        e1 = self.encoder1(self.stem(padded))
        e2 = self.encoder2(self.down1(e1))
        latent = self.bottleneck(self.down2(e2))
        _, geometry = self.bottleneck_module(
            latent,
            geometry_source=padded,
            return_geometry=True,
        )
        return geometry


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
