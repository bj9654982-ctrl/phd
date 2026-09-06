"""Lightweight scalar and spatial gates for a frozen restoration residual."""

from __future__ import annotations

import math

import torch
from torch import nn


class ImageResidualGate(nn.Module):
    """Predict one residual-strength scalar from the blurry image and B0 residual."""

    def __init__(self, width: int = 16, initial_alpha: float = 0.95) -> None:
        super().__init__()
        if width < 4:
            raise ValueError("width must be at least 4")
        if not 0 < initial_alpha < 1:
            raise ValueError("initial_alpha must be in (0, 1)")
        self.encoder = nn.Sequential(
            nn.Conv2d(6, width, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width * 2, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.head.bias, math.log(initial_alpha / (1.0 - initial_alpha)))

    def forward(self, blurry: torch.Tensor, baseline_residual: torch.Tensor) -> torch.Tensor:
        if blurry.ndim != 4 or blurry.shape[1] != 3:
            raise ValueError(f"blurry must have BCHW RGB shape, got {tuple(blurry.shape)}")
        if baseline_residual.shape != blurry.shape:
            raise ValueError("baseline_residual must have the same shape as blurry")
        features = self.encoder(torch.cat((blurry, baseline_residual), dim=1)).flatten(1)
        return self.head(features).sigmoid()[:, :, None, None]


class SpatialResidualGate(nn.Module):
    """Predict a dense residual-strength map from the blurry image and B0 residual.

    The four convolutions have a 33x33 receptive field while preserving the input
    resolution.  This covers the default 31x31 local-oracle supervision window.
    """

    def __init__(self, width: int = 16, initial_alpha: float = 0.95) -> None:
        super().__init__()
        if width < 4:
            raise ValueError("width must be at least 4")
        if not 0 < initial_alpha < 1:
            raise ValueError("initial_alpha must be in (0, 1)")
        self.features = nn.Sequential(
            nn.Conv2d(6, width, 5, padding=2),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=8, dilation=8),
            nn.GELU(),
        )
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.head.bias, math.log(initial_alpha / (1.0 - initial_alpha)))

    def forward(self, blurry: torch.Tensor, baseline_residual: torch.Tensor) -> torch.Tensor:
        if blurry.ndim != 4 or blurry.shape[1] != 3:
            raise ValueError(f"blurry must have BCHW RGB shape, got {tuple(blurry.shape)}")
        if baseline_residual.shape != blurry.shape:
            raise ValueError("baseline_residual must have the same shape as blurry")
        features = self.features(torch.cat((blurry, baseline_residual), dim=1))
        return self.head(features).sigmoid()


class SpatialResidualCorrection(nn.Module):
    """Predict a bounded, zero-spatial-mean correction to an image-level gate.

    The zero-initialized head makes the anchored model exactly reproduce the
    frozen scalar gate before training.  Subtracting the spatial mean prevents
    this branch from replacing the scalar gate's global calibration.
    """

    def __init__(self, width: int = 16, max_correction: float = 0.15) -> None:
        super().__init__()
        if width < 4:
            raise ValueError("width must be at least 4")
        if not 0 < max_correction <= 0.5:
            raise ValueError("max_correction must be in (0, 0.5]")
        self.max_correction = float(max_correction)
        self.features = nn.Sequential(
            nn.Conv2d(6, width, 5, padding=2),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=8, dilation=8),
            nn.GELU(),
        )
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, blurry: torch.Tensor, baseline_residual: torch.Tensor) -> torch.Tensor:
        if blurry.ndim != 4 or blurry.shape[1] != 3:
            raise ValueError(f"blurry must have BCHW RGB shape, got {tuple(blurry.shape)}")
        if baseline_residual.shape != blurry.shape:
            raise ValueError("baseline_residual must have the same shape as blurry")
        raw = self.head(self.features(torch.cat((blurry, baseline_residual), dim=1))).tanh()
        centered = raw - raw.mean(dim=(-2, -1), keepdim=True)
        return 0.5 * self.max_correction * centered


def apply_anchored_residual_correction(
    scalar_alpha: torch.Tensor,
    spatial_correction: torch.Tensor,
) -> torch.Tensor:
    """Add a dense correction to a scalar alpha and keep the result in [0, 1]."""
    if scalar_alpha.ndim != 4 or scalar_alpha.shape[1:] != (1, 1, 1):
        raise ValueError("scalar_alpha must have shape (B, 1, 1, 1)")
    if spatial_correction.ndim != 4 or spatial_correction.shape[1] != 1:
        raise ValueError("spatial_correction must have shape (B, 1, H, W)")
    if spatial_correction.shape[0] != scalar_alpha.shape[0]:
        raise ValueError("scalar_alpha and spatial_correction batch sizes must match")
    return (scalar_alpha + spatial_correction).clamp(0.0, 1.0)


def apply_residual_gate(
    blurry: torch.Tensor,
    baseline_restored: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    if blurry.shape != baseline_restored.shape:
        raise ValueError("blurry and baseline_restored must have the same shape")
    valid_shapes = {
        (blurry.shape[0], 1, 1, 1),
        (blurry.shape[0], 1, blurry.shape[2], blurry.shape[3]),
    }
    if tuple(alpha.shape) not in valid_shapes:
        raise ValueError("alpha must have shape (B, 1, 1, 1) or (B, 1, H, W)")
    return blurry + alpha * (baseline_restored - blurry)


def compose_protected_expert(
    blurry: torch.Tensor,
    baseline_restored: torch.Tensor,
    expert_restored: torch.Tensor,
    baseline_alpha: torch.Tensor,
    expert_gate: torch.Tensor,
) -> torch.Tensor:
    """Compose protected B0 correction and a separately gated expert residual.

    The two branches have distinct jobs: ``baseline_alpha`` controls how much
    of B0's correction to retain, while ``expert_gate`` controls only the
    expert-minus-B0 residual.  No sharp target is required by this operation.
    """
    if expert_restored.shape != blurry.shape:
        raise ValueError("expert_restored must have the same shape as blurry")
    valid_expert_gate_shapes = {
        (blurry.shape[0], 1, 1, 1),
        (blurry.shape[0], 1, blurry.shape[2], blurry.shape[3]),
    }
    if tuple(expert_gate.shape) not in valid_expert_gate_shapes:
        raise ValueError(
            "expert_gate must have shape (B, 1, 1, 1) or (B, 1, H, W)"
        )
    protected = apply_residual_gate(blurry, baseline_restored, baseline_alpha)
    return protected + expert_gate * (expert_restored - baseline_restored)


@torch.no_grad()
def mse_optimal_residual_alpha(
    blurry: torch.Tensor,
    baseline_restored: torch.Tensor,
    sharp: torch.Tensor,
) -> torch.Tensor:
    """Return the per-image alpha in [0, 1] minimizing reconstruction MSE."""
    if blurry.shape != baseline_restored.shape or blurry.shape != sharp.shape:
        raise ValueError("blurry, baseline_restored, and sharp must have the same shape")
    residual = baseline_restored.float() - blurry.float()
    error = blurry.float() - sharp.float()
    numerator = -(error * residual).flatten(1).sum(dim=1)
    denominator = residual.square().flatten(1).sum(dim=1).clamp_min(1e-12)
    return (numerator / denominator).clamp(0.0, 1.0)[:, None, None, None]


@torch.no_grad()
def mse_optimal_residual_alpha_map(
    blurry: torch.Tensor,
    baseline_restored: torch.Tensor,
    sharp: torch.Tensor,
    window_size: int = 31,
) -> torch.Tensor:
    """Return the dense alpha map minimizing MSE inside each local window.

    For baseline residual ``r = baseline_restored - blurry`` and input error
    ``e = blurry - sharp``, each target is
    ``clip(-sum(e*r) / sum(r*r), 0, 1)`` over a local spatial neighborhood and
    all RGB channels.
    """
    if blurry.shape != baseline_restored.shape or blurry.shape != sharp.shape:
        raise ValueError("blurry, baseline_restored, and sharp must have the same shape")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    residual = baseline_restored.float() - blurry.float()
    error = blurry.float() - sharp.float()
    numerator = -(error * residual).sum(dim=1, keepdim=True)
    denominator = residual.square().sum(dim=1, keepdim=True)
    padding = window_size // 2
    numerator = torch.nn.functional.avg_pool2d(
        numerator,
        kernel_size=window_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    )
    denominator = torch.nn.functional.avg_pool2d(
        denominator,
        kernel_size=window_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    ).clamp_min(1e-12)
    return (numerator / denominator).clamp(0.0, 1.0)
