"""Losses shared by the baseline and later controlled ablations."""

from __future__ import annotations

import torch


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + epsilon**2).mean()


def frequency_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_fft = torch.fft.rfft2(prediction.float(), norm="ortho")
    target_fft = torch.fft.rfft2(target.float(), norm="ortho")
    return (pred_fft - target_fft).abs().mean()


def restoration_loss(prediction: torch.Tensor, target: torch.Tensor, frequency_weight: float = 0.1) -> torch.Tensor:
    return charbonnier_loss(prediction, target) + frequency_weight * frequency_l1_loss(prediction, target)

