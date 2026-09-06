"""Metric functions with an explicit [0, 1] image convention."""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = torch.mean((prediction - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return -10.0 * math.log10(mse)

