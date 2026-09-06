"""Deployable blur-strength gating for P7 trajectory specialization."""

from __future__ import annotations

import torch


def input_psnr_per_image(blurry: torch.Tensor, sharp: torch.Tensor) -> torch.Tensor:
    """Paired training-only blur-severity teacher, returned in dB per image."""
    if blurry.shape != sharp.shape or blurry.ndim != 4:
        raise ValueError("blurry and sharp must have the same BCHW shape")
    mse = (blurry.float() - sharp.float()).square().flatten(1).mean(1).clamp_min(1e-12)
    return -10.0 * mse.log10()


def teacher_blur_strength(
    input_psnr_db: torch.Tensor,
    heavy_max_psnr_db: float,
    light_min_psnr_db: float,
) -> torch.Tensor:
    """Map paired input PSNR to [0, 1], where one denotes heavy blur.

    This signal is legal only for training supervision and analysis; it must
    never be supplied to a deployed model.
    """
    if light_min_psnr_db <= heavy_max_psnr_db:
        raise ValueError("light_min_psnr_db must exceed heavy_max_psnr_db")
    linear = (
        (float(light_min_psnr_db) - input_psnr_db.float())
        / (float(light_min_psnr_db) - float(heavy_max_psnr_db))
    ).clamp(0.0, 1.0)
    return linear.square() * (3.0 - 2.0 * linear)


def correction_log_l1(
    blurry: torch.Tensor,
    baseline_output: torch.Tensor,
) -> torch.Tensor:
    """Compute the P7-A selected blurry-only proxy once per image."""
    if blurry.shape != baseline_output.shape or blurry.ndim != 4:
        raise ValueError("blurry and baseline_output must have the same BCHW shape")
    correction_l1 = (baseline_output.float() - blurry.float()).abs().flatten(1).mean(1)
    return correction_l1.clamp_min(1e-12).log()


def deployable_strength_gate(
    blurry: torch.Tensor,
    baseline_output: torch.Tensor,
    light_cut_log_l1: float,
    heavy_cut_log_l1: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a smooth per-image gate and the underlying log-correction proxy."""
    if heavy_cut_log_l1 <= light_cut_log_l1:
        raise ValueError("heavy_cut_log_l1 must exceed light_cut_log_l1")
    proxy = correction_log_l1(blurry, baseline_output)
    linear = (
        (proxy - float(light_cut_log_l1))
        / (float(heavy_cut_log_l1) - float(light_cut_log_l1))
    ).clamp(0.0, 1.0)
    gate = linear.square() * (3.0 - 2.0 * linear)
    return gate[:, None, None, None], proxy


def blend_with_baseline(
    baseline_output: torch.Tensor,
    dynamic_output: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Blend the specialist into B0; gate=0 is exactly the safe baseline."""
    if baseline_output.shape != dynamic_output.shape or baseline_output.ndim != 4:
        raise ValueError("baseline_output and dynamic_output must have the same BCHW shape")
    if gate.ndim != 4 or gate.shape[0] != baseline_output.shape[0] or gate.shape[1:] != (1, 1, 1):
        raise ValueError("gate must have shape (B, 1, 1, 1)")
    return baseline_output + gate.to(baseline_output.dtype) * (dynamic_output - baseline_output)
