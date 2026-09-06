"""Low-degree-of-freedom motion orientation and length prediction."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MotionGeometryHead(nn.Module):
    def __init__(self, channels: int, hidden_channels: int | None = None, max_length: float = 25.0) -> None:
        super().__init__()
        hidden = hidden_channels or max(channels // 2, 16)
        self.max_length = float(max_length)
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.direction_head = nn.Conv2d(hidden, 2, 1)
        self.length_head = nn.Conv2d(hidden, 1, 1)
        self.confidence_head = nn.Conv2d(hidden, 1, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.body(features)
        direction = F.normalize(self.direction_head(hidden), dim=1, eps=1e-6)
        length = 1.0 + (self.max_length - 1.0) * self.length_head(hidden).sigmoid()
        confidence = self.confidence_head(hidden).sigmoid()
        return direction, length, confidence


class GeometryProbe(nn.Module):
    """Small encoder used only for the E0 synthetic geometry learnability gate."""

    def __init__(self, width: int = 24, max_length: float = 25.0) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.statistics_projection = nn.Sequential(
            nn.Linear(4, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        # Concatenation preserves absolute blur-strength statistics. Adding the
        # branches allowed the unconstrained learned feature to cancel them.
        self.head = MotionGeometryHead(width * 3, max_length=max_length)

    @staticmethod
    def _gradient_statistics(blurry: torch.Tensor) -> torch.Tensor:
        gray = blurry.mean(dim=1)
        gradient_x = gray[:, :, 2:] - gray[:, :, :-2]
        gradient_y = gray[:, 2:, :] - gray[:, :-2, :]
        gradient_x = gradient_x[:, 1:-1, :]
        gradient_y = gradient_y[:, :, 1:-1]
        jxx = gradient_x.square().mean(dim=(-2, -1))
        jyy = gradient_y.square().mean(dim=(-2, -1))
        jxy = (gradient_x * gradient_y).mean(dim=(-2, -1))
        energy = jxx + jyy + 1e-8
        orientation_x = (jyy - jxx) / energy
        orientation_y = (-2.0 * jxy) / energy
        anisotropy = torch.sqrt(orientation_x.square() + orientation_y.square() + 1e-8)
        log_energy = torch.log(energy)
        # The first three values already have a natural O(1) range. Centering
        # log-energy keeps the last feature on a comparable numerical scale.
        normalized_log_energy = (log_energy + 6.0) / 2.0
        return torch.stack((orientation_x, orientation_y, anisotropy, normalized_log_energy), dim=1)

    def forward(self, blurry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        learned = self.pool(self.encoder(blurry)).flatten(1)
        statistics = self.statistics_projection(self._gradient_statistics(blurry))
        fused = torch.cat((learned, statistics), dim=1)[:, :, None, None]
        return self.head(fused)


def geometry_supervision_loss(
    direction: torch.Tensor,
    length: torch.Tensor,
    confidence: torch.Tensor,
    target_direction: torch.Tensor,
    target_length: torch.Tensor,
    max_length: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_direction = target_direction[:, :, None, None]
    target_length = target_length[:, None, None, None]
    target_double_angle = torch.cat(
        (
            target_direction[:, :1].square() - target_direction[:, 1:2].square(),
            2.0 * target_direction[:, :1] * target_direction[:, 1:2],
        ),
        dim=1,
    )
    dot = (direction * target_double_angle).sum(dim=1, keepdim=True)
    direction_loss = (1.0 - dot).mean()
    normalized_length_error = (length - target_length) / (max_length - 1.0)
    length_loss = normalized_length_error.square().mean()
    confidence_loss = (1.0 - confidence.float()).square().mean()
    total = direction_loss + 4.0 * length_loss + 0.05 * confidence_loss
    return total, {
        "direction": direction_loss.detach(),
        "length": length_loss.detach(),
        "confidence": confidence_loss.detach(),
    }


def paired_motion_direction_pseudo_label(
    blurry: torch.Tensor,
    sharp: torch.Tensor,
    output_size: tuple[int, int],
    window: int = 51,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer a dense unoriented motion direction from paired gradient loss.

    This function is intended for training pairs only. The dominant eigenvector
    of ``J_sharp - J_blur`` estimates the direction whose gradient energy was
    most strongly attenuated by blur. It returns a double-angle map and a
    reliability weight in ``[0, 1]``.
    """
    if blurry.shape != sharp.shape or blurry.ndim != 4:
        raise ValueError("blurry and sharp must have the same BCHW shape")
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")

    def tensor_components(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gray = image.float().mean(dim=1, keepdim=True)
        gradient_x = F.pad(gray[:, :, :, 2:] - gray[:, :, :, :-2], (1, 1, 0, 0), mode="replicate")
        gradient_y = F.pad(gray[:, :, 2:, :] - gray[:, :, :-2, :], (0, 0, 1, 1), mode="replicate")
        maximum_window = min(image.shape[-2:])
        if maximum_window % 2 == 0:
            maximum_window -= 1
        effective_window = min(window, maximum_window)
        source_height, source_width = image.shape[-2:]
        output_height, output_width = output_size
        stride = (source_height // output_height, source_width // output_width)
        if source_height != output_height * stride[0] or source_width != output_width * stride[1]:
            stride = (1, 1)
        padding = effective_window // 2
        jxx = F.avg_pool2d(gradient_x.square(), effective_window, stride=stride, padding=padding)
        jyy = F.avg_pool2d(gradient_y.square(), effective_window, stride=stride, padding=padding)
        jxy = F.avg_pool2d(gradient_x * gradient_y, effective_window, stride=stride, padding=padding)
        if jxx.shape[-2:] != output_size:
            resize = {"size": output_size, "mode": "bilinear", "align_corners": False}
            jxx = F.interpolate(jxx, **resize)
            jyy = F.interpolate(jyy, **resize)
            jxy = F.interpolate(jxy, **resize)
        return jxx, jyy, jxy

    blur_xx, blur_yy, blur_xy = tensor_components(blurry)
    sharp_xx, sharp_yy, sharp_xy = tensor_components(sharp)
    delta_xx = sharp_xx - blur_xx
    delta_yy = sharp_yy - blur_yy
    delta_xy = sharp_xy - blur_xy
    orientation = torch.cat((delta_xx - delta_yy, 2.0 * delta_xy), dim=1)
    magnitude = orientation.norm(dim=1, keepdim=True)
    double_angle = F.normalize(orientation, dim=1, eps=1e-8)
    sharp_energy = sharp_xx + sharp_yy + 1e-8
    positive_loss = (delta_xx + delta_yy).clamp_min(0.0)
    confidence = (magnitude / sharp_energy).clamp(0.0, 1.0)
    confidence = confidence * (positive_loss / sharp_energy).clamp(0.0, 1.0)
    return double_angle, confidence


def paired_motion_geometry_pseudo_label(
    blurry: torch.Tensor,
    sharp: torch.Tensor,
    output_size: tuple[int, int],
    window: int = 31,
    max_feature_length: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate paired-image motion direction, feature length, and reliability.

    This is a diagnostic pseudo oracle, not an inference-time estimator.  The
    direction comes from the dominant paired gradient-energy loss.  The feature
    length is a monotonic mapping of local gradient attenuation into
    ``[1, max_feature_length]``.
    """
    if max_feature_length <= 1.0:
        raise ValueError("max_feature_length must be greater than 1")
    double_angle, direction_confidence = paired_motion_direction_pseudo_label(
        blurry,
        sharp,
        output_size,
        window=window,
    )

    def local_gradient_energy(image: torch.Tensor) -> torch.Tensor:
        gray = image.float().mean(dim=1, keepdim=True)
        gradient_x = F.pad(
            gray[:, :, :, 2:] - gray[:, :, :, :-2],
            (1, 1, 0, 0),
            mode="replicate",
        )
        gradient_y = F.pad(
            gray[:, :, 2:, :] - gray[:, :, :-2, :],
            (0, 0, 1, 1),
            mode="replicate",
        )
        maximum_window = min(image.shape[-2:])
        if maximum_window % 2 == 0:
            maximum_window -= 1
        effective_window = min(window, maximum_window)
        padding = effective_window // 2
        energy = F.avg_pool2d(
            gradient_x.square() + gradient_y.square(),
            kernel_size=effective_window,
            stride=1,
            padding=padding,
        )
        if energy.shape[-2:] != output_size:
            energy = F.interpolate(
                energy,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return energy

    sharp_energy = local_gradient_energy(sharp)
    blur_energy = local_gradient_energy(blurry)
    attenuation = ((sharp_energy - blur_energy).clamp_min(0.0) / sharp_energy.clamp_min(1e-8)).clamp(0.0, 1.0)
    strength = attenuation.sqrt()
    length = 1.0 + (max_feature_length - 1.0) * strength
    angle = 0.5 * torch.atan2(double_angle[:, 1:2], double_angle[:, :1])
    direction = torch.cat((angle.cos(), angle.sin()), dim=1)
    confidence = direction_confidence.sqrt().clamp(0.0, 1.0)
    return direction, length, confidence


def dense_paired_geometry_supervision_loss(
    predicted_double_angle: torch.Tensor,
    predicted_length: torch.Tensor,
    predicted_confidence: torch.Tensor,
    target_direction: torch.Tensor,
    target_length: torch.Tensor,
    target_confidence: torch.Tensor,
    max_feature_length: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise blurry-only dense geometry with paired-image pseudo labels.

    Motion direction is unoriented: ``d`` and ``-d`` describe the same blur
    line.  Double-angle encoding makes those two representations identical.
    Direction and length errors are weighted by detached teacher confidence so
    unreliable paired-gradient regions do not dominate the learning signal.
    """
    if max_feature_length <= 1.0:
        raise ValueError("max_feature_length must be greater than 1")
    if predicted_double_angle.ndim != 4 or predicted_double_angle.shape[1] != 2:
        raise ValueError("predicted_double_angle must have shape (B, 2, H, W)")
    expected_scalar_shape = predicted_double_angle[:, :1].shape
    for name, value in (
        ("predicted_length", predicted_length),
        ("predicted_confidence", predicted_confidence),
        ("target_length", target_length),
        ("target_confidence", target_confidence),
    ):
        if value.shape != expected_scalar_shape:
            raise ValueError(f"{name} must have shape {tuple(expected_scalar_shape)}")
    if target_direction.shape != predicted_double_angle.shape:
        raise ValueError("target_direction must have shape (B, 2, H, W)")

    target_direction = F.normalize(target_direction.float(), dim=1, eps=1e-8)
    target_double_angle = torch.cat(
        (
            target_direction[:, :1].square() - target_direction[:, 1:2].square(),
            2.0 * target_direction[:, :1] * target_direction[:, 1:2],
        ),
        dim=1,
    )
    predicted_double_angle = F.normalize(predicted_double_angle.float(), dim=1, eps=1e-8)
    reliability = target_confidence.detach().float().clamp(0.0, 1.0)
    normalizer = reliability.sum().clamp_min(1e-8)

    cosine = (predicted_double_angle * target_double_angle).sum(dim=1, keepdim=True)
    direction_loss = ((1.0 - cosine) * reliability).sum() / normalizer
    normalized_length_error = (
        predicted_length.float() - target_length.float()
    ) / (max_feature_length - 1.0)
    length_loss = (
        F.smooth_l1_loss(
            normalized_length_error,
            torch.zeros_like(normalized_length_error),
            reduction="none",
        )
        * reliability
    ).sum() / normalizer
    confidence_loss = F.smooth_l1_loss(
        predicted_confidence.float(),
        target_confidence.detach().float(),
    )
    total = direction_loss + 2.0 * length_loss + 0.25 * confidence_loss
    return total, {
        "direction": direction_loss.detach(),
        "length": length_loss.detach(),
        "confidence": confidence_loss.detach(),
    }


def paired_direction_supervision_loss(
    predicted_double_angle: torch.Tensor,
    pseudo_double_angle: torch.Tensor,
    pseudo_confidence: torch.Tensor,
) -> torch.Tensor:
    """Compare patch-level directions after confidence-weighted circular pooling."""
    if predicted_double_angle.shape != pseudo_double_angle.shape:
        raise ValueError("predicted and pseudo direction maps must have the same shape")
    if pseudo_confidence.shape != predicted_double_angle[:, :1].shape:
        raise ValueError("pseudo confidence must have shape (B, 1, H, W)")
    predicted_sum = (predicted_double_angle.float() * pseudo_confidence).sum(dim=(-2, -1))
    pseudo_sum = (pseudo_double_angle * pseudo_confidence).sum(dim=(-2, -1))
    predicted_global = F.normalize(predicted_sum, dim=1, eps=1e-8)
    pseudo_global = F.normalize(pseudo_sum, dim=1, eps=1e-8)
    consistency = pseudo_sum.norm(dim=1) / pseudo_confidence.sum(dim=(-2, -1)).flatten().clamp_min(1e-8)
    cosine = (predicted_global * pseudo_global).sum(dim=1).clamp(-1.0, 1.0)
    return ((1.0 - cosine) * consistency.detach()).sum() / consistency.sum().clamp_min(1e-8)


@torch.no_grad()
def geometry_metrics(
    direction: torch.Tensor,
    length: torch.Tensor,
    target_direction: torch.Tensor,
    target_length: torch.Tensor,
) -> tuple[float, float]:
    target_direction = target_direction[:, :, None, None]
    target_length = target_length[:, None, None, None]
    target_double_angle = torch.cat(
        (
            target_direction[:, :1].square() - target_direction[:, 1:2].square(),
            2.0 * target_direction[:, :1] * target_direction[:, 1:2],
        ),
        dim=1,
    )
    cosine = (direction * target_double_angle).sum(dim=1).clamp(-1, 1)
    angle_error = (0.5 * torch.rad2deg(torch.acos(cosine))).mean().item()
    length_error = (length - target_length).abs().mean().item()
    return angle_error, length_error
