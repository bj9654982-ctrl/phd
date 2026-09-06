"""Motion-trajectory-conditioned nonlocal feature aggregation."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class TrajectoryNonlocalBlock(nn.Module):
    """Aggregate a fixed number of samples along a predicted local blur line.

    The geometry has only two degrees of freedom: unoriented line angle and
    length. No free 2-D residual offsets are learned.
    """

    def __init__(
        self,
        channels: int,
        samples: int = 5,
        max_feature_length: float = 7.0,
        reduction: int = 4,
        statistics_window: int = 51,
        fusion_mode: str = "aggregate",
        angle_hypotheses: int = 1,
        max_angle_offset_degrees: float = 30.0,
        direction_residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if channels < reduction:
            raise ValueError("channels must be at least reduction")
        if samples < 3 or samples % 2 == 0:
            raise ValueError("samples must be an odd integer >= 3")
        if max_feature_length <= 1.0:
            raise ValueError("max_feature_length must be greater than 1")
        if statistics_window < 3 or statistics_window % 2 == 0:
            raise ValueError("statistics_window must be an odd integer >= 3")
        if fusion_mode not in {"aggregate", "detail"}:
            raise ValueError("fusion_mode must be 'aggregate' or 'detail'")
        if angle_hypotheses < 1 or angle_hypotheses % 2 == 0:
            raise ValueError("angle_hypotheses must be a positive odd integer")
        if max_angle_offset_degrees < 0:
            raise ValueError("max_angle_offset_degrees must be nonnegative")
        if direction_residual_scale <= 0:
            raise ValueError("direction_residual_scale must be positive")

        hidden = max(channels // reduction, 8)
        self.samples = int(samples)
        self.max_feature_length = float(max_feature_length)
        self.statistics_window = int(statistics_window)
        self.fusion_mode = fusion_mode
        self.angle_hypotheses = int(angle_hypotheses)
        self.direction_residual_scale = float(direction_residual_scale)
        self.normalization = nn.GroupNorm(1, channels)
        self.geometry_features = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.statistics_projection = nn.Sequential(
            nn.Conv2d(4, hidden, 1),
            nn.GELU(),
        )
        self.direction_residual = nn.Conv2d(hidden, 2, 1)
        self.length_head = nn.Conv2d(hidden, 1, 1)
        self.confidence_head = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.direction_residual.weight)
        nn.init.zeros_(self.direction_residual.bias)
        self.query = nn.Conv2d(channels, hidden, 1)
        self.key_value = nn.Conv2d(channels, hidden * 2, 1)
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.relative_bias = nn.Parameter(torch.zeros(samples * angle_hypotheses))
        self.output = nn.Conv2d(hidden, channels, 1)
        self.residual_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        # Start as an exact identity insertion so the common U-Net has the same
        # initial function as the local baseline.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        positions = torch.linspace(-0.5, 0.5, samples)
        self.register_buffer("trajectory_positions", positions, persistent=False)
        if angle_hypotheses == 1:
            angle_offsets = torch.zeros(1)
        else:
            maximum_offset = math.radians(max_angle_offset_degrees)
            angle_offsets = torch.linspace(-maximum_offset, maximum_offset, angle_hypotheses)
        self.register_buffer("trajectory_angle_offsets", angle_offsets, persistent=False)

    def _dense_gradient_statistics(self, source: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        source = source.float()
        gray = source.mean(dim=1, keepdim=True)
        gradient_x = F.pad(gray[:, :, :, 2:] - gray[:, :, :, :-2], (1, 1, 0, 0), mode="replicate")
        gradient_y = F.pad(gray[:, :, 2:, :] - gray[:, :, :-2, :], (0, 0, 1, 1), mode="replicate")
        maximum_window = min(source.shape[-2:])
        if maximum_window % 2 == 0:
            maximum_window -= 1
        window = min(self.statistics_window, maximum_window)
        padding = window // 2
        source_height, source_width = source.shape[-2:]
        output_height, output_width = output_size
        stride = (source_height // output_height, source_width // output_width)
        if source_height != output_height * stride[0] or source_width != output_width * stride[1]:
            stride = (1, 1)
        jxx = F.avg_pool2d(gradient_x.square(), kernel_size=window, stride=stride, padding=padding)
        jyy = F.avg_pool2d(gradient_y.square(), kernel_size=window, stride=stride, padding=padding)
        jxy = F.avg_pool2d(gradient_x * gradient_y, kernel_size=window, stride=stride, padding=padding)
        energy = jxx + jyy + 1e-8
        orientation_x = (jyy - jxx) / energy
        orientation_y = (-2.0 * jxy) / energy
        anisotropy = torch.sqrt(orientation_x.square() + orientation_y.square() + 1e-8)
        normalized_log_energy = (torch.log(energy) + 6.0) / 2.0
        statistics = torch.cat((orientation_x, orientation_y, anisotropy, normalized_log_energy), dim=1)
        if statistics.shape[-2:] != output_size:
            statistics = F.interpolate(statistics, size=output_size, mode="bilinear", align_corners=False)
        return statistics

    def _predict_geometry(
        self,
        features: torch.Tensor,
        geometry_source: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if geometry_source is None:
            statistics = features.new_zeros((features.shape[0], 4, *features.shape[-2:]))
            statistics[:, 0] = 1.0
        else:
            statistics = self._dense_gradient_statistics(geometry_source, features.shape[-2:]).to(features.dtype)
        hidden = self.geometry_features(features) + self.statistics_projection(statistics)
        direction_prior = F.normalize(statistics[:, :2], dim=1, eps=1e-6)
        direction_delta = self.direction_residual_scale * torch.tanh(self.direction_residual(hidden))
        double_angle = F.normalize(direction_prior + direction_delta, dim=1, eps=1e-6)
        angle = 0.5 * torch.atan2(double_angle[:, 1:2], double_angle[:, :1])
        direction = torch.cat((angle.cos(), angle.sin()), dim=1)
        length = 1.0 + (self.max_feature_length - 1.0) * self.length_head(hidden).sigmoid()
        confidence = self.confidence_head(hidden).sigmoid()
        return double_angle, direction, length, confidence

    def _prepare_geometry_override(
        self,
        features: torch.Tensor,
        geometry_override: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert known line geometry to dense feature-space maps.

        ``length`` is expressed in feature pixels. This path exists for
        controlled oracle experiments and is not used during normal inference.
        """
        if "direction" not in geometry_override or "length" not in geometry_override:
            raise ValueError("geometry_override requires direction and length")

        batch, _, height, width = features.shape

        def dense_map(value: torch.Tensor, channels: int, name: str) -> torch.Tensor:
            value = value.to(device=features.device, dtype=features.dtype)
            if value.ndim == 1:
                value = value[:, None, None, None]
            elif value.ndim == 2:
                value = value[:, :, None, None]
            if value.ndim != 4 or value.shape[0] != batch or value.shape[1] != channels:
                raise ValueError(
                    f"{name} must have shape (B, {channels}) or (B, {channels}, H, W), "
                    f"got {tuple(value.shape)}"
                )
            if value.shape[-2:] == (1, 1):
                value = value.expand(-1, -1, height, width)
            elif value.shape[-2:] != (height, width):
                value = F.interpolate(value, size=(height, width), mode="bilinear", align_corners=False)
            return value

        direction = F.normalize(dense_map(geometry_override["direction"], 2, "direction"), dim=1, eps=1e-6)
        length = dense_map(geometry_override["length"], 1, "length").clamp(1.0, self.max_feature_length)
        if "confidence" in geometry_override:
            confidence = dense_map(geometry_override["confidence"], 1, "confidence").clamp(0.0, 1.0)
        else:
            confidence = torch.ones_like(length)
        double_angle = torch.cat(
            (
                direction[:, :1].square() - direction[:, 1:2].square(),
                2.0 * direction[:, :1] * direction[:, 1:2],
            ),
            dim=1,
        )
        return double_angle, direction, length, confidence

    def _sample_along_line(
        self,
        features: torch.Tensor,
        direction: torch.Tensor,
        length: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = features.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=features.device, dtype=features.dtype),
            torch.linspace(-1, 1, width, device=features.device, dtype=features.dtype),
            indexing="ij",
        )
        base = torch.stack((xx, yy), dim=-1)[None, None, None]
        positions = self.trajectory_positions.to(dtype=features.dtype).view(1, 1, self.samples, 1, 1)
        angle_offsets = self.trajectory_angle_offsets.to(dtype=features.dtype).view(
            1,
            self.angle_hypotheses,
            1,
            1,
            1,
        )
        direction_x = direction[:, :1].unsqueeze(1)
        direction_y = direction[:, 1:2].unsqueeze(1)
        rotated_x = direction_x * angle_offsets.cos() - direction_y * angle_offsets.sin()
        rotated_y = direction_x * angle_offsets.sin() + direction_y * angle_offsets.cos()
        extent = (length - 1.0).clamp_min(0.0).unsqueeze(1)
        offset_x = positions * extent * rotated_x * (2.0 / max(width - 1, 1))
        offset_y = positions * extent * rotated_y * (2.0 / max(height - 1, 1))
        offsets = torch.stack((offset_x, offset_y), dim=-1)
        total_samples = self.angle_hypotheses * self.samples
        grid = (base + offsets).reshape(batch * total_samples, height, width, 2)
        expanded = features[:, None, None].expand(
            batch,
            self.angle_hypotheses,
            self.samples,
            channels,
            height,
            width,
        )
        expanded = expanded.reshape(batch * total_samples, channels, height, width)
        sampled = F.grid_sample(
            expanded,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.reshape(batch, total_samples, channels, height, width)

    def forward(
        self,
        features: torch.Tensor,
        geometry_source: torch.Tensor | None = None,
        geometry_override: dict[str, torch.Tensor] | None = None,
        return_geometry: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        normalized = self.normalization(features)
        if geometry_override is None:
            double_angle, direction, length, confidence = self._predict_geometry(normalized, geometry_source)
        else:
            double_angle, direction, length, confidence = self._prepare_geometry_override(
                normalized,
                geometry_override,
            )
        query = self.query(normalized)
        key, value = self.key_value(normalized).chunk(2, dim=1)
        sampled = self._sample_along_line(torch.cat((key, value), dim=1), direction, length)
        sampled_key, sampled_value = sampled.chunk(2, dim=2)
        logits = (query[:, None] * sampled_key).sum(dim=2) / math.sqrt(query.shape[1])
        logits = logits + self.relative_bias[None, :, None, None]
        attention = logits.softmax(dim=1)
        trajectory = (attention[:, :, None] * sampled_value).sum(dim=1)
        local = self.local(value)
        if self.fusion_mode == "aggregate":
            fused = confidence * trajectory + (1.0 - confidence) * local
        else:
            directional_detail = value - trajectory
            fused = local + confidence * directional_detail
        restored_features = features + self.residual_scale * self.output(fused)

        if not return_geometry:
            return restored_features
        geometry = {
            "double_angle": double_angle,
            "direction": direction,
            "length": length,
            "confidence": confidence,
            "attention": attention,
        }
        return restored_features, geometry
