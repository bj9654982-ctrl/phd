"""Controlled fixed, Top-K, conditioned, and protected nonlocal blocks.

The three modes intentionally share exactly the same learned parameters.  This
lets the experiments isolate the aggregation rule instead of silently changing
model capacity.  All modes use a zero-initialized output projection, so inserting
the block into a trained local U-Net preserves the baseline function at step 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class MultiScaleNonlocalBlock(nn.Module):
    """Aggregate fixed multi-scale neighbours with optional sparse conditioning.

    ``fixed`` attends to every candidate with uniform scale priors.
    ``topk`` keeps only the most content-similar candidates.
    ``scale_adaptive`` predicts only per-pixel scale weights while retaining a
    fixed 0.5 fusion gate. ``gate_adaptive`` uses uniform scale priors and predicts
    only the local versus nonlocal gate. ``adaptive`` enables both condition paths.
    ``protected_gate_adaptive`` additionally multiplies the entire nonlocal module
    residual by that gate, so a closed gate is an exact identity intervention.
    All condition predictions use image degradation statistics and latent features.

    Candidate locations are still gathered with a dense tensor implementation;
    ``topk`` therefore describes sparse aggregation, not a measured runtime
    reduction.  Complexity claims must use measured FLOPs/runtime separately.
    """

    MODES = {
        "fixed",
        "topk",
        "scale_adaptive",
        "gate_adaptive",
        "protected_gate_adaptive",
        "adaptive",
    }

    def __init__(
        self,
        channels: int,
        scales: Sequence[int] = (1, 3, 7),
        top_k: int = 6,
        reduction: int = 4,
        statistics_window: int = 5,
        mode: str = "adaptive",
    ) -> None:
        super().__init__()
        scales = tuple(int(scale) for scale in scales)
        if channels < reduction:
            raise ValueError("channels must be at least reduction")
        if not scales or any(scale <= 0 for scale in scales):
            raise ValueError("scales must contain positive integers")
        if len(set(scales)) != len(scales):
            raise ValueError("scales must be unique")
        if tuple(sorted(scales)) != scales:
            raise ValueError("scales must be strictly increasing")
        if statistics_window < 3 or statistics_window % 2 == 0:
            raise ValueError("statistics_window must be an odd integer >= 3")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")

        # Eight fixed directions at every radius.  They form a controlled,
        # reference-style large support without quadratic global attention.
        unit_offsets = (
            (-1.0, -1.0),
            (0.0, -1.0),
            (1.0, -1.0),
            (-1.0, 0.0),
            (1.0, 0.0),
            (-1.0, 1.0),
            (0.0, 1.0),
            (1.0, 1.0),
        )
        offsets = [
            (dx * scale, dy * scale)
            for scale in scales
            for dx, dy in unit_offsets
        ]
        candidate_count = len(offsets)
        if top_k <= 0 or top_k > candidate_count:
            raise ValueError(f"top_k must be in [1, {candidate_count}]")

        hidden = max(channels // reduction, 8)
        self.channels = int(channels)
        self.hidden_channels = int(hidden)
        self.scales = scales
        self.top_k = int(top_k)
        self.statistics_window = int(statistics_window)
        self.mode = mode

        self.normalization = nn.GroupNorm(1, channels)
        self.query = nn.Conv2d(channels, hidden, 1)
        self.key_value = nn.Conv2d(channels, hidden * 2, 1)
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)

        # The conditioning branch exists in every mode so B1/B2/B3 have exactly
        # the same parameter count.  Fixed and Top-K controls simply do not use it.
        self.condition_features = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.statistics_projection = nn.Sequential(
            nn.Conv2d(3, hidden, 1),
            nn.GELU(),
        )
        self.condition_head = nn.Conv2d(hidden, len(scales) + 1, 1)
        nn.init.zeros_(self.condition_head.weight)
        nn.init.zeros_(self.condition_head.bias)

        self.relative_bias = nn.Parameter(torch.zeros(candidate_count))
        self.output = nn.Conv2d(hidden, channels, 1)
        self.residual_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        self.register_buffer(
            "candidate_offsets",
            torch.tensor(offsets, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "candidate_scale_indices",
            torch.arange(len(scales), dtype=torch.long).repeat_interleave(len(unit_offsets)),
            persistent=False,
        )

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_offsets.shape[0])

    def _degradation_statistics(
        self,
        source: torch.Tensor | None,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Return gradient, high-pass, and local-contrast cues at feature scale."""
        if source is None:
            return features.new_zeros((features.shape[0], 3, *features.shape[-2:]))
        if source.ndim != 4 or source.shape[0] != features.shape[0]:
            raise ValueError("geometry_source must be BCHW with the same batch size as features")

        gray = source.float().mean(dim=1, keepdim=True)
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
        gradient = torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-8)
        padding = self.statistics_window // 2
        local_mean = F.avg_pool2d(
            gray,
            kernel_size=self.statistics_window,
            stride=1,
            padding=padding,
        )
        local_square_mean = F.avg_pool2d(
            gray.square(),
            kernel_size=self.statistics_window,
            stride=1,
            padding=padding,
        )
        high_pass = (gray - local_mean).abs()
        local_contrast = torch.sqrt(
            (local_square_mean - local_mean.square()).clamp_min(0.0) + 1e-8
        )
        statistics = torch.cat((gradient, high_pass, local_contrast), dim=1)
        statistics = F.interpolate(
            statistics,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return statistics.to(dtype=features.dtype)

    def _sample_candidates(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=features.device, dtype=features.dtype),
            torch.linspace(-1.0, 1.0, width, device=features.device, dtype=features.dtype),
            indexing="ij",
        )
        base = torch.stack((xx, yy), dim=-1)[None, None]
        offsets = self.candidate_offsets.to(device=features.device, dtype=features.dtype)
        normalized_offsets = torch.stack(
            (
                offsets[:, 0] * (2.0 / max(width - 1, 1)),
                offsets[:, 1] * (2.0 / max(height - 1, 1)),
            ),
            dim=-1,
        )
        grid = base + normalized_offsets[None, :, None, None]
        grid = grid.expand(batch, -1, -1, -1, -1).reshape(
            batch * self.candidate_count,
            height,
            width,
            2,
        )
        expanded = features[:, None].expand(
            batch,
            self.candidate_count,
            channels,
            height,
            width,
        )
        expanded = expanded.reshape(batch * self.candidate_count, channels, height, width)
        sampled = F.grid_sample(
            expanded,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.reshape(batch, self.candidate_count, channels, height, width)

    def _condition(
        self,
        normalized: torch.Tensor,
        geometry_source: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        statistics = self._degradation_statistics(geometry_source, normalized)
        hidden = self.condition_features(normalized) + self.statistics_projection(statistics)
        condition = self.condition_head(hidden)
        scale_weights = condition[:, : len(self.scales)].softmax(dim=1)
        nonlocal_gate = condition[:, len(self.scales) :].sigmoid()
        return scale_weights, nonlocal_gate, statistics

    def forward(
        self,
        features: torch.Tensor,
        geometry_source: torch.Tensor | None = None,
        geometry_override: dict[str, torch.Tensor] | None = None,
        return_geometry: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(
                f"expected BCHW features with {self.channels} channels, got {tuple(features.shape)}"
            )
        if geometry_override is not None:
            raise ValueError("geometry_override is not defined for multi-scale nonlocal attention")

        normalized = self.normalization(features)
        query = self.query(normalized)
        key, value = self.key_value(normalized).chunk(2, dim=1)
        sampled = self._sample_candidates(torch.cat((key, value), dim=1))
        sampled_key, sampled_value = sampled.chunk(2, dim=2)
        logits = (query[:, None] * sampled_key).sum(dim=2) / math.sqrt(query.shape[1])
        logits = logits + self.relative_bias[None, :, None, None]

        conditioned_modes = {
            "scale_adaptive",
            "gate_adaptive",
            "protected_gate_adaptive",
            "adaptive",
        }
        if self.mode in conditioned_modes:
            learned_scale_weights, learned_nonlocal_gate, statistics = self._condition(
                normalized,
                geometry_source,
            )
            batch, _, height, width = features.shape
            scale_weights = (
                learned_scale_weights
                if self.mode in {"scale_adaptive", "adaptive"}
                else features.new_full(
                    (batch, len(self.scales), height, width),
                    1.0 / len(self.scales),
                )
            )
            nonlocal_gate = (
                learned_nonlocal_gate
                if self.mode in {"gate_adaptive", "protected_gate_adaptive", "adaptive"}
                else features.new_full((batch, 1, height, width), 0.5)
            )
        else:
            batch, _, height, width = features.shape
            scale_weights = features.new_full(
                (batch, len(self.scales), height, width),
                1.0 / len(self.scales),
            )
            nonlocal_gate = features.new_full((batch, 1, height, width), 0.5)
            statistics = self._degradation_statistics(geometry_source, normalized)

        if self.mode in {"scale_adaptive", "adaptive"}:
            candidate_priors = scale_weights[:, self.candidate_scale_indices]
            logits = logits + candidate_priors.clamp_min(1e-6).log()

        if self.mode == "fixed":
            selected_mask = torch.ones_like(logits, dtype=torch.bool)
        else:
            indices = logits.topk(self.top_k, dim=1).indices
            selected_mask = torch.zeros_like(logits, dtype=torch.bool)
            selected_mask.scatter_(1, indices, True)
            logits = logits.masked_fill(~selected_mask, torch.finfo(logits.dtype).min)

        attention = logits.softmax(dim=1)
        nonlocal_features = (attention[:, :, None] * sampled_value).sum(dim=1)
        local_features = self.local(value)
        if self.mode in {"gate_adaptive", "protected_gate_adaptive", "adaptive"}:
            fused = (1.0 - nonlocal_gate) * local_features + nonlocal_gate * nonlocal_features
        else:
            fused = 0.5 * (local_features + nonlocal_features)
        intervention_gate = (
            nonlocal_gate
            if self.mode == "protected_gate_adaptive"
            else features.new_ones((batch, 1, height, width))
        )
        restored = features + intervention_gate * self.residual_scale * self.output(fused)

        if not return_geometry:
            return restored
        diagnostics = {
            "attention": attention,
            "selected_mask": selected_mask,
            "scale_weights": scale_weights,
            "nonlocal_gate": nonlocal_gate,
            "intervention_gate": intervention_gate,
            "degradation_statistics": statistics,
        }
        return restored, diagnostics
