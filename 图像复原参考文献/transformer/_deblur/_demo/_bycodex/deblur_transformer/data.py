from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

from .io_utils import list_images, pil_to_tensor


def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    coordinates = torch.arange(size, dtype=torch.float32) - size // 2
    kernel_1d = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d)


def motion_kernel(size: int, angle_degrees: float) -> torch.Tensor:
    """Create an anti-aliased line kernel without requiring OpenCV."""
    kernel = np.zeros((size, size), dtype=np.float32)
    center = (size - 1) / 2.0
    angle = math.radians(angle_degrees)
    samples = max(16, size * 8)
    for offset in np.linspace(-center, center, samples):
        x = center + offset * math.cos(angle)
        y = center + offset * math.sin(angle)
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        dx, dy = x - x0, y - y0
        for yy, wy in ((y0, 1.0 - dy), (y0 + 1, dy)):
            for xx, wx in ((x0, 1.0 - dx), (x0 + 1, dx)):
                if 0 <= xx < size and 0 <= yy < size:
                    kernel[yy, xx] += wx * wy
    kernel /= max(float(kernel.sum()), 1e-8)
    return torch.from_numpy(kernel)


def apply_kernel(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    channels = image.shape[0]
    padding = kernel.shape[-1] // 2
    padded = F.pad(image.unsqueeze(0), (padding,) * 4, mode="reflect")
    weight = kernel.to(image).expand(channels, 1, -1, -1)
    return F.conv2d(padded, weight, groups=channels).squeeze(0)


def random_blur(image: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
    if random.random() < 0.5:
        size = random.choice((3, 5, 7, 9, 11))
        sigma = random.uniform(0.6, max(0.7, size / 3.0))
        kernel = gaussian_kernel(size, sigma)
    else:
        size = random.choice((5, 7, 9, 11, 13, 15))
        kernel = motion_kernel(size, random.uniform(0.0, 180.0))
    blurred = apply_kernel(image, kernel)
    if noise_std > 0:
        blurred = blurred + torch.randn_like(blurred) * random.uniform(0.0, noise_std)
    return blurred.clamp(0.0, 1.0)


class SyntheticBlurDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load sharp images and synthesize blur on the fly."""

    def __init__(self, root: str | Path, patch_size: int = 128, noise_std: float = 0.01) -> None:
        self.images = list_images(root)
        self.patch_size = patch_size
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.images)

    def _load_crop(self, path: Path) -> Image.Image:
        with Image.open(path) as source:
            image = source.convert("RGB")
        width, height = image.size
        scale = max(self.patch_size / width, self.patch_size / height, 1.0)
        if scale > 1.0:
            image = image.resize(
                (math.ceil(width * scale), math.ceil(height * scale)),
                Image.Resampling.BICUBIC,
            )
        width, height = image.size
        left = random.randint(0, width - self.patch_size)
        top = random.randint(0, height - self.patch_size)
        image = image.crop((left, top, left + self.patch_size, top + self.patch_size))
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        rotations = random.randint(0, 3)
        if rotations:
            image = image.rotate(rotations * 90)
        return image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sharp = pil_to_tensor(self._load_crop(self.images[index]))
        blurred = random_blur(sharp, self.noise_std)
        return blurred, sharp
