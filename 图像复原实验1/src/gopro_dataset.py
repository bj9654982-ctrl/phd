"""
GOPRO 动态场景去模糊数据集加载器
支持标准 GOPRO_Large 目录结构：
    root/
        train/
            scene_001/
                blur/
                    000001.png
                sharp/
                    000001.png
        test/
            ...
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class GOPRODataset(Dataset):
    """GOPRO 动态场景去模糊数据集，加载真实 blur/sharp 图像对。"""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        patch_size: int = 256,
        augment: bool = True,
    ):
        """
        Args:
            root: GOPRO_Large 根目录（包含 train/ 和 test/）
            split: 'train' 或 'test'
            patch_size: 训练时裁剪的 patch 大小
            augment: 是否进行数据增强（随机翻转、旋转）
        """
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.augment = augment and (split == "train")

        self.pairs: List[Tuple[Path, Path]] = self._scan_pairs()
        if len(self.pairs) == 0:
            raise RuntimeError(f"No blur/sharp pairs found in {self.root / split}")

    def _scan_pairs(self) -> List[Tuple[Path, Path]]:
        """扫描所有场景，收集 (blur_path, sharp_path) 对。"""
        pairs = []
        split_dir = self.root / self.split
        if not split_dir.is_dir():
            return pairs

        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            blur_dir = scene_dir / "blur"
            sharp_dir = scene_dir / "sharp"
            if not blur_dir.is_dir() or not sharp_dir.is_dir():
                continue

            blur_images = sorted(blur_dir.glob("*.png"))
            for blur_path in blur_images:
                sharp_path = sharp_dir / blur_path.name
                if sharp_path.is_file():
                    pairs.append((blur_path, sharp_path))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _random_crop(
        self, blur: Image.Image, sharp: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """对 blur/sharp 同步随机裁剪。"""
        w, h = blur.size
        if w < self.patch_size or h < self.patch_size:
            # 图像太小，先放大
            scale = max(self.patch_size / w, self.patch_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            blur = blur.resize((new_w, new_h), Image.BICUBIC)
            sharp = sharp.resize((new_w, new_h), Image.BICUBIC)
            w, h = new_w, new_h

        left = random.randint(0, w - self.patch_size)
        top = random.randint(0, h - self.patch_size)
        blur_crop = blur.crop((left, top, left + self.patch_size, top + self.patch_size))
        sharp_crop = sharp.crop((left, top, left + self.patch_size, top + self.patch_size))
        return blur_crop, sharp_crop

    def _augment(
        self, blur: Image.Image, sharp: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """同步数据增强：水平翻转 + 90度旋转。"""
        if random.random() < 0.5:
            blur = blur.transpose(Image.FLIP_LEFT_RIGHT)
            sharp = sharp.transpose(Image.FLIP_LEFT_RIGHT)

        rotations = random.randint(0, 3)
        if rotations > 0:
            blur = blur.rotate(rotations * 90)
            sharp = sharp.rotate(rotations * 90)

        return blur, sharp

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        blur_path, sharp_path = self.pairs[index]

        with Image.open(blur_path) as f:
            blur = f.convert("RGB")
        with Image.open(sharp_path) as f:
            sharp = f.convert("RGB")

        if self.split == "train":
            blur, sharp = self._random_crop(blur, sharp)
            if self.augment:
                blur, sharp = self._augment(blur, sharp)
        # test 时不裁剪，返回全图（评估时整图推理）

        blur_tensor = self._pil_to_tensor(blur)
        sharp_tensor = self._pil_to_tensor(sharp)
        return blur_tensor, sharp_tensor


class GOPROTestDataset(Dataset):
    """GOPRO 测试集，返回全图及文件名，用于评估。"""

    def __init__(self, root: str | Path):
        self.root = Path(root) / "test"
        self.pairs: List[Tuple[Path, Path, str]] = self._scan_pairs()

    def _scan_pairs(self):
        pairs = []
        for scene_dir in sorted(self.root.iterdir()):
            if not scene_dir.is_dir():
                continue
            blur_dir = scene_dir / "blur"
            sharp_dir = scene_dir / "sharp"
            if not blur_dir.is_dir() or not sharp_dir.is_dir():
                continue
            for blur_path in sorted(blur_dir.glob("*.png")):
                sharp_path = sharp_dir / blur_path.name
                if sharp_path.is_file():
                    pairs.append((blur_path, sharp_path, f"{scene_dir.name}/{blur_path.name}"))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        blur_path, sharp_path, name = self.pairs[index]
        with Image.open(blur_path) as f:
            blur = f.convert("RGB")
        with Image.open(sharp_path) as f:
            sharp = f.convert("RGB")
        blur_t = GOPRODataset._pil_to_tensor(blur)
        sharp_t = GOPRODataset._pil_to_tensor(sharp)
        return blur_t, sharp_t, name
