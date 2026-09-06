from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data import GoProPairDataset, discover_gopro_pairs, recording_group


def _write_rgb(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


class GoProDataTests(unittest.TestCase):
    def test_recording_group_removes_only_the_numeric_clip_suffix(self) -> None:
        self.assertEqual(recording_group("GOPR0374_11_03"), "GOPR0374_11")
        self.assertEqual(recording_group("scene_a"), "scene_a")

    def test_discovery_and_synchronized_crop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blur = root / "train" / "scene_a" / "blur_gamma" / "000001.png"
            sharp = root / "train" / "scene_a" / "sharp" / "000001.png"
            _write_rgb(blur, (20, 18), (10, 20, 30))
            _write_rgb(sharp, (20, 18), (10, 20, 30))

            pairs, issues = discover_gopro_pairs(root, splits=("train",))
            self.assertEqual(len(pairs), 1)
            self.assertEqual(issues, [])

            dataset = GoProPairDataset(root, split="train", patch_size=12, augment=True)
            sample = dataset[0]
            self.assertEqual(tuple(sample["blur"].shape), (3, 12, 12))
            self.assertTrue(sample["blur"].equal(sample["sharp"]))

    def test_missing_pair_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blur = root / "test" / "scene_b" / "blur" / "000001.png"
            sharp = root / "test" / "scene_b" / "sharp" / "000002.png"
            _write_rgb(blur, (16, 16), (0, 0, 0))
            _write_rgb(sharp, (16, 16), (0, 0, 0))

            pairs, issues = discover_gopro_pairs(root, splits=("test",))
            self.assertEqual(pairs, [])
            self.assertEqual({issue.kind for issue in issues}, {"missing_blur", "missing_sharp"})

    def test_validation_crop_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blur = root / "test" / "scene_c" / "blur_gamma" / "000001.png"
            sharp = root / "test" / "scene_c" / "sharp" / "000001.png"
            _write_rgb(blur, (21, 19), (40, 50, 60))
            _write_rgb(sharp, (21, 19), (40, 50, 60))

            dataset = GoProPairDataset(root, split="test", patch_size=12, augment=False)
            first = dataset[0]
            second = dataset[0]
            self.assertTrue(first["blur"].equal(second["blur"]))
            self.assertTrue(first["sharp"].equal(second["sharp"]))

    def test_limited_subset_spans_sorted_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for scene_index in range(3):
                for image_index in range(2):
                    name = f"{image_index:06d}.png"
                    blur = root / "train" / f"scene_{scene_index}" / "blur_gamma" / name
                    sharp = root / "train" / f"scene_{scene_index}" / "sharp" / name
                    _write_rgb(blur, (16, 16), (scene_index, image_index, 0))
                    _write_rgb(sharp, (16, 16), (scene_index, image_index, 0))

            dataset = GoProPairDataset(root, split="train", max_pairs=3)
            self.assertEqual({pair.scene for pair in dataset.pairs}, {"scene_0", "scene_1", "scene_2"})

    def test_recording_group_holdout_has_no_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenes = ("GOPR0001_01_00", "GOPR0001_01_01", "GOPR0002_01_00")
            for scene in scenes:
                blur = root / "train" / scene / "blur_gamma" / "000001.png"
                sharp = root / "train" / scene / "sharp" / "000001.png"
                _write_rgb(blur, (16, 16), (1, 2, 3))
                _write_rgb(sharp, (16, 16), (1, 2, 3))

            validation_groups = ("GOPR0001_01",)
            train_set = GoProPairDataset(
                root,
                split="train",
                exclude_recording_groups=validation_groups,
            )
            val_set = GoProPairDataset(
                root,
                split="train",
                include_recording_groups=validation_groups,
            )

            self.assertEqual(len(train_set), 1)
            self.assertEqual(len(val_set), 2)
            self.assertEqual(val_set.recording_groups, validation_groups)
            self.assertFalse(set(train_set.recording_groups) & set(val_set.recording_groups))

    def test_include_and_exclude_group_cannot_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = "GOPR0001_01_00"
            blur = root / "train" / scene / "blur_gamma" / "000001.png"
            sharp = root / "train" / scene / "sharp" / "000001.png"
            _write_rgb(blur, (16, 16), (1, 2, 3))
            _write_rgb(sharp, (16, 16), (1, 2, 3))

            with self.assertRaisesRegex(ValueError, "both included and excluded"):
                GoProPairDataset(
                    root,
                    split="train",
                    include_recording_groups=("GOPR0001_01",),
                    exclude_recording_groups=("GOPR0001_01",),
                )


if __name__ == "__main__":
    unittest.main()
