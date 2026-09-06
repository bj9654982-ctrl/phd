"""Render a compact qualitative grid from a trained GoPro baseline checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GoProPairDataset  # noqa: E402
from src.metrics import psnr  # noqa: E402
from src.models import CompactDeblurUNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GoPro baseline examples from a checkpoint.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    return parser.parse_args()


def _to_image(tensor: torch.Tensor):
    return tensor.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script has no CPU fallback")
    if args.samples <= 0:
        raise ValueError("samples must be positive")

    device = torch.device("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    if checkpoint_args.get("validation_source") != "train_holdout":
        raise ValueError("Checkpoint was not trained with validation_source=train_holdout")
    validation_groups = tuple(checkpoint_args.get("validation_groups") or ())
    if not validation_groups:
        raise ValueError("Checkpoint does not contain frozen validation recording groups")
    width = int(checkpoint_args["width"])
    model = CompactDeblurUNet(width=width).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    dataset = GoProPairDataset(
        args.data_root,
        split="train",
        patch_size=args.patch_size,
        augment=False,
        max_pairs=args.samples,
        include_recording_groups=validation_groups,
    )

    figure, axes = plt.subplots(len(dataset), 3, figsize=(12, 4 * len(dataset)), squeeze=False)
    records: list[dict[str, float | str]] = []
    for row, sample in enumerate(dataset):
        blurry = sample["blur"].unsqueeze(0).to(device)
        sharp = sample["sharp"].unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            restored = model(blurry)
        input_psnr = psnr(blurry, sharp)
        output_psnr = psnr(restored, sharp)
        records.append(
            {
                "name": str(sample["name"]),
                "input_psnr_db": input_psnr,
                "output_psnr_db": output_psnr,
                "gain_db": output_psnr - input_psnr,
            }
        )

        images = (blurry[0], restored[0], sharp[0])
        titles = (
            f"Blurry | {input_psnr:.2f} dB",
            f"Baseline | {output_psnr:.2f} dB",
            f"Sharp | {sample['name']}",
        )
        for column, (image, title) in enumerate(zip(images, titles)):
            axes[row, column].imshow(_to_image(image))
            axes[row, column].set_title(title, fontsize=9)
            axes[row, column].axis("off")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output / "comparison_grid.png", dpi=160)
    plt.close(figure)
    (output / "example_metrics.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(records, indent=2, ensure_ascii=False))
    print(output / "comparison_grid.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
