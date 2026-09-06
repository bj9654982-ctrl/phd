"""Evaluate a trained local baseline on its frozen GoPro training holdout."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GoProPairDataset, recording_group  # noqa: E402
from src.metrics import psnr  # noqa: E402
from src.models import CompactDeblurUNet  # noqa: E402


DEFAULT_RUN = PROJECT_ROOT / "results" / "runs" / "P3_baseline_group_split_v0"
METRIC_FIELDS = (
    "input_psnr_db",
    "restored_psnr_db",
    "psnr_gain_db",
    "input_ssim",
    "restored_ssim",
    "ssim_gain",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate B0 on its frozen training holdout without touching the official GoPro test split."
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "GOPRO_Large")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN / "best.pt")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN / "holdout_evaluation")
    parser.add_argument("--max-pairs", type=int, default=0, help="0 evaluates the complete frozen holdout.")
    parser.add_argument("--visual-samples", type=int, default=9)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def _image_ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction_array = prediction.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    target_array = target.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return float(
        structural_similarity(
            target_array,
            prediction_array,
            data_range=1.0,
            channel_axis=2,
        )
    )


def _uniform_indices(indices: list[int], maximum: int) -> list[int]:
    if maximum >= len(indices):
        return indices
    if maximum == 1:
        return [indices[len(indices) // 2]]
    positions = [(index * (len(indices) - 1)) // (maximum - 1) for index in range(maximum)]
    return [indices[position] for position in positions]


def _balanced_visual_indices(dataset: GoProPairDataset, maximum: int) -> set[int]:
    if maximum <= 0:
        return set()
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(dataset.pairs):
        grouped[recording_group(pair.scene)].append(index)
    group_names = sorted(grouped)
    base, remainder = divmod(min(maximum, len(dataset)), len(group_names))
    selected: set[int] = set()
    for group_index, group_name in enumerate(group_names):
        quota = base + (1 if group_index < remainder else 0)
        if quota:
            selected.update(_uniform_indices(grouped[group_name], quota))
    return selected


def _aggregate(records: list[dict[str, float | str]]) -> dict[str, float | int]:
    return {
        "pairs": len(records),
        **{
            field: statistics.fmean(float(record[field]) for record in records)
            for field in METRIC_FIELDS
        },
    }


def _write_csv(path: Path, records: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _render_grid(
    output: Path,
    examples: list[tuple[dict[str, float | str], torch.Tensor, torch.Tensor, torch.Tensor]],
) -> None:
    if not examples:
        return
    figure, axes = plt.subplots(len(examples), 4, figsize=(14, 3.4 * len(examples)), squeeze=False)
    for row, (record, blurry, restored, sharp) in enumerate(examples):
        rgb_images = (blurry, restored, sharp)
        titles = (
            f"Blurry | {float(record['input_psnr_db']):.2f} dB | {float(record['input_ssim']):.4f}",
            f"Restored | {float(record['restored_psnr_db']):.2f} dB | {float(record['restored_ssim']):.4f}",
            f"Sharp | {record['name']}",
        )
        for column, (image, title) in enumerate(zip(rgb_images, titles)):
            axes[row, column].imshow(image.clamp(0, 1).permute(1, 2, 0).numpy())
            axes[row, column].set_title(title, fontsize=8)
            axes[row, column].axis("off")
        absolute_error = (restored - sharp).abs().mean(dim=0).numpy()
        axes[row, 3].imshow(absolute_error, cmap="magma", vmin=0.0, vmax=0.15)
        axes[row, 3].set_title(
            f"|Restored - Sharp| | gain {float(record['psnr_gain_db']):+.2f} dB",
            fontsize=8,
        )
        axes[row, 3].axis("off")
    figure.tight_layout()
    figure.savefig(output / "qualitative_grid.png", dpi=160)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script has no CPU fallback")
    if args.max_pairs < 0 or args.visual_samples < 0 or args.num_workers < 0:
        raise ValueError("max-pairs, visual-samples, and num-workers must be nonnegative")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    device = torch.device("cuda:0")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    if checkpoint_args.get("variant") != "local":
        raise ValueError("This evaluator accepts only the local B0 baseline checkpoint")
    if checkpoint_args.get("validation_source") != "train_holdout":
        raise ValueError("Checkpoint was not trained with validation_source=train_holdout")
    validation_groups = tuple(checkpoint_args.get("validation_groups") or ())
    if not validation_groups:
        raise ValueError("Checkpoint does not contain frozen validation recording groups")

    patch_size = int(checkpoint_args["val_patch_size"])
    dataset = GoProPairDataset(
        args.data_root,
        split="train",
        patch_size=patch_size,
        augment=False,
        max_pairs=args.max_pairs or None,
        include_recording_groups=validation_groups,
    )
    if set(dataset.recording_groups) != set(validation_groups):
        raise RuntimeError("Loaded holdout groups do not match the groups frozen in the checkpoint")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    model = CompactDeblurUNet(width=int(checkpoint_args["width"])).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)

    visual_indices = _balanced_visual_indices(dataset, args.visual_samples)
    examples: list[tuple[dict[str, float | str], torch.Tensor, torch.Tensor, torch.Tensor]] = []
    records: list[dict[str, float | str]] = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for index, batch in enumerate(tqdm(loader, desc="B0 frozen holdout")):
        blurry = batch["blur"].to(device, non_blocking=True)
        sharp = batch["sharp"].to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            restored = model(blurry)

        pair = dataset.pairs[index]
        input_psnr = psnr(blurry, sharp)
        restored_psnr = psnr(restored, sharp)
        input_ssim = _image_ssim(blurry[0], sharp[0])
        restored_ssim = _image_ssim(restored[0], sharp[0])
        record: dict[str, float | str] = {
            "recording_group": recording_group(pair.scene),
            "scene": pair.scene,
            "name": f"{pair.scene}/{pair.name}",
            "input_psnr_db": input_psnr,
            "restored_psnr_db": restored_psnr,
            "psnr_gain_db": restored_psnr - input_psnr,
            "input_ssim": input_ssim,
            "restored_ssim": restored_ssim,
            "ssim_gain": restored_ssim - input_ssim,
        }
        records.append(record)
        if index in visual_indices:
            examples.append(
                (
                    record,
                    blurry[0].detach().float().clamp(0, 1).cpu(),
                    restored[0].detach().float().clamp(0, 1).cpu(),
                    sharp[0].detach().float().clamp(0, 1).cpu(),
                )
            )

    elapsed_seconds = time.perf_counter() - started
    grouped_records: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for record in records:
        grouped_records[str(record["recording_group"])].append(record)
    group_summaries: list[dict[str, float | int | str]] = [
        {"recording_group": group_name, **_aggregate(group_records)}
        for group_name, group_records in sorted(grouped_records.items())
    ]
    overall = _aggregate(records)

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_image_metrics.csv", records)
    _write_csv(output / "per_group_metrics.csv", group_summaries)
    _render_grid(output, examples)
    summary = {
        "experiment": "P3_baseline_group_split_v0_holdout_evaluation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "data_root": str(args.data_root.expanduser().resolve()),
        "source_split": "train",
        "validation_recording_groups": list(validation_groups),
        "patch_size": patch_size,
        "complete_holdout": args.max_pairs == 0,
        "overall": overall,
        "per_group": group_summaries,
        "visualized_pairs": [str(record[0]["name"]) for record in examples],
        "elapsed_seconds": elapsed_seconds,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "official_test_accessed": False,
        "paper_result": False,
        "note": "Frozen train-holdout center-crop evaluation for method development; not an official GoPro test result.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(output / "qualitative_grid.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
