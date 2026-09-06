"""P6-D: attribute predicted-geometry failure and measure angle tolerance."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GoProPairDataset, recording_group  # noqa: E402
from src.models import (  # noqa: E402
    CompactDeblurUNet,
    TrajectoryNonlocalBlock,
    paired_motion_geometry_pseudo_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the P6 geometry error budget.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "GOPRO_Large")
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "P3_baseline_group_split_v0" / "best.pt",
    )
    parser.add_argument(
        "--predicted-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "P6C_wide_direction_geometry_v0" / "best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "P6D_geometry_error_budget_v0",
    )
    parser.add_argument("--val-pairs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--oracle-window", type=int, default=31)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def _feature_size(image: torch.Tensor) -> tuple[int, int]:
    height, width = image.shape[-2:]
    padded_height = height + (4 - height % 4) % 4
    padded_width = width + (4 - width % 4) % 4
    return padded_height // 4, padded_width // 4


@torch.no_grad()
def _paired_geometry(
    blurry: torch.Tensor,
    sharp: torch.Tensor,
    window: int,
    max_feature_length: float,
) -> dict[str, torch.Tensor]:
    direction, length, confidence = paired_motion_geometry_pseudo_label(
        blurry,
        sharp,
        output_size=_feature_size(blurry),
        window=window,
        max_feature_length=max_feature_length,
    )
    return {"direction": direction, "length": length, "confidence": confidence}


def rotate_direction(direction: torch.Tensor, degrees: float) -> torch.Tensor:
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return torch.cat(
        (
            cosine * direction[:, :1] - sine * direction[:, 1:2],
            sine * direction[:, :1] + cosine * direction[:, 1:2],
        ),
        dim=1,
    )


def _psnr_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    mse = (prediction - target).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return -10.0 * mse.log10()


def _write_csv(path: Path, records: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _make_predicted_model(
    checkpoint: dict[str, object],
    width: int,
) -> CompactDeblurUNet:
    oracle_args = checkpoint["oracle_args"]
    run_args = checkpoint["args"]
    block = TrajectoryNonlocalBlock(
        channels=width * 4,
        samples=int(oracle_args["trajectory_samples"]),
        max_feature_length=float(oracle_args["max_feature_length"]),
        fusion_mode=str(oracle_args["trajectory_fusion"]),
        angle_hypotheses=1,
        direction_residual_scale=float(run_args["direction_residual_scale"]),
    )
    model = CompactDeblurUNet(width=width, bottleneck_module=block)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script has no CPU fallback")
    if args.val_pairs < 0 or args.num_workers < 0 or args.batch_size <= 0:
        raise ValueError("val-pairs and num-workers must be nonnegative; batch-size must be positive")
    if args.oracle_window < 3 or args.oracle_window % 2 == 0:
        raise ValueError("oracle-window must be an odd integer >= 3")

    baseline_path = args.baseline_checkpoint.expanduser().resolve()
    predicted_path = args.predicted_checkpoint.expanduser().resolve()
    if not baseline_path.is_file() or not predicted_path.is_file():
        raise FileNotFoundError("baseline and predicted checkpoints must both exist")
    baseline_checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=False)
    predicted_checkpoint = torch.load(predicted_path, map_location="cpu", weights_only=False)
    baseline_args = baseline_checkpoint["args"]
    validation_groups = tuple(baseline_args.get("validation_groups") or ())
    if not validation_groups:
        raise ValueError("The baseline checkpoint does not contain frozen validation groups")

    validation_set = GoProPairDataset(
        args.data_root,
        split="train",
        patch_size=int(baseline_args["val_patch_size"]),
        augment=False,
        max_pairs=args.val_pairs or None,
        include_recording_groups=validation_groups,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device("cuda:0")
    width = int(baseline_args["width"])
    baseline = CompactDeblurUNet(width=width)
    baseline.load_state_dict(baseline_checkpoint["model"], strict=True)
    baseline = baseline.to(device).eval()
    model = _make_predicted_model(predicted_checkpoint, width).to(device).eval()
    max_feature_length = float(predicted_checkpoint["oracle_args"]["max_feature_length"])

    angle_offsets = (5, 10, 20, 30, 45, 90)
    output_fields = (
        "baseline",
        "predicted_all",
        "oracle_all",
        "direction_error_only",
        "length_error_only",
        "confidence_error_only",
        *(f"oracle_angle_plus_{angle}" for angle in angle_offsets),
    )
    records: list[dict[str, float | str]] = []
    started = time.perf_counter()
    for batch in tqdm(validation_loader, desc="P6-D geometry error budget"):
        blurry = batch["blur"].to(device, non_blocking=True)
        sharp = batch["sharp"].to(device, non_blocking=True)
        oracle = _paired_geometry(blurry, sharp, args.oracle_window, max_feature_length)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs: dict[str, torch.Tensor] = {"baseline": baseline(blurry)}
            outputs["predicted_all"], predicted = model(blurry, return_geometry=True)
            outputs["oracle_all"] = model(blurry, geometry_override=oracle)
            outputs["direction_error_only"] = model(
                blurry,
                geometry_override={
                    "direction": predicted["direction"],
                    "length": oracle["length"],
                    "confidence": oracle["confidence"],
                },
            )
            outputs["length_error_only"] = model(
                blurry,
                geometry_override={
                    "direction": oracle["direction"],
                    "length": predicted["length"],
                    "confidence": oracle["confidence"],
                },
            )
            outputs["confidence_error_only"] = model(
                blurry,
                geometry_override={
                    "direction": oracle["direction"],
                    "length": oracle["length"],
                    "confidence": predicted["confidence"],
                },
            )
            for angle in angle_offsets:
                outputs[f"oracle_angle_plus_{angle}"] = model(
                    blurry,
                    geometry_override={
                        "direction": rotate_direction(oracle["direction"], angle),
                        "length": oracle["length"],
                        "confidence": oracle["confidence"],
                    },
                )
        psnr = {name: _psnr_per_image(value, sharp).cpu() for name, value in outputs.items()}
        for index, name in enumerate(batch["name"]):
            scene = str(name).split("/", maxsplit=1)[0]
            record: dict[str, float | str] = {
                "recording_group": recording_group(scene),
                "scene": scene,
                "name": str(name),
            }
            baseline_psnr = float(psnr["baseline"][index])
            for field in output_fields:
                value = float(psnr[field][index])
                record[f"{field}_psnr_db"] = value
                if field != "baseline":
                    record[f"{field}_minus_baseline_db"] = value - baseline_psnr
            records.append(record)

    metric_fields = [key for key in records[0] if key.endswith("_db")]

    def aggregate(items: list[dict[str, float | str]]) -> dict[str, float | int]:
        return {
            "pairs": len(items),
            **{
                field: statistics.fmean(float(item[field]) for item in items)
                for field in metric_fields
            },
        }

    overall = aggregate(records)
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for record in records:
        grouped[str(record["recording_group"])].append(record)
    per_group: list[dict[str, float | int | str]] = [
        {"recording_group": group_name, **aggregate(items)}
        for group_name, items in sorted(grouped.items())
    ]
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_image_metrics.csv", records)
    _write_csv(output / "per_group_metrics.csv", per_group)
    report = {
        "experiment": output.name,
        "diagnostic": "component substitution and controlled oracle-angle perturbation",
        "baseline_checkpoint": str(baseline_path),
        "baseline_checkpoint_step": int(baseline_checkpoint["step"]),
        "predicted_checkpoint": str(predicted_path),
        "predicted_checkpoint_step": int(predicted_checkpoint["step"]),
        "validation_recording_groups": list(validation_groups),
        "angle_offsets_degrees": list(angle_offsets),
        **overall,
        "per_group": per_group,
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "official_test_accessed": False,
        "paper_result": False,
        "note": (
            "Sharp images are used to construct controlled oracle geometry. This is a "
            "development diagnostic, not a deployable or benchmark result."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
