"""P6-B: replace paired-oracle geometry with a blurry-only predictor."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GoProPairDataset, recording_group  # noqa: E402
from src.losses import restoration_loss  # noqa: E402
from src.models import (  # noqa: E402
    CompactDeblurUNet,
    TrajectoryNonlocalBlock,
    dense_paired_geometry_supervision_loss,
    paired_motion_geometry_pseudo_label,
)


DEFAULT_BASELINE_RUN = PROJECT_ROOT / "results" / "runs" / "P3_baseline_group_split_v0"
DEFAULT_ORACLE_RUN = PROJECT_ROOT / "results" / "runs" / "P6A_paired_oracle_trajectory_v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train blurry-only geometry heads while freezing the P6-A trajectory restorer."
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "GOPRO_Large")
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=DEFAULT_BASELINE_RUN / "best.pt",
    )
    parser.add_argument(
        "--oracle-checkpoint",
        type=Path,
        default=DEFAULT_ORACLE_RUN / "best_aligned.pt",
    )
    parser.add_argument("--initialization-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "P6B_predicted_trajectory_geometry_v0",
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-pairs", type=int, default=0)
    parser.add_argument("--val-pairs", type=int, default=0)
    parser.add_argument("--train-patch-size", type=int, default=256)
    parser.add_argument("--oracle-window", type=int, default=31)
    parser.add_argument("--direction-residual-scale", type=float, default=0.25)
    parser.add_argument(
        "--training-mode",
        choices=("geometry_heads", "confidence_calibration", "aggregation_adaptation"),
        default="geometry_heads",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--geometry-weight", type=float, default=0.05)
    parser.add_argument("--frequency-weight", type=float, default=0.1)
    parser.add_argument("--val-interval", type=int, default=375)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _psnr_value(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    mse = (prediction - target).square().mean().clamp_min(1e-12)
    return float(-10.0 * mse.log10())


def _ssim_value(prediction: torch.Tensor, target: torch.Tensor) -> float:
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


def _write_csv(path: Path, records: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


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


def _make_model(
    oracle_state: dict[str, torch.Tensor],
    width: int,
    oracle_args: dict[str, object],
    direction_residual_scale: float,
    training_mode: str,
) -> CompactDeblurUNet:
    block = TrajectoryNonlocalBlock(
        channels=width * 4,
        samples=int(oracle_args["trajectory_samples"]),
        max_feature_length=float(oracle_args["max_feature_length"]),
        fusion_mode=str(oracle_args["trajectory_fusion"]),
        angle_hypotheses=1,
        direction_residual_scale=direction_residual_scale,
    )
    model = CompactDeblurUNet(width=width, bottleneck_module=block)
    model.load_state_dict(oracle_state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if training_mode == "geometry_heads":
        trainable_components = (
            block.geometry_features,
            block.statistics_projection,
            block.direction_residual,
            block.length_head,
            block.confidence_head,
        )
    elif training_mode == "confidence_calibration":
        trainable_components = (block.confidence_head,)
    elif training_mode == "aggregation_adaptation":
        trainable_components = (
            block.normalization,
            block.query,
            block.key_value,
            block.local,
            block.output,
        )
    else:
        raise ValueError(f"unsupported training mode: {training_mode}")
    for component in trainable_components:
        for parameter in component.parameters():
            parameter.requires_grad_(True)
    if training_mode == "aggregation_adaptation":
        block.relative_bias.requires_grad_(True)
        block.residual_scale.requires_grad_(True)
    return model


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _weighted_geometry_metrics(
    predicted: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> tuple[float, float, float]:
    target_direction = torch.nn.functional.normalize(target["direction"].float(), dim=1, eps=1e-8)
    target_double_angle = torch.cat(
        (
            target_direction[:, :1].square() - target_direction[:, 1:2].square(),
            2.0 * target_direction[:, :1] * target_direction[:, 1:2],
        ),
        dim=1,
    )
    reliability = target["confidence"].float().clamp(0.0, 1.0)
    normalizer = reliability.sum().clamp_min(1e-8)
    cosine = (
        torch.nn.functional.normalize(predicted["double_angle"].float(), dim=1, eps=1e-8)
        * target_double_angle
    ).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    direction_error = (0.5 * torch.rad2deg(torch.acos(cosine)) * reliability).sum() / normalizer
    length_error = ((predicted["length"].float() - target["length"]).abs() * reliability).sum() / normalizer
    confidence_error = (predicted["confidence"].float() - target["confidence"]).abs().mean()
    return float(direction_error), float(length_error), float(confidence_error)


@torch.no_grad()
def evaluate(
    baseline: CompactDeblurUNet,
    model: CompactDeblurUNet,
    loader: DataLoader,
    oracle_window: int,
    max_feature_length: float,
    detailed: bool = False,
) -> tuple[dict[str, float | int], list[dict[str, float | str]], list[dict[str, float | int | str]]]:
    baseline.eval()
    model.eval()
    records: list[dict[str, float | str]] = []
    for batch in loader:
        blurry = batch["blur"].cuda(non_blocking=True)
        sharp = batch["sharp"].cuda(non_blocking=True)
        oracle_geometry = _paired_geometry(
            blurry,
            sharp,
            oracle_window,
            max_feature_length,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            baseline_output = baseline(blurry)
            predicted_output, predicted_geometry = model(blurry, return_geometry=True)
            oracle_output = model(blurry, geometry_override=oracle_geometry)
        direction_error, length_error, confidence_error = _weighted_geometry_metrics(
            predicted_geometry,
            oracle_geometry,
        )
        name = str(batch["name"][0])
        scene = name.split("/", maxsplit=1)[0]
        baseline_psnr = _psnr_value(baseline_output, sharp)
        predicted_psnr = _psnr_value(predicted_output, sharp)
        oracle_psnr = _psnr_value(oracle_output, sharp)
        record: dict[str, float | str] = {
            "recording_group": recording_group(scene),
            "scene": scene,
            "name": name,
            "input_psnr_db": _psnr_value(blurry, sharp),
            "baseline_psnr_db": baseline_psnr,
            "predicted_psnr_db": predicted_psnr,
            "oracle_psnr_db": oracle_psnr,
            "predicted_minus_baseline_db": predicted_psnr - baseline_psnr,
            "oracle_minus_baseline_db": oracle_psnr - baseline_psnr,
            "predicted_minus_oracle_db": predicted_psnr - oracle_psnr,
            "direction_error_degrees": direction_error,
            "length_error_feature_pixels": length_error,
            "confidence_mae": confidence_error,
            "predicted_length_mean": float(predicted_geometry["length"].mean()),
            "predicted_confidence_mean": float(predicted_geometry["confidence"].mean()),
            "oracle_length_mean": float(oracle_geometry["length"].mean()),
            "oracle_confidence_mean": float(oracle_geometry["confidence"].mean()),
        }
        if detailed:
            baseline_ssim = _ssim_value(baseline_output[0], sharp[0])
            predicted_ssim = _ssim_value(predicted_output[0], sharp[0])
            oracle_ssim = _ssim_value(oracle_output[0], sharp[0])
            record.update(
                {
                    "input_ssim": _ssim_value(blurry[0], sharp[0]),
                    "baseline_ssim": baseline_ssim,
                    "predicted_ssim": predicted_ssim,
                    "oracle_ssim": oracle_ssim,
                    "predicted_minus_baseline_ssim": predicted_ssim - baseline_ssim,
                    "oracle_minus_baseline_ssim": oracle_ssim - baseline_ssim,
                    "predicted_minus_oracle_ssim": predicted_ssim - oracle_ssim,
                }
            )
        records.append(record)

    numeric_fields = [
        "input_psnr_db",
        "baseline_psnr_db",
        "predicted_psnr_db",
        "oracle_psnr_db",
        "predicted_minus_baseline_db",
        "oracle_minus_baseline_db",
        "predicted_minus_oracle_db",
        "direction_error_degrees",
        "length_error_feature_pixels",
        "confidence_mae",
        "predicted_length_mean",
        "predicted_confidence_mean",
        "oracle_length_mean",
        "oracle_confidence_mean",
    ]
    if detailed:
        numeric_fields.extend(
            [
                "input_ssim",
                "baseline_ssim",
                "predicted_ssim",
                "oracle_ssim",
                "predicted_minus_baseline_ssim",
                "oracle_minus_baseline_ssim",
                "predicted_minus_oracle_ssim",
            ]
        )

    def aggregate(items: list[dict[str, float | str]]) -> dict[str, float | int]:
        return {
            "pairs": len(items),
            **{
                field: statistics.fmean(float(item[field]) for item in items)
                for field in numeric_fields
            },
        }

    overall = aggregate(records)
    oracle_gain = float(overall["oracle_minus_baseline_db"])
    overall["oracle_gain_retention"] = (
        float(overall["predicted_minus_baseline_db"]) / oracle_gain
        if oracle_gain > 1e-8
        else float("nan")
    )
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for record in records:
        grouped[str(record["recording_group"])].append(record)
    per_group: list[dict[str, float | int | str]] = []
    for group_name, items in sorted(grouped.items()):
        group = {"recording_group": group_name, **aggregate(items)}
        group_oracle_gain = float(group["oracle_minus_baseline_db"])
        group["oracle_gain_retention"] = (
            float(group["predicted_minus_baseline_db"]) / group_oracle_gain
            if group_oracle_gain > 1e-8
            else float("nan")
        )
        per_group.append(group)
    return overall, records, per_group


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script has no CPU fallback")
    if min(args.steps, args.batch_size, args.train_patch_size, args.val_interval) <= 0:
        raise ValueError("steps, batch-size, train-patch-size, and val-interval must be positive")
    if args.train_pairs < 0 or args.val_pairs < 0 or args.num_workers < 0:
        raise ValueError("train-pairs, val-pairs, and num-workers must be nonnegative")
    if args.oracle_window < 3 or args.oracle_window % 2 == 0:
        raise ValueError("oracle-window must be an odd integer >= 3")
    if args.geometry_weight < 0:
        raise ValueError("geometry-weight must be nonnegative")
    if args.direction_residual_scale <= 0:
        raise ValueError("direction-residual-scale must be positive")

    set_determinism(args.seed)
    device = torch.device("cuda:0")
    baseline_path = args.baseline_checkpoint.expanduser().resolve()
    oracle_path = args.oracle_checkpoint.expanduser().resolve()
    initialization_path = (
        args.initialization_checkpoint.expanduser().resolve()
        if args.initialization_checkpoint is not None
        else oracle_path
    )
    if not baseline_path.is_file():
        raise FileNotFoundError(f"baseline checkpoint not found: {baseline_path}")
    if not oracle_path.is_file():
        raise FileNotFoundError(f"P6-A oracle checkpoint not found: {oracle_path}")
    if not initialization_path.is_file():
        raise FileNotFoundError(f"initialization checkpoint not found: {initialization_path}")
    baseline_checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=False)
    oracle_checkpoint = torch.load(oracle_path, map_location="cpu", weights_only=False)
    initialization_checkpoint = (
        oracle_checkpoint
        if initialization_path == oracle_path
        else torch.load(initialization_path, map_location="cpu", weights_only=False)
    )
    baseline_args = baseline_checkpoint.get("args", {})
    oracle_args = oracle_checkpoint.get("args", {})
    if baseline_args.get("variant") != "local":
        raise ValueError("The baseline checkpoint must use variant=local")
    if baseline_args.get("validation_source") != "train_holdout":
        raise ValueError("The baseline checkpoint must use validation_source=train_holdout")
    validation_groups = tuple(baseline_args.get("validation_groups") or ())
    if not validation_groups:
        raise ValueError("The baseline checkpoint does not contain frozen validation groups")
    required_oracle_args = ("trajectory_samples", "max_feature_length", "trajectory_fusion")
    if any(name not in oracle_args for name in required_oracle_args):
        raise ValueError("The P6-A checkpoint is missing trajectory configuration")

    train_set = GoProPairDataset(
        args.data_root,
        split="train",
        patch_size=args.train_patch_size,
        augment=True,
        max_pairs=args.train_pairs or None,
        exclude_recording_groups=validation_groups,
    )
    val_set = GoProPairDataset(
        args.data_root,
        split="train",
        patch_size=int(baseline_args["val_patch_size"]),
        augment=False,
        max_pairs=args.val_pairs or None,
        include_recording_groups=validation_groups,
    )
    if set(train_set.recording_groups) & set(val_set.recording_groups):
        raise RuntimeError("Training/validation recording-group leakage detected")
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_options,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **loader_options)

    width = int(baseline_args["width"])
    baseline = CompactDeblurUNet(width=width)
    baseline.load_state_dict(baseline_checkpoint["model"], strict=True)
    baseline = baseline.to(device).eval()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)
    model = _make_model(
        initialization_checkpoint["model"],
        width,
        oracle_args,
        args.direction_residual_scale,
        args.training_mode,
    ).to(device)
    trainable_parameters = _trainable_parameters(model)
    if not trainable_parameters:
        raise RuntimeError("No geometry-prediction parameters were enabled")
    trainable_parameter_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if args.training_mode == "geometry_heads":
        allowed_geometry_components = (
            "bottleneck_module.geometry_features.",
            "bottleneck_module.statistics_projection.",
            "bottleneck_module.direction_residual.",
            "bottleneck_module.length_head.",
            "bottleneck_module.confidence_head.",
        )
    elif args.training_mode == "confidence_calibration":
        allowed_geometry_components = ("bottleneck_module.confidence_head.",)
    else:
        allowed_geometry_components = (
            "bottleneck_module.normalization.",
            "bottleneck_module.query.",
            "bottleneck_module.key_value.",
            "bottleneck_module.local.",
            "bottleneck_module.output.",
            "bottleneck_module.relative_bias",
            "bottleneck_module.residual_scale",
        )
    if any(not name.startswith(allowed_geometry_components) for name in trainable_parameter_names):
        raise RuntimeError("A non-geometry parameter was accidentally enabled for P6-B")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda")
    max_feature_length = float(oracle_args["max_feature_length"])

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    initial, _, initial_groups = evaluate(
        baseline,
        model,
        val_loader,
        args.oracle_window,
        max_feature_length,
    )
    metadata = {
        "args": vars(args),
        "baseline_checkpoint": str(baseline_path),
        "oracle_checkpoint": str(oracle_path),
        "initialization_checkpoint": str(initialization_path),
        "oracle_args": oracle_args,
    }
    torch.save({"step": 0, "model": model.state_dict(), **metadata}, output / "best.pt")
    best_psnr = float(initial["predicted_psnr_db"])
    history: list[dict[str, float | int]] = [
        {
            "step": 0,
            "total_train_loss": float("nan"),
            "restoration_train_loss": float("nan"),
            "geometry_train_loss": float("nan"),
            "direction_train_loss": float("nan"),
            "length_train_loss": float("nan"),
            "confidence_train_loss": float("nan"),
            "baseline_psnr_db": float(initial["baseline_psnr_db"]),
            "predicted_psnr_db": float(initial["predicted_psnr_db"]),
            "oracle_psnr_db": float(initial["oracle_psnr_db"]),
            "predicted_minus_baseline_db": float(initial["predicted_minus_baseline_db"]),
            "oracle_gain_retention": float(initial["oracle_gain_retention"]),
            "direction_error_degrees": float(initial["direction_error_degrees"]),
            "length_error_feature_pixels": float(initial["length_error_feature_pixels"]),
            "minimum_group_predicted_minus_baseline_db": min(
                float(group["predicted_minus_baseline_db"]) for group in initial_groups
            ),
            "learning_rate": args.learning_rate,
        }
    ]

    train_iterator = iter(train_loader)
    last_losses = {
        "total": float("nan"),
        "restoration": float("nan"),
        "geometry": float("nan"),
        "direction": float("nan"),
        "length": float("nan"),
        "confidence": float("nan"),
    }
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    progress = tqdm(range(1, args.steps + 1), desc="P6-B blurry-only geometry")
    for step in progress:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        blurry = batch["blur"].to(device, non_blocking=True)
        sharp = batch["sharp"].to(device, non_blocking=True)
        target_geometry = _paired_geometry(
            blurry,
            sharp,
            args.oracle_window,
            max_feature_length,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction, predicted_geometry = model(blurry, return_geometry=True)
            restoration = restoration_loss(
                prediction,
                sharp,
                frequency_weight=args.frequency_weight,
            )
            geometry, geometry_components = dense_paired_geometry_supervision_loss(
                predicted_geometry["double_angle"],
                predicted_geometry["length"],
                predicted_geometry["confidence"],
                target_geometry["direction"],
                target_geometry["length"],
                target_geometry["confidence"],
                max_feature_length=max_feature_length,
            )
            total = restoration + args.geometry_weight * geometry
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        last_losses = {
            "total": float(total.detach()),
            "restoration": float(restoration.detach()),
            "geometry": float(geometry.detach()),
            **{name: float(value) for name, value in geometry_components.items()},
        }
        if step == 1 or step % 20 == 0:
            progress.set_postfix(
                restoration=f"{last_losses['restoration']:.4f}",
                geometry=f"{last_losses['geometry']:.4f}",
            )

        if step % args.val_interval == 0 or step == args.steps:
            validation, _, groups = evaluate(
                baseline,
                model,
                val_loader,
                args.oracle_window,
                max_feature_length,
            )
            history.append(
                {
                    "step": step,
                    "total_train_loss": last_losses["total"],
                    "restoration_train_loss": last_losses["restoration"],
                    "geometry_train_loss": last_losses["geometry"],
                    "direction_train_loss": last_losses["direction"],
                    "length_train_loss": last_losses["length"],
                    "confidence_train_loss": last_losses["confidence"],
                    "baseline_psnr_db": float(validation["baseline_psnr_db"]),
                    "predicted_psnr_db": float(validation["predicted_psnr_db"]),
                    "oracle_psnr_db": float(validation["oracle_psnr_db"]),
                    "predicted_minus_baseline_db": float(validation["predicted_minus_baseline_db"]),
                    "oracle_gain_retention": float(validation["oracle_gain_retention"]),
                    "direction_error_degrees": float(validation["direction_error_degrees"]),
                    "length_error_feature_pixels": float(validation["length_error_feature_pixels"]),
                    "minimum_group_predicted_minus_baseline_db": min(
                        float(group["predicted_minus_baseline_db"]) for group in groups
                    ),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            print(
                f"\nstep={step:05d} B0={validation['baseline_psnr_db']:.4f} dB "
                f"predicted={validation['predicted_psnr_db']:.4f} dB "
                f"oracle={validation['oracle_psnr_db']:.4f} dB "
                f"predicted_vs_B0={validation['predicted_minus_baseline_db']:+.4f} dB "
                f"retention={validation['oracle_gain_retention']:.1%} "
                f"direction_error={validation['direction_error_degrees']:.2f} deg"
            )
            if float(validation["predicted_psnr_db"]) > best_psnr:
                best_psnr = float(validation["predicted_psnr_db"])
                torch.save(
                    {"step": step, "model": model.state_dict(), **metadata},
                    output / "best.pt",
                )

    _write_csv(output / "metrics.csv", history)
    torch.save({"step": args.steps, "model": model.state_dict(), **metadata}, output / "last.pt")
    best_checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"], strict=True)
    final, per_image, per_group = evaluate(
        baseline,
        model,
        val_loader,
        args.oracle_window,
        max_feature_length,
        detailed=True,
    )
    _write_csv(output / "per_image_metrics.csv", per_image)
    _write_csv(output / "per_group_metrics.csv", per_group)

    minimum_group_delta = min(float(group["predicted_minus_baseline_db"]) for group in per_group)
    retention_passed = float(final["oracle_gain_retention"]) >= 0.60
    psnr_passed = float(final["predicted_minus_baseline_db"]) >= 0.075
    ssim_passed = float(final["predicted_minus_baseline_ssim"]) >= 0.0
    group_passed = minimum_group_delta >= -0.01
    report = {
        "experiment": output.name,
        "stage": (
            "restoration-driven confidence calibration with frozen direction, length, and aggregation"
            if args.training_mode == "confidence_calibration"
            else (
                "predicted-geometry aggregation adaptation with frozen geometry and U-Net"
                if args.training_mode == "aggregation_adaptation"
                else "blurry-only geometry predictor with frozen P6-A aggregation"
            )
        ),
        "baseline_checkpoint": str(baseline_path),
        "baseline_checkpoint_step": int(baseline_checkpoint["step"]),
        "oracle_checkpoint": str(oracle_path),
        "oracle_checkpoint_step": int(oracle_checkpoint["step"]),
        "initialization_checkpoint": str(initialization_path),
        "initialization_checkpoint_step": int(initialization_checkpoint["step"]),
        "training_mode": args.training_mode,
        "predicted_checkpoint_step": int(best_checkpoint["step"]),
        "seed": args.seed,
        "steps": args.steps,
        "train_pairs": len(train_set),
        "val_pairs": len(val_set),
        "train_patch_size": args.train_patch_size,
        "validation_patch_size": int(baseline_args["val_patch_size"]),
        "validation_recording_groups": list(validation_groups),
        "trajectory_samples": int(oracle_args["trajectory_samples"]),
        "max_feature_length": max_feature_length,
        "trajectory_fusion": str(oracle_args["trajectory_fusion"]),
        "oracle_window": args.oracle_window,
        "direction_residual_scale": args.direction_residual_scale,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_geometry_parameters": sum(parameter.numel() for parameter in trainable_parameters),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
        "trainable_parameter_names": trainable_parameter_names,
        **final,
        "per_group": per_group,
        "minimum_group_predicted_minus_baseline_db": minimum_group_delta,
        "images_predicted_better_than_baseline": sum(
            float(record["predicted_minus_baseline_db"]) > 0 for record in per_image
        ),
        "images_predicted_better_than_oracle": sum(
            float(record["predicted_minus_oracle_db"]) > 0 for record in per_image
        ),
        "oracle_gain_retention_threshold": 0.60,
        "predicted_over_baseline_threshold_db": 0.075,
        "minimum_group_threshold_db": -0.01,
        "retention_gate_passed": retention_passed,
        "psnr_gate_passed": psnr_passed,
        "ssim_gate_passed": ssim_passed,
        "group_gate_passed": group_passed,
        "passed": retention_passed and psnr_passed and ssim_passed and group_passed,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "wall_seconds": time.perf_counter() - started,
        "predicted_branch_uses_sharp_at_inference": False,
        "oracle_branch_uses_sharp": True,
        "official_test_accessed": False,
        "paper_result": False,
        "note": (
            "Development-holdout experiment. The predicted branch reads only blurry images; "
            "sharp images are used during training as a teacher and in the oracle diagnostic."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
