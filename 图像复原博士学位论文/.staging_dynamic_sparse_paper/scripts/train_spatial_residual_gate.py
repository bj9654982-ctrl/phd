"""Train a dense residual-strength gate while keeping the B0 model frozen."""

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
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GoProPairDataset, recording_group  # noqa: E402
from src.losses import restoration_loss  # noqa: E402
from src.models import (  # noqa: E402
    CompactDeblurUNet,
    ImageResidualGate,
    SpatialResidualCorrection,
    SpatialResidualGate,
    apply_anchored_residual_correction,
    apply_residual_gate,
    mse_optimal_residual_alpha_map,
)


DEFAULT_BASELINE_RUN = PROJECT_ROOT / "results" / "runs" / "P3_baseline_group_split_v0"
DEFAULT_SCALAR_RUN = PROJECT_ROOT / "results" / "runs" / "P4_scalar_residual_gate_v1_patch256"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a spatial residual gate and compare it with the frozen scalar-gate v1."
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "GOPRO_Large")
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_RUN / "best.pt")
    parser.add_argument("--scalar-checkpoint", type=Path, default=DEFAULT_SCALAR_RUN / "best_gate.pt")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "P5_spatial_residual_gate_v0",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-pairs", type=int, default=0, help="0 uses all allowed training pairs.")
    parser.add_argument("--val-pairs", type=int, default=0, help="0 uses the complete frozen holdout.")
    parser.add_argument(
        "--gate-train-patch-size",
        type=int,
        help="Training crop; defaults to the B0 validation crop to match deployment scale.",
    )
    parser.add_argument("--gate-width", type=int, default=16)
    parser.add_argument("--initial-alpha", type=float, default=0.95)
    parser.add_argument(
        "--spatial-mode",
        choices=("independent", "anchored"),
        default="independent",
        help="anchored learns a bounded zero-mean correction around the frozen scalar gate.",
    )
    parser.add_argument("--max-correction", type=float, default=0.15)
    parser.add_argument("--oracle-window", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--frequency-weight", type=float, default=0.1)
    parser.add_argument("--oracle-weight", type=float, default=1.0)
    parser.add_argument("--restoration-weight", type=float, default=0.1)
    parser.add_argument("--smoothness-weight", type=float, default=0.01)
    parser.add_argument("--val-interval", type=int, default=250)
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


def _total_variation(alpha: torch.Tensor) -> torch.Tensor:
    horizontal = (alpha[:, :, :, 1:] - alpha[:, :, :, :-1]).abs().mean()
    vertical = (alpha[:, :, 1:, :] - alpha[:, :, :-1, :]).abs().mean()
    return horizontal + vertical


def _residual_energy_weights(residual: torch.Tensor, window_size: int) -> torch.Tensor:
    """Give reliable, high-residual neighborhoods more oracle-supervision weight."""
    energy = residual.float().square().sum(dim=1, keepdim=True)
    energy = F.avg_pool2d(
        energy,
        kernel_size=window_size,
        stride=1,
        padding=window_size // 2,
        count_include_pad=False,
    )
    relative_energy = energy / energy.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return relative_energy.clamp(0.1, 4.0)


def _compose_spatial_alpha(
    spatial_gate: SpatialResidualGate | SpatialResidualCorrection,
    spatial_mode: str,
    blurry: torch.Tensor,
    residual: torch.Tensor,
    scalar_alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if spatial_mode == "anchored":
        correction = spatial_gate(blurry, residual)
        alpha = apply_anchored_residual_correction(scalar_alpha, correction)
        return alpha, correction
    if spatial_mode == "independent":
        alpha = spatial_gate(blurry, residual)
        correction = alpha - scalar_alpha.expand_as(alpha)
        return alpha, correction
    raise ValueError(f"unsupported spatial mode: {spatial_mode}")


@torch.no_grad()
def evaluate(
    baseline: CompactDeblurUNet,
    scalar_gate: ImageResidualGate,
    spatial_gate: SpatialResidualGate | SpatialResidualCorrection,
    loader: DataLoader,
    device: torch.device,
    oracle_window: int,
    spatial_mode: str,
    detailed: bool = False,
) -> tuple[dict[str, float | int], list[dict[str, float | str]], list[dict[str, float | int | str]]]:
    baseline.eval()
    scalar_gate.eval()
    spatial_gate.eval()
    records: list[dict[str, float | str]] = []
    for batch in loader:
        blurry = batch["blur"].to(device, non_blocking=True)
        sharp = batch["sharp"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            baseline_restored = baseline(blurry)
        residual = baseline_restored.float() - blurry.float()
        scalar_alpha = scalar_gate(blurry.float(), residual)
        spatial_alpha, spatial_correction = _compose_spatial_alpha(
            spatial_gate,
            spatial_mode,
            blurry.float(),
            residual,
            scalar_alpha,
        )
        scalar_restored = apply_residual_gate(blurry.float(), baseline_restored.float(), scalar_alpha)
        spatial_restored = apply_residual_gate(blurry.float(), baseline_restored.float(), spatial_alpha)
        oracle_alpha = mse_optimal_residual_alpha_map(
            blurry,
            baseline_restored,
            sharp,
            window_size=oracle_window,
        )
        name = str(batch["name"][0])
        scene = name.split("/", maxsplit=1)[0]
        baseline_psnr = _psnr_value(baseline_restored, sharp)
        scalar_psnr = _psnr_value(scalar_restored, sharp)
        spatial_psnr = _psnr_value(spatial_restored, sharp)
        record: dict[str, float | str] = {
            "recording_group": recording_group(scene),
            "scene": scene,
            "name": name,
            "input_psnr_db": _psnr_value(blurry, sharp),
            "baseline_psnr_db": baseline_psnr,
            "scalar_psnr_db": scalar_psnr,
            "spatial_psnr_db": spatial_psnr,
            "scalar_minus_baseline_db": scalar_psnr - baseline_psnr,
            "spatial_minus_baseline_db": spatial_psnr - baseline_psnr,
            "spatial_minus_scalar_db": spatial_psnr - scalar_psnr,
            "scalar_alpha": float(scalar_alpha),
            "spatial_alpha_mean": float(spatial_alpha.mean()),
            "spatial_alpha_std": float(spatial_alpha.std(unbiased=False)),
            "spatial_alpha_min": float(spatial_alpha.min()),
            "spatial_alpha_max": float(spatial_alpha.max()),
            "spatial_correction_mean": float(spatial_correction.mean()),
            "spatial_correction_std": float(spatial_correction.std(unbiased=False)),
            "spatial_correction_abs_max": float(spatial_correction.abs().max()),
            "oracle_alpha_mean": float(oracle_alpha.mean()),
            "oracle_alpha_std": float(oracle_alpha.std(unbiased=False)),
            "oracle_alpha_mae": float((spatial_alpha - oracle_alpha).abs().mean()),
        }
        if detailed:
            baseline_ssim = _ssim_value(baseline_restored[0], sharp[0])
            scalar_ssim = _ssim_value(scalar_restored[0], sharp[0])
            spatial_ssim = _ssim_value(spatial_restored[0], sharp[0])
            record.update(
                {
                    "input_ssim": _ssim_value(blurry[0], sharp[0]),
                    "baseline_ssim": baseline_ssim,
                    "scalar_ssim": scalar_ssim,
                    "spatial_ssim": spatial_ssim,
                    "scalar_minus_baseline_ssim": scalar_ssim - baseline_ssim,
                    "spatial_minus_baseline_ssim": spatial_ssim - baseline_ssim,
                    "spatial_minus_scalar_ssim": spatial_ssim - scalar_ssim,
                }
            )
        records.append(record)

    numeric_fields = [
        "input_psnr_db",
        "baseline_psnr_db",
        "scalar_psnr_db",
        "spatial_psnr_db",
        "scalar_minus_baseline_db",
        "spatial_minus_baseline_db",
        "spatial_minus_scalar_db",
        "scalar_alpha",
        "spatial_alpha_mean",
        "spatial_alpha_std",
        "spatial_alpha_min",
        "spatial_alpha_max",
        "spatial_correction_mean",
        "spatial_correction_std",
        "spatial_correction_abs_max",
        "oracle_alpha_mean",
        "oracle_alpha_std",
        "oracle_alpha_mae",
    ]
    if detailed:
        numeric_fields.extend(
            [
                "input_ssim",
                "baseline_ssim",
                "scalar_ssim",
                "spatial_ssim",
                "scalar_minus_baseline_ssim",
                "spatial_minus_baseline_ssim",
                "spatial_minus_scalar_ssim",
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
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for record in records:
        grouped[str(record["recording_group"])].append(record)
    group_summaries: list[dict[str, float | int | str]] = [
        {"recording_group": group_name, **aggregate(items)}
        for group_name, items in sorted(grouped.items())
    ]
    return overall, records, group_summaries


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script has no CPU fallback")
    if args.steps <= 0 or args.batch_size <= 0 or args.val_interval <= 0:
        raise ValueError("steps, batch-size, and val-interval must be positive")
    if args.train_pairs < 0 or args.val_pairs < 0 or args.num_workers < 0:
        raise ValueError("train-pairs, val-pairs, and num-workers must be nonnegative")
    if args.gate_train_patch_size is not None and args.gate_train_patch_size <= 0:
        raise ValueError("gate-train-patch-size must be positive")
    if args.oracle_window <= 0 or args.oracle_window % 2 == 0:
        raise ValueError("oracle-window must be a positive odd integer")
    if not 0 < args.max_correction <= 0.5:
        raise ValueError("max-correction must be in (0, 0.5]")
    if min(args.oracle_weight, args.restoration_weight, args.smoothness_weight) < 0:
        raise ValueError("loss weights must be nonnegative")
    if args.oracle_weight == 0 and args.restoration_weight == 0:
        raise ValueError("oracle-weight and restoration-weight cannot both be zero")

    set_determinism(args.seed)
    device = torch.device("cuda:0")
    baseline_path = args.baseline_checkpoint.expanduser().resolve()
    scalar_path = args.scalar_checkpoint.expanduser().resolve()
    if not baseline_path.is_file():
        raise FileNotFoundError(f"baseline checkpoint not found: {baseline_path}")
    if not scalar_path.is_file():
        raise FileNotFoundError(f"scalar-gate checkpoint not found: {scalar_path}")

    baseline_checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=False)
    baseline_args = baseline_checkpoint.get("args", {})
    if baseline_args.get("variant") != "local":
        raise ValueError("The baseline checkpoint must use variant=local")
    if baseline_args.get("validation_source") != "train_holdout":
        raise ValueError("The baseline checkpoint must use validation_source=train_holdout")
    validation_groups = tuple(baseline_args.get("validation_groups") or ())
    if not validation_groups:
        raise ValueError("The baseline checkpoint does not contain frozen validation groups")

    scalar_checkpoint = torch.load(scalar_path, map_location="cpu", weights_only=False)
    scalar_args = scalar_checkpoint.get("args", {})
    if Path(scalar_checkpoint["baseline_checkpoint"]).resolve() != baseline_path:
        raise ValueError("The scalar reference and spatial gate must use the same B0 checkpoint")

    gate_train_patch_size = args.gate_train_patch_size or int(baseline_args["val_patch_size"])
    train_set = GoProPairDataset(
        args.data_root,
        split="train",
        patch_size=gate_train_patch_size,
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

    generator = torch.Generator().manual_seed(args.seed)
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
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **loader_options)

    baseline = CompactDeblurUNet(width=int(baseline_args["width"]))
    baseline.load_state_dict(baseline_checkpoint["model"], strict=True)
    baseline = baseline.to(device).eval()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)

    scalar_gate = ImageResidualGate(
        width=int(scalar_args["gate_width"]),
        initial_alpha=float(scalar_args["initial_alpha"]),
    )
    scalar_gate.load_state_dict(scalar_checkpoint["gate"], strict=True)
    scalar_gate = scalar_gate.to(device).eval()
    for parameter in scalar_gate.parameters():
        parameter.requires_grad_(False)

    if args.spatial_mode == "anchored":
        spatial_gate = SpatialResidualCorrection(
            width=args.gate_width,
            max_correction=args.max_correction,
        ).to(device)
    else:
        spatial_gate = SpatialResidualGate(
            width=args.gate_width,
            initial_alpha=args.initial_alpha,
        ).to(device)
    gate_parameters = sum(
        parameter.numel() for parameter in spatial_gate.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(spatial_gate.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    initial, _, initial_groups = evaluate(
        baseline,
        scalar_gate,
        spatial_gate,
        val_loader,
        device,
        args.oracle_window,
        args.spatial_mode,
    )
    best_psnr = float(initial["spatial_psnr_db"])
    best_step = 0
    checkpoint_metadata = {
        "args": vars(args),
        "baseline_checkpoint": str(baseline_path),
        "scalar_checkpoint": str(scalar_path),
    }
    torch.save(
        {"step": 0, "gate": spatial_gate.state_dict(), **checkpoint_metadata},
        output / "best_gate.pt",
    )
    history.append(
        {
            "step": 0,
            "train_loss": float("nan"),
            "train_oracle_loss": float("nan"),
            "train_restoration_loss": float("nan"),
            "train_smoothness_loss": float("nan"),
            "baseline_psnr_db": float(initial["baseline_psnr_db"]),
            "scalar_psnr_db": float(initial["scalar_psnr_db"]),
            "spatial_psnr_db": float(initial["spatial_psnr_db"]),
            "spatial_minus_baseline_db": float(initial["spatial_minus_baseline_db"]),
            "spatial_minus_scalar_db": float(initial["spatial_minus_scalar_db"]),
            "spatial_alpha_mean": float(initial["spatial_alpha_mean"]),
            "spatial_alpha_std": float(initial["spatial_alpha_std"]),
            "minimum_group_delta_db": min(
                float(group["spatial_minus_baseline_db"]) for group in initial_groups
            ),
            "learning_rate": args.learning_rate,
        }
    )

    train_iterator = iter(train_loader)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    progress = tqdm(range(1, args.steps + 1), desc="Frozen B0 spatial gate")
    for step in progress:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        blurry = batch["blur"].to(device, non_blocking=True)
        sharp = batch["sharp"].to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            baseline_restored = baseline(blurry)
        residual = baseline_restored.float() - blurry.float()

        spatial_gate.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            scalar_alpha = scalar_gate(blurry.float(), residual)
        alpha, spatial_correction = _compose_spatial_alpha(
            spatial_gate,
            args.spatial_mode,
            blurry.float(),
            residual,
            scalar_alpha,
        )
        gated_restored = apply_residual_gate(blurry.float(), baseline_restored.float(), alpha)
        oracle_alpha = mse_optimal_residual_alpha_map(
            blurry,
            baseline_restored,
            sharp,
            window_size=args.oracle_window,
        )
        confidence = _residual_energy_weights(residual, args.oracle_window)
        oracle_error = F.smooth_l1_loss(alpha, oracle_alpha, reduction="none")
        oracle_loss = (oracle_error * confidence).sum() / confidence.sum().clamp_min(1e-12)
        restore_loss = restoration_loss(
            gated_restored,
            sharp.float(),
            frequency_weight=args.frequency_weight,
        )
        smoothness_loss = _total_variation(alpha)
        loss = (
            args.oracle_weight * oracle_loss
            + args.restoration_weight * restore_loss
            + args.smoothness_weight * smoothness_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(spatial_gate.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        loss_value = float(loss.detach())
        if step == 1 or step % 20 == 0:
            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                alpha=f"{float(alpha.mean()):.3f}",
                target=f"{float(oracle_alpha.mean()):.3f}",
            )

        if step % args.val_interval == 0 or step == args.steps:
            validation, _, groups = evaluate(
                baseline,
                scalar_gate,
                spatial_gate,
                val_loader,
                device,
                args.oracle_window,
                args.spatial_mode,
            )
            group_minimum = min(float(group["spatial_minus_baseline_db"]) for group in groups)
            history.append(
                {
                    "step": step,
                    "train_loss": loss_value,
                    "train_oracle_loss": float(oracle_loss.detach()),
                    "train_restoration_loss": float(restore_loss.detach()),
                    "train_smoothness_loss": float(smoothness_loss.detach()),
                    "baseline_psnr_db": float(validation["baseline_psnr_db"]),
                    "scalar_psnr_db": float(validation["scalar_psnr_db"]),
                    "spatial_psnr_db": float(validation["spatial_psnr_db"]),
                    "spatial_minus_baseline_db": float(validation["spatial_minus_baseline_db"]),
                    "spatial_minus_scalar_db": float(validation["spatial_minus_scalar_db"]),
                    "spatial_alpha_mean": float(validation["spatial_alpha_mean"]),
                    "spatial_alpha_std": float(validation["spatial_alpha_std"]),
                    "minimum_group_delta_db": group_minimum,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            print(
                f"\nstep={step:05d} B0={validation['baseline_psnr_db']:.4f} dB "
                f"scalar={validation['scalar_psnr_db']:.4f} dB "
                f"spatial={validation['spatial_psnr_db']:.4f} dB "
                f"vs_scalar={validation['spatial_minus_scalar_db']:+.4f} dB "
                f"alpha={validation['spatial_alpha_mean']:.3f} "
                f"map_std={validation['spatial_alpha_std']:.3f}"
            )
            if float(validation["spatial_psnr_db"]) > best_psnr:
                best_psnr = float(validation["spatial_psnr_db"])
                best_step = step
                torch.save(
                    {"step": step, "gate": spatial_gate.state_dict(), **checkpoint_metadata},
                    output / "best_gate.pt",
                )

    torch.save(
        {"step": args.steps, "gate": spatial_gate.state_dict(), **checkpoint_metadata},
        output / "last_gate.pt",
    )
    _write_csv(output / "metrics.csv", history)

    best_gate_checkpoint = torch.load(output / "best_gate.pt", map_location=device, weights_only=False)
    spatial_gate.load_state_dict(best_gate_checkpoint["gate"], strict=True)
    final, per_image, per_group = evaluate(
        baseline,
        scalar_gate,
        spatial_gate,
        val_loader,
        device,
        args.oracle_window,
        args.spatial_mode,
        detailed=True,
    )
    _write_csv(output / "per_image_metrics.csv", per_image)
    _write_csv(output / "per_group_metrics.csv", per_group)

    minimum_group_delta = min(
        float(group["spatial_minus_baseline_db"]) for group in per_group
    )
    psnr_passed = float(final["spatial_minus_scalar_db"]) >= 0.02
    ssim_passed = float(final["spatial_minus_scalar_ssim"]) >= 0.0
    group_passed = minimum_group_delta >= -0.01
    report = {
        "experiment": output.name,
        "baseline_checkpoint": str(baseline_path),
        "baseline_checkpoint_step": int(baseline_checkpoint["step"]),
        "scalar_checkpoint": str(scalar_path),
        "scalar_checkpoint_step": int(scalar_checkpoint["step"]),
        "spatial_checkpoint_step": int(best_gate_checkpoint["step"]),
        "seed": args.seed,
        "steps": args.steps,
        "train_pairs": len(train_set),
        "val_pairs": len(val_set),
        "gate_train_patch_size": gate_train_patch_size,
        "validation_patch_size": int(baseline_args["val_patch_size"]),
        "validation_recording_groups": list(validation_groups),
        "oracle_window": args.oracle_window,
        "spatial_gate_receptive_field": 33,
        "gate_parameters": gate_parameters,
        "spatial_mode": args.spatial_mode,
        "initial_alpha": args.initial_alpha,
        "max_correction": args.max_correction,
        "oracle_weight": args.oracle_weight,
        "restoration_weight": args.restoration_weight,
        "smoothness_weight": args.smoothness_weight,
        **final,
        "per_group": per_group,
        "minimum_group_spatial_minus_baseline_db": minimum_group_delta,
        "images_spatial_better_than_scalar": sum(
            float(record["spatial_minus_scalar_db"]) > 0 for record in per_image
        ),
        "images_spatial_better_than_baseline": sum(
            float(record["spatial_minus_baseline_db"]) > 0 for record in per_image
        ),
        "psnr_gain_over_scalar_threshold_db": 0.02,
        "minimum_group_gain_over_baseline_threshold_db": -0.01,
        "psnr_gate_passed": psnr_passed,
        "ssim_gate_passed": ssim_passed,
        "group_gate_passed": group_passed,
        "passed": psnr_passed and ssim_passed and group_passed,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "wall_seconds": time.perf_counter() - started,
        "official_test_accessed": False,
        "paper_result": False,
        "note": (
            f"Frozen-B0 spatial-gate screening experiment in {args.spatial_mode} mode. "
            "Passing only supports spatial adaptation and does not establish the final "
            "trajectory-nonlocal method."
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
