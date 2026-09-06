"""P6-A: train aligned and orthogonal trajectory blocks on a frozen real-data B0."""

from __future__ import annotations

import argparse
import copy
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
    paired_motion_geometry_pseudo_label,
)


DEFAULT_BASELINE_RUN = PROJECT_ROOT / "results" / "runs" / "P3_baseline_group_split_v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train paired-pseudo-oracle aligned and orthogonal trajectory controls."
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "GOPRO_Large")
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_RUN / "best.pt")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "P6A_paired_oracle_trajectory_v0",
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-pairs", type=int, default=0)
    parser.add_argument("--val-pairs", type=int, default=0)
    parser.add_argument("--train-patch-size", type=int, default=256)
    parser.add_argument("--trajectory-samples", type=int, default=5)
    parser.add_argument("--max-feature-length", type=float, default=7.0)
    parser.add_argument("--trajectory-fusion", choices=("aggregate", "detail"), default="detail")
    parser.add_argument("--oracle-window", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
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
def _paired_geometries(
    blurry: torch.Tensor,
    sharp: torch.Tensor,
    window: int,
    max_feature_length: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    direction, length, confidence = paired_motion_geometry_pseudo_label(
        blurry,
        sharp,
        output_size=_feature_size(blurry),
        window=window,
        max_feature_length=max_feature_length,
    )
    orthogonal_direction = torch.cat((-direction[:, 1:2], direction[:, :1]), dim=1)
    aligned = {"direction": direction, "length": length, "confidence": confidence}
    orthogonal = {
        "direction": orthogonal_direction,
        "length": length,
        "confidence": confidence,
    }
    return aligned, orthogonal


def _make_augmented_model(
    baseline_state: dict[str, torch.Tensor],
    width: int,
    block: TrajectoryNonlocalBlock,
) -> CompactDeblurUNet:
    model = CompactDeblurUNet(width=width)
    model.load_state_dict(baseline_state, strict=True)
    model.bottleneck_module = block
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_components = (
        block.normalization,
        block.query,
        block.key_value,
        block.local,
        block.output,
    )
    for component in trainable_components:
        for parameter in component.parameters():
            parameter.requires_grad_(True)
    block.relative_bias.requires_grad_(True)
    block.residual_scale.requires_grad_(True)
    return model


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


@torch.no_grad()
def evaluate(
    baseline: CompactDeblurUNet,
    aligned_model: CompactDeblurUNet,
    orthogonal_model: CompactDeblurUNet,
    loader: DataLoader,
    device: torch.device,
    oracle_window: int,
    max_feature_length: float,
    detailed: bool = False,
) -> tuple[dict[str, float | int], list[dict[str, float | str]], list[dict[str, float | int | str]]]:
    baseline.eval()
    aligned_model.eval()
    orthogonal_model.eval()
    records: list[dict[str, float | str]] = []
    for batch in loader:
        blurry = batch["blur"].to(device, non_blocking=True)
        sharp = batch["sharp"].to(device, non_blocking=True)
        aligned_geometry, orthogonal_geometry = _paired_geometries(
            blurry,
            sharp,
            oracle_window,
            max_feature_length,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            baseline_output = baseline(blurry)
            aligned_output = aligned_model(blurry, geometry_override=aligned_geometry)
            orthogonal_output = orthogonal_model(blurry, geometry_override=orthogonal_geometry)
        name = str(batch["name"][0])
        scene = name.split("/", maxsplit=1)[0]
        baseline_psnr = _psnr_value(baseline_output, sharp)
        aligned_psnr = _psnr_value(aligned_output, sharp)
        orthogonal_psnr = _psnr_value(orthogonal_output, sharp)
        record: dict[str, float | str] = {
            "recording_group": recording_group(scene),
            "scene": scene,
            "name": name,
            "input_psnr_db": _psnr_value(blurry, sharp),
            "baseline_psnr_db": baseline_psnr,
            "aligned_psnr_db": aligned_psnr,
            "orthogonal_psnr_db": orthogonal_psnr,
            "aligned_minus_baseline_db": aligned_psnr - baseline_psnr,
            "orthogonal_minus_baseline_db": orthogonal_psnr - baseline_psnr,
            "aligned_minus_orthogonal_db": aligned_psnr - orthogonal_psnr,
            "oracle_length_mean": float(aligned_geometry["length"].mean()),
            "oracle_confidence_mean": float(aligned_geometry["confidence"].mean()),
        }
        if detailed:
            baseline_ssim = _ssim_value(baseline_output[0], sharp[0])
            aligned_ssim = _ssim_value(aligned_output[0], sharp[0])
            orthogonal_ssim = _ssim_value(orthogonal_output[0], sharp[0])
            record.update(
                {
                    "input_ssim": _ssim_value(blurry[0], sharp[0]),
                    "baseline_ssim": baseline_ssim,
                    "aligned_ssim": aligned_ssim,
                    "orthogonal_ssim": orthogonal_ssim,
                    "aligned_minus_baseline_ssim": aligned_ssim - baseline_ssim,
                    "orthogonal_minus_baseline_ssim": orthogonal_ssim - baseline_ssim,
                    "aligned_minus_orthogonal_ssim": aligned_ssim - orthogonal_ssim,
                }
            )
        records.append(record)

    numeric_fields = [
        "input_psnr_db",
        "baseline_psnr_db",
        "aligned_psnr_db",
        "orthogonal_psnr_db",
        "aligned_minus_baseline_db",
        "orthogonal_minus_baseline_db",
        "aligned_minus_orthogonal_db",
        "oracle_length_mean",
        "oracle_confidence_mean",
    ]
    if detailed:
        numeric_fields.extend(
            [
                "input_ssim",
                "baseline_ssim",
                "aligned_ssim",
                "orthogonal_ssim",
                "aligned_minus_baseline_ssim",
                "orthogonal_minus_baseline_ssim",
                "aligned_minus_orthogonal_ssim",
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
    per_group: list[dict[str, float | int | str]] = [
        {"recording_group": group_name, **aggregate(items)}
        for group_name, items in sorted(grouped.items())
    ]
    return overall, records, per_group


def _train_step(
    model: CompactDeblurUNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    blurry: torch.Tensor,
    sharp: torch.Tensor,
    geometry: dict[str, torch.Tensor],
    frequency_weight: float,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = model(blurry, geometry_override=geometry)
        loss = restoration_loss(prediction, sharp, frequency_weight=frequency_weight)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(_trainable_parameters(model), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach())


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

    set_determinism(args.seed)
    device = torch.device("cuda:0")
    checkpoint_path = args.baseline_checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"baseline checkpoint not found: {checkpoint_path}")
    baseline_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    baseline_args = baseline_checkpoint.get("args", {})
    if baseline_args.get("variant") != "local":
        raise ValueError("The baseline checkpoint must use variant=local")
    if baseline_args.get("validation_source") != "train_holdout":
        raise ValueError("The baseline checkpoint must use validation_source=train_holdout")
    validation_groups = tuple(baseline_args.get("validation_groups") or ())
    if not validation_groups:
        raise ValueError("The baseline checkpoint does not contain frozen validation groups")

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

    common_block = TrajectoryNonlocalBlock(
        channels=width * 4,
        samples=args.trajectory_samples,
        max_feature_length=args.max_feature_length,
        fusion_mode=args.trajectory_fusion,
        angle_hypotheses=1,
    )
    aligned_model = _make_augmented_model(
        baseline_checkpoint["model"],
        width,
        copy.deepcopy(common_block),
    ).to(device)
    orthogonal_model = _make_augmented_model(
        baseline_checkpoint["model"],
        width,
        copy.deepcopy(common_block),
    ).to(device)
    aligned_optimizer = torch.optim.AdamW(
        _trainable_parameters(aligned_model),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    orthogonal_optimizer = torch.optim.AdamW(
        _trainable_parameters(orthogonal_model),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    aligned_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(aligned_optimizer, T_max=args.steps)
    orthogonal_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(orthogonal_optimizer, T_max=args.steps)
    aligned_scaler = torch.amp.GradScaler("cuda")
    orthogonal_scaler = torch.amp.GradScaler("cuda")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    initial, _, initial_groups = evaluate(
        baseline,
        aligned_model,
        orthogonal_model,
        val_loader,
        device,
        args.oracle_window,
        args.max_feature_length,
    )
    if abs(float(initial["aligned_minus_baseline_db"])) > 1e-8:
        raise RuntimeError("aligned identity initialization does not reproduce B0")
    if abs(float(initial["orthogonal_minus_baseline_db"])) > 1e-8:
        raise RuntimeError("orthogonal identity initialization does not reproduce B0")

    metadata = {"args": vars(args), "baseline_checkpoint": str(checkpoint_path)}
    torch.save({"step": 0, "model": aligned_model.state_dict(), **metadata}, output / "best_aligned.pt")
    torch.save({"step": 0, "model": orthogonal_model.state_dict(), **metadata}, output / "best_orthogonal.pt")
    best_aligned = float(initial["aligned_psnr_db"])
    best_orthogonal = float(initial["orthogonal_psnr_db"])
    history: list[dict[str, float | int]] = [
        {
            "step": 0,
            "aligned_train_loss": float("nan"),
            "orthogonal_train_loss": float("nan"),
            "baseline_psnr_db": float(initial["baseline_psnr_db"]),
            "aligned_psnr_db": float(initial["aligned_psnr_db"]),
            "orthogonal_psnr_db": float(initial["orthogonal_psnr_db"]),
            "aligned_minus_baseline_db": float(initial["aligned_minus_baseline_db"]),
            "aligned_minus_orthogonal_db": float(initial["aligned_minus_orthogonal_db"]),
            "minimum_group_aligned_minus_baseline_db": min(
                float(group["aligned_minus_baseline_db"]) for group in initial_groups
            ),
            "learning_rate": args.learning_rate,
        }
    ]

    train_iterator = iter(train_loader)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    progress = tqdm(range(1, args.steps + 1), desc="P6-A aligned vs orthogonal")
    for step in progress:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        blurry = batch["blur"].to(device, non_blocking=True)
        sharp = batch["sharp"].to(device, non_blocking=True)
        aligned_geometry, orthogonal_geometry = _paired_geometries(
            blurry,
            sharp,
            args.oracle_window,
            args.max_feature_length,
        )
        aligned_loss = _train_step(
            aligned_model,
            aligned_optimizer,
            aligned_scaler,
            blurry,
            sharp,
            aligned_geometry,
            args.frequency_weight,
        )
        orthogonal_loss = _train_step(
            orthogonal_model,
            orthogonal_optimizer,
            orthogonal_scaler,
            blurry,
            sharp,
            orthogonal_geometry,
            args.frequency_weight,
        )
        aligned_scheduler.step()
        orthogonal_scheduler.step()
        if step == 1 or step % 20 == 0:
            progress.set_postfix(aligned=f"{aligned_loss:.4f}", orthogonal=f"{orthogonal_loss:.4f}")

        if step % args.val_interval == 0 or step == args.steps:
            validation, _, groups = evaluate(
                baseline,
                aligned_model,
                orthogonal_model,
                val_loader,
                device,
                args.oracle_window,
                args.max_feature_length,
            )
            history.append(
                {
                    "step": step,
                    "aligned_train_loss": aligned_loss,
                    "orthogonal_train_loss": orthogonal_loss,
                    "baseline_psnr_db": float(validation["baseline_psnr_db"]),
                    "aligned_psnr_db": float(validation["aligned_psnr_db"]),
                    "orthogonal_psnr_db": float(validation["orthogonal_psnr_db"]),
                    "aligned_minus_baseline_db": float(validation["aligned_minus_baseline_db"]),
                    "aligned_minus_orthogonal_db": float(validation["aligned_minus_orthogonal_db"]),
                    "minimum_group_aligned_minus_baseline_db": min(
                        float(group["aligned_minus_baseline_db"]) for group in groups
                    ),
                    "learning_rate": aligned_optimizer.param_groups[0]["lr"],
                }
            )
            print(
                f"\nstep={step:05d} B0={validation['baseline_psnr_db']:.4f} dB "
                f"aligned={validation['aligned_psnr_db']:.4f} dB "
                f"orthogonal={validation['orthogonal_psnr_db']:.4f} dB "
                f"aligned_vs_B0={validation['aligned_minus_baseline_db']:+.4f} dB "
                f"aligned_vs_orth={validation['aligned_minus_orthogonal_db']:+.4f} dB"
            )
            if float(validation["aligned_psnr_db"]) > best_aligned:
                best_aligned = float(validation["aligned_psnr_db"])
                torch.save(
                    {"step": step, "model": aligned_model.state_dict(), **metadata},
                    output / "best_aligned.pt",
                )
            if float(validation["orthogonal_psnr_db"]) > best_orthogonal:
                best_orthogonal = float(validation["orthogonal_psnr_db"])
                torch.save(
                    {"step": step, "model": orthogonal_model.state_dict(), **metadata},
                    output / "best_orthogonal.pt",
                )

    _write_csv(output / "metrics.csv", history)
    torch.save({"step": args.steps, "model": aligned_model.state_dict(), **metadata}, output / "last_aligned.pt")
    torch.save(
        {"step": args.steps, "model": orthogonal_model.state_dict(), **metadata},
        output / "last_orthogonal.pt",
    )

    best_aligned_checkpoint = torch.load(output / "best_aligned.pt", map_location=device, weights_only=False)
    best_orthogonal_checkpoint = torch.load(
        output / "best_orthogonal.pt",
        map_location=device,
        weights_only=False,
    )
    aligned_model.load_state_dict(best_aligned_checkpoint["model"], strict=True)
    orthogonal_model.load_state_dict(best_orthogonal_checkpoint["model"], strict=True)
    final, per_image, per_group = evaluate(
        baseline,
        aligned_model,
        orthogonal_model,
        val_loader,
        device,
        args.oracle_window,
        args.max_feature_length,
        detailed=True,
    )
    _write_csv(output / "per_image_metrics.csv", per_image)
    _write_csv(output / "per_group_metrics.csv", per_group)

    minimum_group_delta = min(float(group["aligned_minus_baseline_db"]) for group in per_group)
    b0_psnr_passed = float(final["aligned_minus_baseline_db"]) >= 0.10
    direction_passed = float(final["aligned_minus_orthogonal_db"]) >= 0.05
    ssim_passed = float(final["aligned_minus_baseline_ssim"]) >= 0.0
    group_passed = minimum_group_delta >= -0.01
    report = {
        "experiment": output.name,
        "diagnostic": "paired-image pseudo oracle; unavailable at inference",
        "baseline_checkpoint": str(checkpoint_path),
        "baseline_checkpoint_step": int(baseline_checkpoint["step"]),
        "aligned_checkpoint_step": int(best_aligned_checkpoint["step"]),
        "orthogonal_checkpoint_step": int(best_orthogonal_checkpoint["step"]),
        "seed": args.seed,
        "steps": args.steps,
        "train_pairs": len(train_set),
        "val_pairs": len(val_set),
        "train_patch_size": args.train_patch_size,
        "validation_patch_size": int(baseline_args["val_patch_size"]),
        "validation_recording_groups": list(validation_groups),
        "trajectory_samples": args.trajectory_samples,
        "max_feature_length": args.max_feature_length,
        "trajectory_fusion": args.trajectory_fusion,
        "oracle_window": args.oracle_window,
        "aligned_total_parameters": sum(parameter.numel() for parameter in aligned_model.parameters()),
        "aligned_trainable_parameters": sum(
            parameter.numel() for parameter in aligned_model.parameters() if parameter.requires_grad
        ),
        **final,
        "per_group": per_group,
        "minimum_group_aligned_minus_baseline_db": minimum_group_delta,
        "images_aligned_better_than_baseline": sum(
            float(record["aligned_minus_baseline_db"]) > 0 for record in per_image
        ),
        "images_aligned_better_than_orthogonal": sum(
            float(record["aligned_minus_orthogonal_db"]) > 0 for record in per_image
        ),
        "aligned_over_baseline_threshold_db": 0.10,
        "aligned_over_orthogonal_threshold_db": 0.05,
        "minimum_group_threshold_db": -0.01,
        "baseline_psnr_gate_passed": b0_psnr_passed,
        "direction_gate_passed": direction_passed,
        "ssim_gate_passed": ssim_passed,
        "group_gate_passed": group_passed,
        "passed": b0_psnr_passed and direction_passed and ssim_passed and group_passed,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "wall_seconds": time.perf_counter() - started,
        "official_test_accessed": False,
        "paper_result": False,
        "note": (
            "P6-A upper-bound diagnostic. Sharp images create geometry at validation, so these "
            "numbers are not deployable or reportable benchmark results."
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
