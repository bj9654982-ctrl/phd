"""Train a minimal image-level residual gate while keeping the B0 model frozen."""

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
    apply_residual_gate,
    mse_optimal_residual_alpha,
)


DEFAULT_BASELINE_RUN = PROJECT_ROOT / "results" / "runs" / "P3_baseline_group_split_v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a scalar residual gate on a frozen B0 baseline.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "GOPRO_Large")
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_RUN / "best.pt")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "P4_scalar_residual_gate_v0",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-pairs", type=int, default=0, help="0 uses all allowed training pairs.")
    parser.add_argument("--val-pairs", type=int, default=0, help="0 uses the complete frozen holdout.")
    parser.add_argument(
        "--gate-train-patch-size",
        type=int,
        help="Training crop used by the scalar gate; defaults to the B0 training crop.",
    )
    parser.add_argument("--gate-width", type=int, default=16)
    parser.add_argument("--initial-alpha", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--frequency-weight", type=float, default=0.1)
    parser.add_argument("--oracle-weight", type=float, default=1.0)
    parser.add_argument("--restoration-weight", type=float, default=0.1)
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


@torch.no_grad()
def evaluate(
    baseline: CompactDeblurUNet,
    gate: ImageResidualGate,
    loader: DataLoader,
    device: torch.device,
    detailed: bool = False,
) -> tuple[dict[str, float], list[dict[str, float | str]], list[dict[str, float | int | str]]]:
    baseline.eval()
    gate.eval()
    records: list[dict[str, float | str]] = []
    for batch in loader:
        blurry = batch["blur"].to(device, non_blocking=True)
        sharp = batch["sharp"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            baseline_restored = baseline(blurry)
        residual = baseline_restored.float() - blurry.float()
        alpha = gate(blurry.float(), residual)
        gated_restored = apply_residual_gate(blurry.float(), baseline_restored.float(), alpha)
        oracle_alpha = mse_optimal_residual_alpha(blurry, baseline_restored, sharp)
        name = str(batch["name"][0])
        scene = name.split("/", maxsplit=1)[0]
        input_psnr = _psnr_value(blurry, sharp)
        baseline_psnr = _psnr_value(baseline_restored, sharp)
        gated_psnr = _psnr_value(gated_restored, sharp)
        record: dict[str, float | str] = {
            "recording_group": recording_group(scene),
            "scene": scene,
            "name": name,
            "input_psnr_db": input_psnr,
            "baseline_psnr_db": baseline_psnr,
            "gated_psnr_db": gated_psnr,
            "gated_minus_baseline_db": gated_psnr - baseline_psnr,
            "alpha": float(alpha),
            "oracle_alpha": float(oracle_alpha),
            "alpha_absolute_error": float((alpha - oracle_alpha).abs()),
        }
        if detailed:
            input_ssim = _ssim_value(blurry[0], sharp[0])
            baseline_ssim = _ssim_value(baseline_restored[0], sharp[0])
            gated_ssim = _ssim_value(gated_restored[0], sharp[0])
            record.update(
                {
                    "input_ssim": input_ssim,
                    "baseline_ssim": baseline_ssim,
                    "gated_ssim": gated_ssim,
                    "gated_minus_baseline_ssim": gated_ssim - baseline_ssim,
                }
            )
        records.append(record)

    numeric_fields = [
        "input_psnr_db",
        "baseline_psnr_db",
        "gated_psnr_db",
        "gated_minus_baseline_db",
        "alpha",
        "oracle_alpha",
        "alpha_absolute_error",
    ]
    if detailed:
        numeric_fields.extend(
            ["input_ssim", "baseline_ssim", "gated_ssim", "gated_minus_baseline_ssim"]
        )

    def aggregate(items: list[dict[str, float | str]]) -> dict[str, float | int]:
        return {
            "pairs": len(items),
            **{
                field: statistics.fmean(float(item[field]) for item in items)
                for field in numeric_fields
            },
            "alpha_std": statistics.pstdev(float(item["alpha"]) for item in items),
            "alpha_min": min(float(item["alpha"]) for item in items),
            "alpha_max": max(float(item["alpha"]) for item in items),
        }

    overall = aggregate(records)
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for record in records:
        grouped[str(record["recording_group"])].append(record)
    group_summaries: list[dict[str, float | int | str]] = [
        {"recording_group": group_name, **aggregate(items)}
        for group_name, items in sorted(grouped.items())
    ]
    return {key: float(value) for key, value in overall.items()}, records, group_summaries


def _write_csv(path: Path, records: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


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
    if args.oracle_weight < 0 or args.restoration_weight < 0:
        raise ValueError("oracle-weight and restoration-weight must be nonnegative")
    if args.oracle_weight == 0 and args.restoration_weight == 0:
        raise ValueError("at least one gate training loss weight must be positive")

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

    gate_train_patch_size = args.gate_train_patch_size or int(baseline_args["patch_size"])
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
    gate = ImageResidualGate(width=args.gate_width, initial_alpha=args.initial_alpha).to(device)
    gate_parameters = sum(parameter.numel() for parameter in gate.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    initial, _, initial_groups = evaluate(baseline, gate, val_loader, device)
    best_psnr = initial["gated_psnr_db"]
    best_step = 0
    torch.save(
        {"step": 0, "gate": gate.state_dict(), "args": vars(args), "baseline_checkpoint": str(checkpoint_path)},
        output / "best_gate.pt",
    )
    history.append(
        {
            "step": 0,
            "train_loss": float("nan"),
            "train_oracle_loss": float("nan"),
            "train_restoration_loss": float("nan"),
            "baseline_psnr_db": initial["baseline_psnr_db"],
            "gated_psnr_db": initial["gated_psnr_db"],
            "gated_minus_baseline_db": initial["gated_minus_baseline_db"],
            "alpha_mean": initial["alpha"],
            "alpha_std": initial["alpha_std"],
            "minimum_group_delta_db": min(float(group["gated_minus_baseline_db"]) for group in initial_groups),
            "learning_rate": args.learning_rate,
        }
    )

    train_iterator = iter(train_loader)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    progress = tqdm(range(1, args.steps + 1), desc="Frozen B0 scalar gate")
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
        gate.train()
        optimizer.zero_grad(set_to_none=True)
        alpha = gate(blurry.float(), baseline_restored.float() - blurry.float())
        gated_restored = apply_residual_gate(blurry.float(), baseline_restored.float(), alpha)
        oracle_alpha = mse_optimal_residual_alpha(blurry, baseline_restored, sharp)
        oracle_loss = F.smooth_l1_loss(alpha, oracle_alpha)
        restore_loss = restoration_loss(gated_restored, sharp.float(), frequency_weight=args.frequency_weight)
        loss = args.oracle_weight * oracle_loss + args.restoration_weight * restore_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(gate.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        loss_value = float(loss.detach())
        if step == 1 or step % 20 == 0:
            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                alpha=f"{float(alpha.mean()):.3f}",
                target=f"{float(oracle_alpha.mean()):.3f}",
            )

        if step % args.val_interval == 0 or step == args.steps:
            validation, _, groups = evaluate(baseline, gate, val_loader, device)
            group_minimum = min(float(group["gated_minus_baseline_db"]) for group in groups)
            history.append(
                {
                    "step": step,
                    "train_loss": loss_value,
                    "train_oracle_loss": float(oracle_loss.detach()),
                    "train_restoration_loss": float(restore_loss.detach()),
                    "baseline_psnr_db": validation["baseline_psnr_db"],
                    "gated_psnr_db": validation["gated_psnr_db"],
                    "gated_minus_baseline_db": validation["gated_minus_baseline_db"],
                    "alpha_mean": validation["alpha"],
                    "alpha_std": validation["alpha_std"],
                    "minimum_group_delta_db": group_minimum,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            print(
                f"\nstep={step:05d} baseline={validation['baseline_psnr_db']:.4f} dB "
                f"gated={validation['gated_psnr_db']:.4f} dB "
                f"delta={validation['gated_minus_baseline_db']:+.4f} dB "
                f"alpha={validation['alpha']:.3f}+/-{validation['alpha_std']:.3f} "
                f"oracle_mae={validation['alpha_absolute_error']:.3f}"
            )
            if validation["gated_psnr_db"] > best_psnr:
                best_psnr = validation["gated_psnr_db"]
                best_step = step
                torch.save(
                    {
                        "step": step,
                        "gate": gate.state_dict(),
                        "args": vars(args),
                        "baseline_checkpoint": str(checkpoint_path),
                    },
                    output / "best_gate.pt",
                )

    torch.save(
        {
            "step": args.steps,
            "gate": gate.state_dict(),
            "args": vars(args),
            "baseline_checkpoint": str(checkpoint_path),
        },
        output / "last_gate.pt",
    )
    _write_csv(output / "metrics.csv", history)

    best_gate_checkpoint = torch.load(output / "best_gate.pt", map_location=device, weights_only=False)
    gate.load_state_dict(best_gate_checkpoint["gate"], strict=True)
    final, per_image, per_group = evaluate(baseline, gate, val_loader, device, detailed=True)
    _write_csv(output / "per_image_metrics.csv", per_image)
    _write_csv(output / "per_group_metrics.csv", per_group)
    predicted_alphas = np.array([float(record["alpha"]) for record in per_image])
    oracle_alphas = np.array([float(record["oracle_alpha"]) for record in per_image])
    alpha_correlation = (
        float(np.corrcoef(predicted_alphas, oracle_alphas)[0, 1])
        if np.std(predicted_alphas) > 0 and np.std(oracle_alphas) > 0
        else None
    )
    minimum_group_delta = min(float(group["gated_minus_baseline_db"]) for group in per_group)
    psnr_passed = final["gated_minus_baseline_db"] >= 0.05
    ssim_passed = final["gated_minus_baseline_ssim"] >= 0.0
    group_passed = minimum_group_delta >= -0.01
    report = {
        "experiment": output.name,
        "baseline_checkpoint": str(checkpoint_path),
        "baseline_checkpoint_step": int(baseline_checkpoint["step"]),
        "gate_checkpoint_step": int(best_gate_checkpoint["step"]),
        "seed": args.seed,
        "steps": args.steps,
        "train_pairs": len(train_set),
        "val_pairs": len(val_set),
        "gate_train_patch_size": gate_train_patch_size,
        "validation_patch_size": int(baseline_args["val_patch_size"]),
        "validation_recording_groups": list(validation_groups),
        "gate_parameters": gate_parameters,
        "initial_alpha": args.initial_alpha,
        "oracle_weight": args.oracle_weight,
        "restoration_weight": args.restoration_weight,
        "baseline_psnr_db": final["baseline_psnr_db"],
        "gated_psnr_db": final["gated_psnr_db"],
        "gated_minus_baseline_db": final["gated_minus_baseline_db"],
        "baseline_ssim": final["baseline_ssim"],
        "gated_ssim": final["gated_ssim"],
        "gated_minus_baseline_ssim": final["gated_minus_baseline_ssim"],
        "alpha_mean": final["alpha"],
        "alpha_std": final["alpha_std"],
        "alpha_min": final["alpha_min"],
        "alpha_max": final["alpha_max"],
        "oracle_alpha_mean": final["oracle_alpha"],
        "oracle_alpha_mae": final["alpha_absolute_error"],
        "predicted_vs_oracle_alpha_correlation": alpha_correlation,
        "per_group": per_group,
        "minimum_group_delta_db": minimum_group_delta,
        "psnr_threshold_db": 0.05,
        "minimum_group_threshold_db": -0.01,
        "psnr_gate_passed": psnr_passed,
        "ssim_gate_passed": ssim_passed,
        "group_gate_passed": group_passed,
        "passed": psnr_passed and ssim_passed and group_passed,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "wall_seconds": time.perf_counter() - started,
        "official_test_accessed": False,
        "paper_result": False,
        "note": "Frozen-B0 scalar-gate screening experiment; passing does not establish the final nonlocal method.",
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
