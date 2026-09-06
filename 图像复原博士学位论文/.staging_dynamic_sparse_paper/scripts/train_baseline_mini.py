"""Short GoPro run used to establish the real-data baseline before ablations."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GoProPairDataset, make_motion_batch  # noqa: E402
from src.losses import restoration_loss  # noqa: E402
from src.metrics import psnr  # noqa: E402
from src.models import CompactDeblurUNet, TrajectoryNonlocalBlock  # noqa: E402
from src.models.compact_unet import count_parameters  # noqa: E402
from src.models.motion_geometry import geometry_supervision_loss  # noqa: E402
from src.models.motion_geometry import (  # noqa: E402
    paired_direction_supervision_loss,
    paired_motion_direction_pseudo_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the compact baseline on a controlled GoPro subset.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "GoPro")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "runs" / "P1_baseline_mini")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--val-patch-size", type=int, default=256)
    parser.add_argument("--train-pairs", type=int, default=64, help="0 uses the complete training split.")
    parser.add_argument("--val-pairs", type=int, default=16, help="0 uses the complete validation source.")
    parser.add_argument(
        "--validation-source",
        choices=("official_test", "train_holdout"),
        default="official_test",
        help="Use the official test split or held-out recording groups from the training split.",
    )
    parser.add_argument(
        "--validation-groups",
        nargs="+",
        help="Recording groups held out when --validation-source=train_holdout.",
    )
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--variant", choices=("local", "mt_nlr"), default="local")
    parser.add_argument("--trajectory-samples", type=int, default=5)
    parser.add_argument("--trajectory-hypotheses", type=int, default=1)
    parser.add_argument("--max-angle-offset-degrees", type=float, default=30.0)
    parser.add_argument("--max-feature-length", type=float, default=7.0)
    parser.add_argument("--statistics-window", type=int, default=51)
    parser.add_argument("--trajectory-fusion", choices=("aggregate", "detail"), default="aggregate")
    parser.add_argument("--geometry-weight", type=float, default=0.0)
    parser.add_argument("--geometry-checkpoint", type=Path)
    parser.add_argument("--geometry-every", type=int, default=4)
    parser.add_argument("--synthetic-batch-size", type=int, default=4)
    parser.add_argument("--synthetic-size", type=int, default=64)
    parser.add_argument("--paired-direction-weight", type=float, default=0.0)
    parser.add_argument("--paired-direction-window", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--frequency-weight", type=float, default=0.1)
    parser.add_argument("--val-interval", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume", type=Path, help="Resume from a latest.pt written at a validation point.")
    parser.add_argument("--stop-after", type=int, help="Stop cleanly at this step while retaining the full LR schedule.")
    parser.add_argument(
        "--comparison-baseline-psnr",
        type=float,
        help="Optional frozen baseline PSNR used only to report a method-vs-baseline delta.",
    )
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    input_values: list[float] = []
    output_values: list[float] = []
    for batch in loader:
        blurry = batch["blur"].to(device, non_blocking=True)
        sharp = batch["sharp"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            restored = model(blurry)
        for prediction, target in zip(restored, sharp):
            output_values.append(psnr(prediction.unsqueeze(0), target.unsqueeze(0)))
        for source, target in zip(blurry, sharp):
            input_values.append(psnr(source.unsqueeze(0), target.unsqueeze(0)))
    return float(np.mean(input_values)), float(np.mean(output_values))


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script has no CPU fallback")
    if args.steps <= 0 or args.val_interval <= 0:
        raise ValueError("steps and val-interval must be positive")
    if args.geometry_weight < 0 or args.geometry_every <= 0 or args.paired_direction_weight < 0:
        raise ValueError("geometry-weight must be nonnegative and geometry-every must be positive")
    if args.variant == "local" and args.geometry_weight > 0:
        raise ValueError("geometry supervision is only available for the mt_nlr variant")
    if args.variant == "local" and args.geometry_checkpoint is not None:
        raise ValueError("geometry checkpoint is only available for the mt_nlr variant")
    if args.variant == "local" and args.paired_direction_weight > 0:
        raise ValueError("paired direction supervision is only available for the mt_nlr variant")
    if args.paired_direction_window < 3 or args.paired_direction_window % 2 == 0:
        raise ValueError("paired-direction-window must be an odd integer >= 3")
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        raise ValueError("stop-after must be in [1, steps]")
    if args.validation_source == "train_holdout" and not args.validation_groups:
        raise ValueError("validation-groups are required for validation-source=train_holdout")
    if args.validation_source == "official_test" and args.validation_groups:
        raise ValueError("validation-groups can only be used with validation-source=train_holdout")

    set_determinism(args.seed)
    device = torch.device("cuda:0")
    validation_groups = tuple(args.validation_groups or ())
    train_set = GoProPairDataset(
        args.data_root,
        split="train",
        patch_size=args.patch_size,
        augment=True,
        max_pairs=args.train_pairs or None,
        exclude_recording_groups=validation_groups,
    )
    val_set = GoProPairDataset(
        args.data_root,
        split="train" if args.validation_source == "train_holdout" else "test",
        patch_size=args.val_patch_size,
        augment=False,
        max_pairs=args.val_pairs or None,
        include_recording_groups=validation_groups or None,
    )
    recording_group_overlap = sorted(set(train_set.recording_groups) & set(val_set.recording_groups))
    if args.validation_source == "train_holdout" and recording_group_overlap:
        raise RuntimeError(
            "Training/validation recording-group leakage detected: "
            + ", ".join(recording_group_overlap)
        )
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

    # Construct the common backbone first. Otherwise the variant module consumes
    # RNG state and silently changes every shared U-Net initialization.
    model = CompactDeblurUNet(width=args.width)
    if args.variant == "mt_nlr":
        model.bottleneck_module = TrajectoryNonlocalBlock(
            channels=args.width * 4,
            samples=args.trajectory_samples,
            max_feature_length=args.max_feature_length,
            statistics_window=args.statistics_window,
            fusion_mode=args.trajectory_fusion,
            angle_hypotheses=args.trajectory_hypotheses,
            max_angle_offset_degrees=args.max_angle_offset_degrees,
        )
    loaded_geometry_tensors = 0
    if args.geometry_checkpoint is not None:
        checkpoint_path = args.geometry_checkpoint.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"geometry checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        source_state = checkpoint["model"]
        target_state = model.state_dict()
        geometry_prefixes = (
            "bottleneck_module.normalization.",
            "bottleneck_module.geometry_features.",
            "bottleneck_module.statistics_projection.",
            "bottleneck_module.direction_residual.",
            "bottleneck_module.length_head.",
            "bottleneck_module.confidence_head.",
        )
        for name, value in source_state.items():
            if not name.startswith(geometry_prefixes):
                continue
            if name not in target_state or target_state[name].shape != value.shape:
                raise ValueError(f"incompatible geometry checkpoint tensor: {name}")
            target_state[name] = value
            loaded_geometry_tensors += 1
        if loaded_geometry_tensors == 0:
            raise ValueError("geometry checkpoint did not contain any recognized tensors")
        model.load_state_dict(target_state)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_psnr = float("-inf")
    input_psnr = float("nan")
    start_step = 1
    elapsed_before = 0.0
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        history = list(resume["history"])
        best_psnr = float(resume["best_psnr"])
        input_psnr = float(resume["input_psnr"])
        elapsed_before = float(resume.get("train_wall_seconds", 0.0))
        start_step = int(resume["step"]) + 1
        if start_step > args.steps:
            raise ValueError(f"resume step {start_step - 1} already reaches requested steps {args.steps}")
    train_iterator = iter(train_loader)
    torch.cuda.reset_peak_memory_stats(device)
    final_step = args.stop_after or args.steps
    if start_step > final_step:
        raise ValueError(f"resume step {start_step - 1} already reaches stop-after {final_step}")

    progress = tqdm(
        range(start_step, final_step + 1),
        initial=start_step - 1,
        total=final_step,
        desc=f"GoPro mini {args.variant}",
    )
    train_started = time.perf_counter()
    for step in progress:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        blurry = batch["blur"].to(device, non_blocking=True)
        sharp = batch["sharp"].to(device, non_blocking=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            if args.paired_direction_weight > 0:
                prediction, real_geometry = model(blurry, return_geometry=True)
            else:
                prediction = model(blurry)
                real_geometry = None
            restoration = restoration_loss(prediction, sharp, frequency_weight=args.frequency_weight)
        geometry_value = float("nan")
        paired_direction_value = float("nan")
        loss = restoration
        if args.paired_direction_weight > 0:
            pseudo_direction, pseudo_confidence = paired_motion_direction_pseudo_label(
                blurry,
                sharp,
                real_geometry["double_angle"].shape[-2:],
                window=args.paired_direction_window,
            )
            paired_direction_loss = paired_direction_supervision_loss(
                real_geometry["double_angle"].float(),
                pseudo_direction,
                pseudo_confidence,
            )
            loss = loss + args.paired_direction_weight * paired_direction_loss
            paired_direction_value = float(paired_direction_loss.detach())
        if args.geometry_weight > 0 and step % args.geometry_every == 0:
            synthetic_blur, _, synthetic_direction, synthetic_length = make_motion_batch(
                batch_size=args.synthetic_batch_size,
                size=args.synthetic_size,
                seed=args.seed + 1_000_000 + step,
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                geometry = model.predict_geometry(synthetic_blur)
            feature_length = 1.0 + (synthetic_length - 1.0) / 4.0
            geometry_loss, _ = geometry_supervision_loss(
                geometry["double_angle"].float(),
                geometry["length"].float(),
                geometry["confidence"].float(),
                synthetic_direction,
                feature_length,
                max_length=args.max_feature_length,
            )
            loss = loss + args.geometry_weight * geometry_loss
            geometry_value = float(geometry_loss.detach())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_value = float(loss.detach())
        if step == 1 or step % 20 == 0:
            progress.set_postfix(loss=f"{loss_value:.4f}")

        if step == 1 or step % args.val_interval == 0 or step == final_step:
            input_psnr, val_psnr = evaluate(model, val_loader, device)
            record = {
                "step": step,
                "train_loss": loss_value,
                "restoration_loss": float(restoration.detach()),
                "geometry_loss": geometry_value,
                "paired_direction_loss": paired_direction_value,
                "input_psnr_db": input_psnr,
                "val_psnr_db": val_psnr,
                "gain_over_input_db": val_psnr - input_psnr,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            print(
                f"\nstep={step:06d} loss={loss_value:.6f} input_psnr={input_psnr:.3f} dB "
                f"val_psnr={val_psnr:.3f} dB gain={val_psnr - input_psnr:+.3f} dB"
            )
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(
                    {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args)},
                    output / "best.pt",
                )
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "history": history,
                    "best_psnr": best_psnr,
                    "input_psnr": input_psnr,
                    "train_wall_seconds": elapsed_before + (time.perf_counter() - train_started),
                    "args": vars(args),
                },
                output / "latest.pt",
            )

    torch.save(
        {
            "step": final_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        },
        output / "last.pt",
    )
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(history)

    gain_over_input = best_psnr - input_psnr
    delta_vs_comparison = (
        best_psnr - args.comparison_baseline_psnr
        if args.comparison_baseline_psnr is not None
        else None
    )
    train_wall_seconds = elapsed_before + (time.perf_counter() - train_started)
    report = {
        "experiment": output.name,
        "variant": args.variant,
        "seed": args.seed,
        "steps": args.steps,
        "completed_steps": final_step,
        "training_complete": final_step == args.steps,
        "data_root": str(args.data_root.expanduser().resolve()),
        "validation_source": args.validation_source,
        "validation_recording_groups": list(validation_groups),
        "train_recording_groups": list(train_set.recording_groups),
        "val_recording_groups": list(val_set.recording_groups),
        "recording_group_overlap": recording_group_overlap,
        "train_pairs": len(train_set),
        "val_pairs": len(val_set),
        "patch_size": args.patch_size,
        "val_patch_size": args.val_patch_size,
        "parameters": count_parameters(model),
        "trajectory_samples": args.trajectory_samples if args.variant == "mt_nlr" else None,
        "trajectory_hypotheses": args.trajectory_hypotheses if args.variant == "mt_nlr" else None,
        "max_angle_offset_degrees": args.max_angle_offset_degrees if args.variant == "mt_nlr" else None,
        "max_feature_length": args.max_feature_length if args.variant == "mt_nlr" else None,
        "statistics_window": args.statistics_window if args.variant == "mt_nlr" else None,
        "trajectory_fusion": args.trajectory_fusion if args.variant == "mt_nlr" else None,
        "geometry_weight": args.geometry_weight if args.variant == "mt_nlr" else None,
        "paired_direction_weight": args.paired_direction_weight if args.variant == "mt_nlr" else None,
        "paired_direction_window": args.paired_direction_window if args.paired_direction_weight > 0 else None,
        "geometry_checkpoint": str(args.geometry_checkpoint.expanduser().resolve()) if args.geometry_checkpoint else None,
        "loaded_geometry_tensors": loaded_geometry_tensors,
        "geometry_every": args.geometry_every if args.geometry_weight > 0 else None,
        "synthetic_batch_size": args.synthetic_batch_size if args.geometry_weight > 0 else None,
        "synthetic_size": args.synthetic_size if args.geometry_weight > 0 else None,
        "input_psnr_db": input_psnr,
        "best_val_psnr_db": best_psnr,
        "gain_over_input_db": gain_over_input,
        "input_gain_gate_passed": gain_over_input >= 0.1,
        "comparison_baseline_psnr_db": args.comparison_baseline_psnr,
        "delta_vs_comparison_baseline_db": delta_vs_comparison,
        "method_gate_passed": delta_vs_comparison > 0.0 if delta_vs_comparison is not None else None,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "train_wall_seconds": train_wall_seconds,
        "passed": gain_over_input >= 0.1,
        "passed_definition": "gain_over_input_db >= 0.1; this does not mean a variant beats B0",
        "paper_result": False,
        "note": "Short subset run for pipeline validation; do not report as a benchmark result.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
