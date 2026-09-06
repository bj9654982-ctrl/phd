"""E0 gate: learn line-motion orientation and length from synthetic blurry images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import make_motion_batch  # noqa: E402
from src.models.motion_geometry import (  # noqa: E402
    GeometryProbe,
    geometry_metrics,
    geometry_supervision_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the E0 synthetic motion geometry probe.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fixed-train", action="store_true", help="Diagnostic: train on the fixed evaluation batch.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "runs" / "E0_geometry")
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fixed_evaluation_batch(size: int, seed: int):
    degrees = torch.tensor([0, 30, 60, 90, 120, 150], device="cuda", dtype=torch.float32)
    lengths = torch.tensor([5, 9, 13, 17, 21], device="cuda", dtype=torch.float32)
    angles = torch.deg2rad(degrees.repeat_interleave(len(lengths))).repeat(2)
    lengths = lengths.repeat(len(degrees)).repeat(2)
    return make_motion_batch(
        batch_size=len(angles),
        size=size,
        seed=seed,
        angles_radians=angles,
        lengths_pixels=lengths,
    )


@torch.no_grad()
def evaluate(model: GeometryProbe, batch) -> tuple[float, float]:
    blurry, _, target_direction, target_length = batch
    model.eval()
    direction, length, _ = model(blurry)
    return geometry_metrics(direction.float(), length.float(), target_direction, target_length)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script has no CPU fallback")
    if args.steps <= 0 or args.eval_interval <= 0:
        raise ValueError("steps and eval-interval must be positive")

    set_determinism(args.seed)
    device = torch.device("cuda:0")
    model = GeometryProbe(width=args.width, max_length=25.0).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    evaluation_batch = fixed_evaluation_batch(args.size, args.seed + 100_000)
    fixed_training_batch = evaluation_batch if args.fixed_train else None
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    best_angle = float("inf")
    best_length = float("inf")
    best_score = float("inf")
    torch.cuda.reset_peak_memory_stats(device)
    progress = tqdm(range(1, args.steps + 1), desc="E0 motion geometry")
    for step in progress:
        if fixed_training_batch is None:
            blurry, _, target_direction, target_length = make_motion_batch(
                batch_size=args.batch_size,
                size=args.size,
                seed=args.seed + step,
            )
        else:
            blurry, _, target_direction, target_length = fixed_training_batch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        direction, length, confidence = model(blurry)
        loss, components = geometry_supervision_loss(
            direction,
            length,
            confidence,
            target_direction,
            target_length,
            max_length=25.0,
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 20 == 0:
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            angle_error, length_error = evaluate(model, evaluation_batch)
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "direction_loss": float(components["direction"]),
                "length_loss": float(components["length"]),
                "angle_mae_degrees": angle_error,
                "length_mae_pixels": length_error,
            }
            history.append(record)
            print(
                f"\nstep={step:05d} angle_mae={angle_error:.3f} deg "
                f"length_mae={length_error:.3f} px"
            )
            score = angle_error / 10.0 + length_error / 2.0
            if score < best_score:
                best_score = score
                best_angle = angle_error
                best_length = length_error
                torch.save(
                    {"step": step, "model": model.state_dict(), "args": vars(args)},
                    output / "best.pt",
                )

    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    report = {
        "experiment": "E0_geometry",
        "seed": args.seed,
        "steps": args.steps,
        "parameter_count": parameter_count,
        "evaluation_directions_degrees": [0, 30, 60, 90, 120, 150],
        "evaluation_lengths_pixels": [5, 9, 13, 17, 21],
        "best_angle_mae_degrees": best_angle,
        "best_length_mae_pixels": best_length,
        "angle_threshold_degrees": 10.0,
        "length_threshold_pixels": 2.0,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "passed": best_angle < 10.0 and best_length < 2.0,
        "paper_result": False,
        "note": "Synthetic geometry learnability gate; real restoration evidence is still required.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
