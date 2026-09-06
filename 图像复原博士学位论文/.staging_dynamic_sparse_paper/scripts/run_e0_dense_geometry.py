"""E0-Dense gate for the spatial geometry head used by MT-NLR."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_e0_geometry import fixed_evaluation_batch  # noqa: E402
from src.data import make_motion_batch  # noqa: E402
from src.models import CompactDeblurUNet, TrajectoryNonlocalBlock  # noqa: E402
from src.models.motion_geometry import geometry_metrics, geometry_supervision_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dense MT-NLR geometry learnability.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--statistics-window", type=int, default=51)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "runs" / "E0_dense_geometry")
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def geometry_parameters(block: TrajectoryNonlocalBlock) -> list[torch.nn.Parameter]:
    modules = (
        block.normalization,
        block.geometry_features,
        block.statistics_projection,
        block.direction_residual,
        block.length_head,
        block.confidence_head,
    )
    return [parameter for module in modules for parameter in module.parameters()]


@torch.no_grad()
def evaluate(model: CompactDeblurUNet, batch) -> tuple[float, float]:
    blurry, _, target_direction, target_input_length = batch
    model.eval()
    geometry = model.predict_geometry(blurry)
    target_feature_length = 1.0 + (target_input_length - 1.0) / 4.0
    angle, feature_length_error = geometry_metrics(
        geometry["double_angle"].float(),
        geometry["length"].float(),
        target_direction,
        target_feature_length,
    )
    return angle, feature_length_error * 4.0


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script has no CPU fallback")
    set_determinism(args.seed)
    device = torch.device("cuda:0")
    model = CompactDeblurUNet(width=args.width)
    block = TrajectoryNonlocalBlock(
        channels=args.width * 4,
        samples=5,
        max_feature_length=7.0,
        statistics_window=args.statistics_window,
    )
    model.bottleneck_module = block
    model = model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = geometry_parameters(block)
    for parameter in trainable:
        parameter.requires_grad_(True)

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    evaluation_batch = fixed_evaluation_batch(args.size, args.seed + 100_000)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_score = float("inf")
    best_angle = float("inf")
    best_input_length = float("inf")
    torch.cuda.reset_peak_memory_stats(device)

    progress = tqdm(range(1, args.steps + 1), desc="E0 dense geometry")
    for step in progress:
        blurry, _, target_direction, target_input_length = make_motion_batch(
            args.batch_size,
            args.size,
            args.seed + step,
        )
        target_feature_length = 1.0 + (target_input_length - 1.0) / 4.0
        model.train()
        optimizer.zero_grad(set_to_none=True)
        geometry = model.predict_geometry(blurry)
        loss, components = geometry_supervision_loss(
            geometry["double_angle"].float(),
            geometry["length"].float(),
            geometry["confidence"].float(),
            target_direction,
            target_feature_length,
            max_length=7.0,
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 20 == 0:
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            angle, input_length_error = evaluate(model, evaluation_batch)
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "direction_loss": float(components["direction"]),
                "length_loss": float(components["length"]),
                "angle_mae_degrees": angle,
                "input_length_mae_pixels": input_length_error,
            }
            history.append(record)
            print(f"\nstep={step:05d} angle_mae={angle:.3f} deg input_length_mae={input_length_error:.3f} px")
            score = angle / 10.0 + input_length_error / 2.0
            if score < best_score:
                best_score = score
                best_angle = angle
                best_input_length = input_length_error
                torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, output / "best.pt")

    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    report = {
        "experiment": "E0_dense_geometry",
        "seed": args.seed,
        "steps": args.steps,
        "statistics_window": args.statistics_window,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "best_angle_mae_degrees": best_angle,
        "best_input_length_mae_pixels": best_input_length,
        "angle_threshold_degrees": 10.0,
        "input_length_threshold_pixels": 2.0,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "passed": best_angle < 10.0 and best_input_length < 2.0,
        "paper_result": False,
        "note": "Dense synthetic geometry gate; real restoration evidence is still required.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
