"""E2 gate for sharp-reference motion-direction pseudo labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_e0_geometry import fixed_evaluation_batch  # noqa: E402
from src.models.motion_geometry import paired_motion_direction_pseudo_label  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate paired-image direction pseudo labels.")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--windows", type=int, nargs="+", default=(15, 31, 51))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "E2_pseudo_direction_gate",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate_window(
    blurry: torch.Tensor,
    sharp: torch.Tensor,
    target_direction: torch.Tensor,
    window: int,
) -> dict[str, float | int]:
    output_size = (blurry.shape[-2] // 4, blurry.shape[-1] // 4)
    pseudo, confidence = paired_motion_direction_pseudo_label(
        blurry,
        sharp,
        output_size,
        window=window,
    )
    target = torch.cat(
        (
            target_direction[:, :1].square() - target_direction[:, 1:2].square(),
            2.0 * target_direction[:, :1] * target_direction[:, 1:2],
        ),
        dim=1,
    )[:, :, None, None]
    cosine = (pseudo * target).sum(dim=1).clamp(-1.0, 1.0)
    angle = 0.5 * torch.rad2deg(torch.acos(cosine))
    weights = confidence[:, 0]
    weighted_pixel_error = float((angle * weights).sum() / weights.sum().clamp_min(1e-8))

    pooled = (pseudo * confidence).sum(dim=(-2, -1))
    pooled = torch.nn.functional.normalize(pooled, dim=1, eps=1e-8)
    global_cosine = (pooled * target[:, :, 0, 0]).sum(dim=1).clamp(-1.0, 1.0)
    global_error = float((0.5 * torch.rad2deg(torch.acos(global_cosine))).mean())
    return {
        "window": window,
        "global_angle_mae_degrees": global_error,
        "confidence_weighted_pixel_mae_degrees": weighted_pixel_error,
        "mean_confidence": float(confidence.mean()),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script has no CPU fallback")
    blurry, sharp, target_direction, _ = fixed_evaluation_batch(args.size, args.seed + 100_000)
    records = [evaluate_window(blurry, sharp, target_direction, window) for window in args.windows]
    best = min(records, key=lambda item: float(item["global_angle_mae_degrees"]))
    report = {
        "experiment": "E2_pseudo_direction_gate",
        "seed": args.seed,
        "sample_count": len(blurry),
        "records": records,
        "selected_window": best["window"],
        "best_global_angle_mae_degrees": best["global_angle_mae_degrees"],
        "threshold_degrees": 10.0,
        "passed": float(best["global_angle_mae_degrees"]) < 10.0,
        "paper_result": False,
        "note": "Synthetic gate for a training-only pseudo label derived from paired sharp and blurry images.",
    }
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
