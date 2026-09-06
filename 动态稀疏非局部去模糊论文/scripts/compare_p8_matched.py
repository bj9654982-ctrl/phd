"""P8: paired comparison of the local and trajectory end-to-end screens.

The two runs must contain the same images and the same frozen-B0 predictions.
This script reports the trajectory module's marginal contribution over the
matched local control and the unattainable-at-inference endpoint oracle used
only to decide whether better routing still has enough headroom.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = PROJECT_ROOT / "results" / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the matched P8-A/P8-B screens.")
    parser.add_argument(
        "--local-run",
        type=Path,
        default=DEFAULT_RUNS / "P8A_e2e_local_control_v0",
    )
    parser.add_argument(
        "--trajectory-run",
        type=Path,
        default=DEFAULT_RUNS / "P8B_e2e_trajectory_v0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUNS / "P8_matched_comparison_v0",
    )
    parser.add_argument("--method-gain-threshold-db", type=float, default=0.075)
    parser.add_argument("--module-margin-threshold-db", type=float, default=0.02)
    return parser.parse_args()


def _load_rows(run: Path) -> list[dict[str, str]]:
    path = run / "per_image_metrics.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"no per-image rows in {path}")
    return rows


def _best_step(run: Path) -> int:
    with (run / "metrics.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return int(max(rows, key=lambda row: float(row["final_psnr_db"]))["step"])


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _paired_stats(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    mean = _mean(values)
    std = float(statistics.stdev(values)) if count > 1 else 0.0
    sem = std / math.sqrt(count) if count else float("nan")
    # This image-level interval is descriptive only: adjacent video frames are
    # correlated, so it must not be presented as an independent-sample claim.
    return {
        "pairs": count,
        "mean_db": mean,
        "median_db": float(statistics.median(values)),
        "standard_deviation_db": std,
        "descriptive_image_level_95pct_ci_db": [mean - 1.96 * sem, mean + 1.96 * sem],
        "positive_pairs": sum(value > 0.0 for value in values),
        "negative_pairs": sum(value < 0.0 for value in values),
        "ties": sum(value == 0.0 for value in values),
    }


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1)
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = _mean(left)
    right_mean = _mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered, strict=True))
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    return float(numerator / denominator) if denominator > 0.0 else float("nan")


def _oracle_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    improvements = [
        max(0.0, float(row["dynamic_psnr_db"]) - float(row["baseline_psnr_db"]))
        for row in rows
    ]
    by_strength: dict[str, list[float]] = defaultdict(list)
    for row, improvement in zip(rows, improvements, strict=True):
        by_strength[row["strength_class"]].append(improvement)
    return {
        "endpoint_oracle_gain_db": _mean(improvements),
        "dynamic_endpoint_positive_pairs": sum(
            float(row["dynamic_minus_baseline_db"]) > 0.0 for row in rows
        ),
        "per_strength_endpoint_oracle_gain_db": {
            name: _mean(values) for name, values in sorted(by_strength.items())
        },
    }


def _aggregate_by(
    local_rows: list[dict[str, str]],
    trajectory_rows: list[dict[str, str]],
    field: str,
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for local, trajectory in zip(local_rows, trajectory_rows, strict=True):
        grouped[local[field]].append(
            float(trajectory["final_psnr_db"]) - float(local["final_psnr_db"])
        )
    return {name: _paired_stats(values) for name, values in sorted(grouped.items())}


def main() -> None:
    args = parse_args()
    local_rows = _load_rows(args.local_run)
    trajectory_rows = _load_rows(args.trajectory_run)
    if len(local_rows) != len(trajectory_rows):
        raise RuntimeError("P8-A and P8-B contain different numbers of images")

    for index, (local, trajectory) in enumerate(
        zip(local_rows, trajectory_rows, strict=True)
    ):
        identity = ("recording_group", "scene", "name", "strength_class")
        if any(local[key] != trajectory[key] for key in identity):
            raise RuntimeError(f"image identity mismatch at row {index}")
        baseline_difference = abs(
            float(local["baseline_psnr_db"]) - float(trajectory["baseline_psnr_db"])
        )
        if baseline_difference > 1e-8:
            raise RuntimeError(f"baseline mismatch at row {index}: {baseline_difference}")

    final_differences = [
        float(trajectory["final_psnr_db"]) - float(local["final_psnr_db"])
        for local, trajectory in zip(local_rows, trajectory_rows, strict=True)
    ]
    endpoint_differences = [
        float(trajectory["dynamic_psnr_db"]) - float(local["dynamic_psnr_db"])
        for local, trajectory in zip(local_rows, trajectory_rows, strict=True)
    ]
    trajectory_benefit = [float(row["dynamic_minus_baseline_db"]) for row in trajectory_rows]
    deployable_gate = [float(row["gate"]) for row in trajectory_rows]

    local_summary = json.loads((args.local_run / "summary.json").read_text(encoding="utf-8"))
    trajectory_summary = json.loads(
        (args.trajectory_run / "summary.json").read_text(encoding="utf-8")
    )
    local_best = args.local_run / "best.pt"
    trajectory_best = args.trajectory_run / "best.pt"
    # Checkpoint steps are also present in the training CSV as the selected
    # maximum; keep them explicit here to avoid importing torch for analysis.
    local_best_step = _best_step(args.local_run)
    trajectory_best_step = _best_step(args.trajectory_run)

    local_oracle = _oracle_summary(local_rows)
    trajectory_oracle = _oracle_summary(trajectory_rows)
    module_margin = _mean(final_differences)
    method_gain = float(trajectory_summary["final_minus_baseline_db"])
    report = {
        "protocol": {
            "paired_images": len(local_rows),
            "same_image_order_verified": True,
            "same_baseline_predictions_verified": True,
            "official_test_accessed": False,
            "image_level_ci_is_descriptive_due_to_video_frame_correlation": True,
        },
        "runs": {
            "local_control": str(args.local_run.resolve()),
            "trajectory": str(args.trajectory_run.resolve()),
            "local_best_checkpoint": str(local_best.resolve()),
            "trajectory_best_checkpoint": str(trajectory_best.resolve()),
            "local_best_step": local_best_step,
            "trajectory_best_step": trajectory_best_step,
        },
        "baseline_psnr_db": float(local_summary["baseline_psnr_db"]),
        "local_control_final_psnr_db": float(local_summary["final_psnr_db"]),
        "trajectory_final_psnr_db": float(trajectory_summary["final_psnr_db"]),
        "local_control_gain_over_baseline_db": float(local_summary["final_minus_baseline_db"]),
        "trajectory_gain_over_baseline_db": method_gain,
        "trajectory_minus_local_control": _paired_stats(final_differences),
        "trajectory_endpoint_minus_local_endpoint": _paired_stats(endpoint_differences),
        "trajectory_minus_local_by_recording_group": _aggregate_by(
            local_rows, trajectory_rows, "recording_group"
        ),
        "trajectory_minus_local_by_strength": _aggregate_by(
            local_rows, trajectory_rows, "strength_class"
        ),
        "local_control_oracle": local_oracle,
        "trajectory_oracle": trajectory_oracle,
        "trajectory_gate_vs_endpoint_benefit": {
            "pearson": _pearson(deployable_gate, trajectory_benefit),
            "spearman": _pearson(_rank(deployable_gate), _rank(trajectory_benefit)),
        },
        "decision_gates": {
            "method_gain_threshold_db": args.method_gain_threshold_db,
            "module_margin_threshold_db": args.module_margin_threshold_db,
            "method_gain_passed": method_gain >= args.method_gain_threshold_db,
            "trajectory_module_margin_passed": module_margin >= args.module_margin_threshold_db,
            "trajectory_endpoint_oracle_has_method_headroom": (
                float(trajectory_oracle["endpoint_oracle_gain_db"])
                >= args.method_gain_threshold_db
            ),
        },
        "decision": (
            "Do not scale the present trajectory block: its matched gain over the local "
            "control is below the predeclared module margin. Continue only with a "
            "training-only benefit-routing diagnostic if the endpoint oracle retains "
            "the required method headroom."
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "summary.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Summary: {output}")


if __name__ == "__main__":
    main()
