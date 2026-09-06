"""Paired two-seed stability analysis for local A versus candidate G."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "results" / "runs"
RUNS = {
    "seed3407": {
        "seed": 3407,
        "local": "P12A_local_control_v0",
        "candidate": "P12G_uniform_scale_adaptive_gate_v0",
    },
    "seed7813": {
        "seed": 7813,
        "local": "P13A_local_control_seed7813_v0",
        "candidate": "P13G_gate_adaptive_seed7813_v0",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the two-seed A/G replicate.")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_run(directory: Path) -> tuple[dict[str, object], dict[str, dict[str, str]]]:
    summary_path = directory / "summary.json"
    per_image_path = directory / "per_image_metrics.csv"
    if not summary_path.is_file() or not per_image_path.is_file():
        raise FileNotFoundError(f"incomplete run: {directory}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with per_image_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    records = {row["name"]: row for row in rows}
    if len(records) != len(rows):
        raise RuntimeError(f"duplicate image names: {per_image_path}")
    return summary, records


def aggregate(names: list[str], deltas: dict[str, np.ndarray], index: dict[str, int]) -> dict[str, object]:
    positions = np.asarray([index[name] for name in names], dtype=np.int64)
    seed_arrays = {seed: values[positions] for seed, values in deltas.items()}
    mean_array = np.stack(list(seed_arrays.values()), axis=0).mean(axis=0)
    return {
        "pairs": len(names),
        "per_seed": {
            seed: {
                "mean_candidate_minus_local_psnr_db": float(values.mean()),
                "median_candidate_minus_local_psnr_db": float(np.median(values)),
                "positive_images": int((values > 0).sum()),
                "negative_images": int((values < 0).sum()),
            }
            for seed, values in seed_arrays.items()
        },
        "two_seed_mean_candidate_minus_local_psnr_db": float(mean_array.mean()),
        "images_positive_in_both_seeds": int(
            np.logical_and(*(values > 0 for values in seed_arrays.values())).sum()
        ),
        "images_negative_in_both_seeds": int(
            np.logical_and(*(values < 0 for values in seed_arrays.values())).sum()
        ),
    }


def flatten(label: str, value: str, result: dict[str, object]) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        label: value,
        "pairs": int(result["pairs"]),
        "two_seed_mean_candidate_minus_local_psnr_db": float(
            result["two_seed_mean_candidate_minus_local_psnr_db"]
        ),
        "images_positive_in_both_seeds": int(result["images_positive_in_both_seeds"]),
        "images_negative_in_both_seeds": int(result["images_negative_in_both_seeds"]),
    }
    per_seed = result["per_seed"]
    assert isinstance(per_seed, dict)
    for seed_name, metrics in per_seed.items():
        assert isinstance(metrics, dict)
        for metric_name, metric in metrics.items():
            row[f"{seed_name}_{metric_name}"] = metric
    return row


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    runs_root = args.runs_root.expanduser().resolve()
    loaded: dict[str, dict[str, object]] = {}
    reference_names: set[str] | None = None
    for seed_name, config in RUNS.items():
        local_summary, local_records = load_run(runs_root / str(config["local"]))
        candidate_summary, candidate_records = load_run(runs_root / str(config["candidate"]))
        if set(local_records) != set(candidate_records):
            raise RuntimeError(f"local/candidate image mismatch for {seed_name}")
        if reference_names is None:
            reference_names = set(local_records)
        elif set(local_records) != reference_names:
            raise RuntimeError(f"cross-seed image mismatch for {seed_name}")
        for summary, role in ((local_summary, "local"), (candidate_summary, "candidate")):
            if int(summary["seed"]) != int(config["seed"]):
                raise RuntimeError(f"seed mismatch: {seed_name}/{role}")
            if int(summary["steps"]) != 3000 or int(summary["pairs"]) != 385:
                raise RuntimeError(f"protocol mismatch: {seed_name}/{role}")
            if summary.get("official_test_accessed") is not False:
                raise RuntimeError(f"official test audit failed: {seed_name}/{role}")
        loaded[seed_name] = {
            "local_summary": local_summary,
            "candidate_summary": candidate_summary,
            "local_records": local_records,
            "candidate_records": candidate_records,
        }

    assert reference_names is not None
    names = sorted(reference_names)
    index = {name: position for position, name in enumerate(names)}
    metadata = loaded["seed3407"]["local_records"]
    assert isinstance(metadata, dict)
    deltas: dict[str, np.ndarray] = {}
    ssim_deltas: dict[str, np.ndarray] = {}
    for seed_name, content in loaded.items():
        local_records = content["local_records"]
        candidate_records = content["candidate_records"]
        assert isinstance(local_records, dict) and isinstance(candidate_records, dict)
        deltas[seed_name] = np.asarray(
            [
                float(candidate_records[name]["candidate_psnr_db"])
                - float(local_records[name]["candidate_psnr_db"])
                for name in names
            ],
            dtype=np.float64,
        )
        ssim_deltas[seed_name] = np.asarray(
            [
                float(candidate_records[name]["candidate_ssim"])
                - float(local_records[name]["candidate_ssim"])
                for name in names
            ],
            dtype=np.float64,
        )

    grouped: dict[str, list[str]] = defaultdict(list)
    scenes: dict[str, list[str]] = defaultdict(list)
    strengths: dict[str, list[str]] = defaultdict(list)
    for name in names:
        row = metadata[name]
        grouped[row["recording_group"]].append(name)
        scenes[row["scene"]].append(name)
        strengths[row["strength_class"]].append(name)

    overall = aggregate(names, deltas, index)
    per_group = {key: aggregate(value, deltas, index) for key, value in sorted(grouped.items())}
    per_scene = {key: aggregate(value, deltas, index) for key, value in sorted(scenes.items())}
    per_strength = {
        key: aggregate(strengths[key], deltas, index) for key in ("heavy", "medium", "light")
    }
    input_psnr = np.asarray(
        [float(metadata[name]["input_psnr_db"]) for name in names], dtype=np.float64
    )
    mean_delta = np.stack(list(deltas.values()), axis=0).mean(axis=0)
    correlations = {
        "per_image_delta_cross_seed_pearson": float(
            np.corrcoef(deltas["seed3407"], deltas["seed7813"])[0, 1]
        ),
        "two_seed_mean_delta_vs_input_psnr_pearson": float(
            np.corrcoef(mean_delta, input_psnr)[0, 1]
        ),
    }

    overall_threshold = 0.02
    group_floor = -0.01
    per_seed_gates = {}
    for seed_name in RUNS:
        overall_delta = float(overall["per_seed"][seed_name]["mean_candidate_minus_local_psnr_db"])
        group_deltas = {
            group: float(result["per_seed"][seed_name]["mean_candidate_minus_local_psnr_db"])
            for group, result in per_group.items()
        }
        per_seed_gates[seed_name] = {
            "overall_delta_db": overall_delta,
            "minimum_group_delta_db": min(group_deltas.values()),
            "overall_gate_passed": overall_delta >= overall_threshold,
            "group_gate_passed": min(group_deltas.values()) >= group_floor,
            "replicate_gate_passed": (
                overall_delta >= overall_threshold and min(group_deltas.values()) >= group_floor
            ),
        }
    stable_replication_passed = all(
        bool(gate["replicate_gate_passed"]) for gate in per_seed_gates.values()
    )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(
        output / "per_group_seed_comparison.csv",
        [flatten("recording_group", name, result) for name, result in per_group.items()],
    )
    write_csv(
        output / "per_scene_seed_comparison.csv",
        [flatten("scene", name, result) for name, result in per_scene.items()],
    )
    write_csv(
        output / "per_strength_seed_comparison.csv",
        [
            flatten("strength_class", name, per_strength[name])
            for name in ("heavy", "medium", "light")
        ],
    )
    with (output / "per_image_seed_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fieldnames = [
            "recording_group",
            "scene",
            "name",
            "strength_class",
            "input_psnr_db",
            "seed3407_candidate_minus_local_psnr_db",
            "seed7813_candidate_minus_local_psnr_db",
            "two_seed_mean_candidate_minus_local_psnr_db",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for position, name in enumerate(names):
            writer.writerow(
                {
                    "recording_group": metadata[name]["recording_group"],
                    "scene": metadata[name]["scene"],
                    "name": name,
                    "strength_class": metadata[name]["strength_class"],
                    "input_psnr_db": input_psnr[position],
                    "seed3407_candidate_minus_local_psnr_db": deltas["seed3407"][position],
                    "seed7813_candidate_minus_local_psnr_db": deltas["seed7813"][position],
                    "two_seed_mean_candidate_minus_local_psnr_db": mean_delta[position],
                }
            )

    report = {
        "experiment": output.name,
        "stage": "P13 two-seed matched stability analysis",
        "run_protocol": RUNS,
        "pairs": len(names),
        "overall": overall,
        "overall_ssim_candidate_minus_local": {
            seed: float(values.mean()) for seed, values in ssim_deltas.items()
        },
        "per_group": per_group,
        "per_scene": per_scene,
        "per_strength": per_strength,
        "correlations": correlations,
        "predeclared_gates": {
            "overall_threshold_db": overall_threshold,
            "group_floor_db": group_floor,
            "per_seed": per_seed_gates,
            "stable_replication_passed": stable_replication_passed,
        },
        "official_test_accessed": False,
        "paper_result": False,
        "note": (
            "The overall candidate effect replicated, but the frozen recording-group "
            "protection rule remains binding. No seed or threshold was changed after results."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
