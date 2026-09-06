"""Analyze the matched C/S/G/D P12 factorial experiment."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = PROJECT_ROOT / "results" / "runs"
VARIANT_RUNS = {
    "C_uniform_scale_fixed_gate": "P12C_topk_nonlocal_v0",
    "S_adaptive_scale_fixed_gate": "P12S_scale_adaptive_fixed_gate_v0",
    "G_uniform_scale_adaptive_gate": "P12G_uniform_scale_adaptive_gate_v0",
    "D_adaptive_scale_adaptive_gate": "P12D_blur_aware_nonlocal_v0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute P12 2x2 factorial effects.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def load_run(directory: Path) -> tuple[dict[str, object], dict[str, dict[str, str]]]:
    summary_path = directory / "summary.json"
    per_image_path = directory / "per_image_metrics.csv"
    if not summary_path.is_file() or not per_image_path.is_file():
        raise FileNotFoundError(f"incomplete run directory: {directory}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with per_image_path.open("r", newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    by_name = {record["name"]: record for record in records}
    if len(by_name) != len(records):
        raise RuntimeError(f"duplicate image name in {per_image_path}")
    return summary, by_name


def percentile_interval(values: np.ndarray, seed: int, replicates: int) -> list[float] | None:
    if replicates == 0:
        return None
    generator = np.random.default_rng(seed)
    sample_count = values.shape[0]
    means = np.empty(replicates, dtype=np.float64)
    batch = 1000
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        indices = generator.integers(0, sample_count, size=(stop - start, sample_count))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return [float(low), float(high)]


def effects_for_names(
    names: list[str],
    records: dict[str, dict[str, dict[str, str]]],
) -> dict[str, object]:
    values = {
        variant: np.asarray(
            [float(records[variant][name]["candidate_psnr_db"]) for name in names],
            dtype=np.float64,
        )
        for variant in VARIANT_RUNS
    }
    c = values["C_uniform_scale_fixed_gate"]
    s = values["S_adaptive_scale_fixed_gate"]
    g = values["G_uniform_scale_adaptive_gate"]
    d = values["D_adaptive_scale_adaptive_gate"]
    effect_arrays = {
        "scale_effect_at_fixed_gate_db": s - c,
        "scale_effect_at_adaptive_gate_db": d - g,
        "scale_main_effect_db": 0.5 * ((s - c) + (d - g)),
        "gate_effect_at_uniform_scale_db": g - c,
        "gate_effect_at_adaptive_scale_db": d - s,
        "gate_main_effect_db": 0.5 * ((g - c) + (d - s)),
        "interaction_db": d - s - g + c,
        "D_minus_G_db": d - g,
        "D_minus_S_db": d - s,
        "D_minus_C_db": d - c,
    }
    return {
        "pairs": len(names),
        "psnr_db": {variant: float(array.mean()) for variant, array in values.items()},
        "effects": {name: float(array.mean()) for name, array in effect_arrays.items()},
        "positive_image_counts": {
            name: int((array > 0).sum()) for name, array in effect_arrays.items()
        },
        "effect_arrays": effect_arrays,
    }


def flatten(label: str, value: str, analysis: dict[str, object]) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {label: value, "pairs": int(analysis["pairs"])}
    psnr = analysis["psnr_db"]
    effects = analysis["effects"]
    assert isinstance(psnr, dict) and isinstance(effects, dict)
    row.update({f"psnr_{key}": float(metric) for key, metric in psnr.items()})
    row.update({key: float(metric) for key, metric in effects.items()})
    return row


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates < 0:
        raise ValueError("bootstrap-replicates must be nonnegative")
    runs_root = args.runs_root.expanduser().resolve()
    summaries: dict[str, dict[str, object]] = {}
    records: dict[str, dict[str, dict[str, str]]] = {}
    for variant, run_name in VARIANT_RUNS.items():
        summaries[variant], records[variant] = load_run(runs_root / run_name)

    reference_variant = "C_uniform_scale_fixed_gate"
    reference_names = set(records[reference_variant])
    for variant in VARIANT_RUNS:
        if set(records[variant]) != reference_names:
            raise RuntimeError(f"per-image name mismatch for {variant}")
        summary = summaries[variant]
        if summary.get("official_test_accessed") is not False:
            raise RuntimeError(f"official test audit failed for {variant}")
        if int(summary["seed"]) != args.seed:
            raise RuntimeError(f"seed mismatch for {variant}: {summary['seed']}")
        if int(summary["steps"]) != 3000 or int(summary["pairs"]) != 385:
            raise RuntimeError(f"protocol mismatch for {variant}")
        if int(summary["candidate_parameters"]) != int(
            summaries[reference_variant]["candidate_parameters"]
        ):
            raise RuntimeError(f"parameter-count mismatch for {variant}")

    ordered_names = sorted(reference_names)
    metadata = records[reference_variant]
    overall = effects_for_names(ordered_names, records)
    grouped_names: dict[str, list[str]] = defaultdict(list)
    strength_names: dict[str, list[str]] = defaultdict(list)
    for name in ordered_names:
        grouped_names[metadata[name]["recording_group"]].append(name)
        strength_names[metadata[name]["strength_class"]].append(name)
    per_group_analysis = {
        key: effects_for_names(names, records) for key, names in sorted(grouped_names.items())
    }
    per_strength_analysis = {
        key: effects_for_names(strength_names[key], records)
        for key in ("heavy", "medium", "light")
    }

    effect_arrays = overall.pop("effect_arrays")
    assert isinstance(effect_arrays, dict)
    bootstrap = {
        name: {
            "mean_db": float(array.mean()),
            "image_bootstrap_95_interval_db": percentile_interval(
                array,
                args.seed + index,
                args.bootstrap_replicates,
            ),
        }
        for index, (name, array) in enumerate(effect_arrays.items())
    }
    for analysis in [*per_group_analysis.values(), *per_strength_analysis.values()]:
        analysis.pop("effect_arrays")

    effects = overall["effects"]
    assert isinstance(effects, dict)
    main_threshold = 0.005
    conditional_floor = -0.005
    interaction_threshold = 0.01
    scale_supported = (
        float(effects["scale_main_effect_db"]) >= main_threshold
        and float(effects["scale_effect_at_fixed_gate_db"]) >= conditional_floor
        and float(effects["scale_effect_at_adaptive_gate_db"]) >= conditional_floor
    )
    gate_supported = (
        float(effects["gate_main_effect_db"]) >= main_threshold
        and float(effects["gate_effect_at_uniform_scale_db"]) >= conditional_floor
        and float(effects["gate_effect_at_adaptive_scale_db"]) >= conditional_floor
    )
    interaction_meaningful = abs(float(effects["interaction_db"])) >= interaction_threshold

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(
        output / "per_group_factorial.csv",
        [flatten("recording_group", name, analysis) for name, analysis in per_group_analysis.items()],
    )
    write_csv(
        output / "per_strength_factorial.csv",
        [
            flatten("strength_class", name, per_strength_analysis[name])
            for name in ("heavy", "medium", "light")
        ],
    )

    report = {
        "experiment": output.name,
        "stage": "P12 matched 2x2 factorial analysis",
        "factors": {
            "scale": ["uniform", "adaptive"],
            "fusion_gate": ["fixed_0.5", "adaptive"],
        },
        "variant_runs": VARIANT_RUNS,
        "seed": args.seed,
        "steps_per_variant": 3000,
        "pairs": len(ordered_names),
        "candidate_parameters": int(summaries[reference_variant]["candidate_parameters"]),
        "same_parameter_count": True,
        "overall": overall,
        "per_group": per_group_analysis,
        "per_strength": per_strength_analysis,
        "descriptive_image_bootstrap": bootstrap,
        "predeclared_gates": {
            "main_effect_threshold_db": main_threshold,
            "conditional_effect_floor_db": conditional_floor,
            "interaction_threshold_db": interaction_threshold,
            "scale_main_effect_supported": scale_supported,
            "gate_main_effect_supported": gate_supported,
            "interaction_meaningful": interaction_meaningful,
        },
        "official_test_accessed": False,
        "paper_result": False,
        "note": (
            "All effects use separately trained matched variants on the train-internal "
            "frozen development set. Image bootstrap intervals are descriptive because "
            "neighbouring video frames are correlated."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
