"""Paired P15 analysis against the matched local control with gate diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GoProPairDataset  # noqa: E402
from src.models import (  # noqa: E402
    CompactDeblurUNet,
    MultiScaleNonlocalBlock,
    teacher_blur_strength,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze P15 protected nonlocal gating.")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--local-run", default="P12A_local_control_v0")
    parser.add_argument("--candidate-run", default="P15_protected_nonlocal_seed3407_v0")
    parser.add_argument("--previous-candidate-run", default="P12G_uniform_scale_adaptive_gate_v0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_run(directory: Path) -> tuple[dict[str, object], dict[str, dict[str, str]]]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "per_image_metrics.csv").open(
        "r", newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    records = {row["name"]: row for row in rows}
    if len(records) != len(rows):
        raise RuntimeError(f"duplicate image names in {directory}")
    return summary, records


def audit_matched_runs(
    runs_root: Path,
    run_names: tuple[str, str, str],
    summaries: tuple[dict, dict, dict],
    records: tuple[dict, dict, dict],
    calibration: dict,
) -> dict[str, object]:
    """Reject mismatched seeds, evaluation rows, and training budgets before comparison."""
    checkpoints = [
        torch.load(runs_root / name / "best.pt", map_location="cpu", weights_only=False)
        for name in run_names
    ]
    settings = [dict(checkpoint["args"]) for checkpoint in checkpoints]
    expected_variants = ("local", "protected_gate_adaptive", "gate_adaptive")
    common = (
        "seed", "steps", "batch_size", "train_patch_size", "learning_rate",
        "frequency_weight", "val_interval", "num_workers", "train_pairs", "val_pairs",
    )
    for field in common:
        values = [setting[field] for setting in settings]
        if any(value != values[0] for value in values):
            raise RuntimeError(f"unmatched training setting {field}: {values}")
    if settings[0]["seed"] not in (3407, 7813) or settings[0]["steps"] != 3000:
        raise RuntimeError("unexpected seed or training budget")
    if settings[0]["batch_size"] != 6 or settings[0]["train_patch_size"] != 256:
        raise RuntimeError("unexpected batch size or patch size")
    for role, (setting, summary, checkpoint) in enumerate(zip(settings, summaries, checkpoints)):
        if setting["variant"] != expected_variants[role]:
            raise RuntimeError("unexpected comparison variant")
        if setting.get("sampling", "image_uniform") != "image_uniform":
            raise RuntimeError("P15 replication requires image-uniform sampling")
        if summary["seed"] != setting["seed"] or summary["steps"] != setting["steps"]:
            raise RuntimeError("summary/checkpoint settings disagree")
        if summary["best_checkpoint_step"] != checkpoint["step"]:
            raise RuntimeError("summary/checkpoint selection disagrees")
        if summary.get("official_test_accessed") is not False or summary["pairs"] != 385:
            raise RuntimeError("evaluation split audit failed")
        if summary["train_pairs"] != 1718 or summary["recording_group_overlap"]:
            raise RuntimeError("training split audit failed")
        if summary["exact_initialization_max_abs_error"] != 0:
            raise RuntimeError("baseline identity initialization audit failed")
        for field in ("baseline_checkpoint", "data_root"):
            if Path(setting[field]).resolve() != Path(settings[0][field]).resolve():
                raise RuntimeError(f"unmatched {field}")
        for field in ("validation_recording_groups", "train_recording_groups", "validation_patch_size"):
            if summary[field] != summaries[0][field]:
                raise RuntimeError(f"unmatched {field}")
    if settings[1].get("gate_supervision_weight") != 0.01:
        raise RuntimeError("P15 supervision weight differs from the fixed protocol")
    for field in ("scales", "top_k", "statistics_window"):
        if settings[1][field] != settings[2][field]:
            raise RuntimeError(f"unmatched nonlocal setting {field}")
    if calibration.get("official_test_accessed") is not False:
        raise RuntimeError("calibration split audit failed")
    for row_set in records:
        if len(row_set) != 385 or set(row_set) != set(records[0]):
            raise RuntimeError("image names or count disagree")
        for name, row in row_set.items():
            reference = records[0][name]
            for field in ("recording_group", "scene", "strength_class"):
                if row[field] != reference[field]:
                    raise RuntimeError(f"metadata mismatch: {name}/{field}")
            for field in ("input_psnr_db", "baseline_psnr_db", "baseline_ssim"):
                if abs(float(row[field]) - float(reference[field])) > 1e-6:
                    raise RuntimeError(f"baseline/evaluation mismatch: {name}/{field}")
            for field in ("candidate_psnr_db", "candidate_ssim"):
                if not np.isfinite(float(row[field])):
                    raise RuntimeError(f"nonfinite metric: {name}/{field}")
    for summary, row_set in zip(summaries, records):
        for field in ("candidate_psnr_db", "candidate_ssim"):
            actual = np.mean([float(row[field]) for row in row_set.values()])
            if abs(actual - float(summary[field])) > 1e-8:
                raise RuntimeError(f"summary/per-image mean mismatch: {field}")
    return {
        "passed": True,
        "seed": settings[0]["seed"],
        "matched_training_settings": {field: settings[0][field] for field in common},
        "sampling": "image_uniform",
        "same_b0_initialization": True,
        "note": "Both seeds fine-tune the same trained B0; these are not independent B0 retrainings.",
        "checkpoint_sha256": {
            name: hashlib.sha256((runs_root / name / "best.pt").read_bytes()).hexdigest()
            for name in run_names
        },
    }


def aggregate(names: list[str], rows: dict[str, dict[str, object]]) -> dict[str, object]:
    psnr = np.asarray([float(rows[name]["candidate_minus_local_psnr_db"]) for name in names])
    ssim = np.asarray([float(rows[name]["candidate_minus_local_ssim"]) for name in names])
    gates = np.asarray([float(rows[name]["intervention_gate_mean"]) for name in names])
    return {
        "pairs": len(names),
        "mean_candidate_minus_local_psnr_db": float(psnr.mean()),
        "median_candidate_minus_local_psnr_db": float(np.median(psnr)),
        "mean_candidate_minus_local_ssim": float(ssim.mean()),
        "positive_images": int((psnr > 0).sum()),
        "negative_images": int((psnr < 0).sum()),
        "mean_intervention_gate": float(gates.mean()),
        "std_intervention_gate": float(gates.std()),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this project intentionally has no CPU fallback")

    runs_root = args.runs_root.expanduser().resolve()
    local_summary, local = load_run(runs_root / args.local_run)
    candidate_summary, candidate = load_run(runs_root / args.candidate_run)
    previous_summary, previous = load_run(runs_root / args.previous_candidate_run)
    calibration = json.loads(
        args.calibration_summary.expanduser().resolve().read_text(encoding="utf-8")
    )
    protocol_audit = audit_matched_runs(
        runs_root,
        (args.local_run, args.candidate_run, args.previous_candidate_run),
        (local_summary, candidate_summary, previous_summary),
        (local, candidate, previous),
        calibration,
    )
    if set(local) != set(candidate) or set(local) != set(previous):
        raise RuntimeError("P12-A/P12-G/P15 image names are not exactly aligned")
    for summary in (local_summary, candidate_summary, previous_summary):
        if int(summary["pairs"]) != 385 or summary.get("official_test_accessed") is not False:
            raise RuntimeError("run protocol or official-test audit failed")
    if candidate_summary.get("variant") != "protected_gate_adaptive":
        raise ValueError("candidate run is not protected_gate_adaptive")

    baseline_checkpoint = torch.load(
        args.baseline_checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    baseline_args = dict(baseline_checkpoint["args"])
    checkpoint = torch.load(
        runs_root / args.candidate_run / "best.pt", map_location="cpu", weights_only=False
    )
    candidate_args = dict(checkpoint["args"])
    width = int(baseline_args["width"])
    block = MultiScaleNonlocalBlock(
        channels=width * 4,
        scales=tuple(int(value) for value in candidate_args["scales"]),
        top_k=int(candidate_args["top_k"]),
        statistics_window=int(candidate_args["statistics_window"]),
        mode="protected_gate_adaptive",
    )
    model = CompactDeblurUNet(width=width, bottleneck_module=block)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.cuda().eval()

    validation_groups = tuple(baseline_args["validation_groups"])
    dataset = GoProPairDataset(
        args.data_root,
        split="train",
        patch_size=int(baseline_args["val_patch_size"]),
        augment=False,
        include_recording_groups=validation_groups,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    gate_by_name: dict[str, tuple[float, float]] = {}
    with torch.inference_mode():
        for batch in loader:
            blurry = batch["blur"].cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                diagnostics = model.predict_geometry(blurry)
            gate = diagnostics["intervention_gate"].float()
            gate_by_name[str(batch["name"][0])] = (float(gate.mean()), float(gate.std()))
    if set(gate_by_name) != set(local):
        raise RuntimeError("gate-evaluation names are not exactly aligned")

    heavy_max = float(calibration["heavy_medium_threshold_input_psnr_db"])
    light_min = float(calibration["medium_light_threshold_input_psnr_db"])

    names = sorted(local)
    rows: dict[str, dict[str, object]] = {}
    for name in names:
        input_psnr = float(local[name]["input_psnr_db"])
        teacher = float(
            teacher_blur_strength(torch.tensor([input_psnr]), heavy_max, light_min)[0]
        )
        gate_mean, gate_std = gate_by_name[name]
        rows[name] = {
            "recording_group": local[name]["recording_group"],
            "scene": local[name]["scene"],
            "name": name,
            "strength_class": local[name]["strength_class"],
            "input_psnr_db": input_psnr,
            "local_psnr_db": float(local[name]["candidate_psnr_db"]),
            "candidate_psnr_db": float(candidate[name]["candidate_psnr_db"]),
            "candidate_minus_local_psnr_db": (
                float(candidate[name]["candidate_psnr_db"])
                - float(local[name]["candidate_psnr_db"])
            ),
            "candidate_minus_local_ssim": (
                float(candidate[name]["candidate_ssim"])
                - float(local[name]["candidate_ssim"])
            ),
            "candidate_minus_previous_g_psnr_db": (
                float(candidate[name]["candidate_psnr_db"])
                - float(previous[name]["candidate_psnr_db"])
            ),
            "teacher_strength": teacher,
            "intervention_gate_mean": gate_mean,
            "intervention_gate_std": gate_std,
        }

    groups: dict[str, list[str]] = defaultdict(list)
    strengths: dict[str, list[str]] = defaultdict(list)
    for name in names:
        groups[str(rows[name]["recording_group"])].append(name)
        strengths[str(rows[name]["strength_class"])].append(name)
    overall = aggregate(names, rows)
    per_group = {name: aggregate(items, rows) for name, items in sorted(groups.items())}
    per_strength = {
        name: aggregate(strengths[name], rows) for name in ("heavy", "medium", "light")
    }

    psnr_delta = np.asarray([float(rows[name]["candidate_minus_local_psnr_db"]) for name in names])
    previous_delta = np.asarray(
        [float(rows[name]["candidate_minus_previous_g_psnr_db"]) for name in names]
    )
    input_psnr = np.asarray([float(rows[name]["input_psnr_db"]) for name in names])
    teachers = np.asarray([float(rows[name]["teacher_strength"]) for name in names])
    gates = np.asarray([float(rows[name]["intervention_gate_mean"]) for name in names])
    generator = np.random.default_rng(1515)
    bootstrap = psnr_delta[
        generator.integers(0, len(psnr_delta), size=(5000, len(psnr_delta)))
    ].mean(axis=1)

    group_floor = min(
        float(value["mean_candidate_minus_local_psnr_db"]) for value in per_group.values()
    )
    strength_floor = min(
        float(value["mean_candidate_minus_local_psnr_db"])
        for value in per_strength.values()
    )
    passed = (
        float(overall["mean_candidate_minus_local_psnr_db"]) >= 0.02
        and float(overall["mean_candidate_minus_local_ssim"]) >= 0.0
        and group_floor >= -0.01
        and strength_floor >= -0.01
    )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_image_comparison.csv", [rows[name] for name in names])
    write_csv(
        output / "per_group_comparison.csv",
        [{"recording_group": name, **value} for name, value in per_group.items()],
    )
    write_csv(
        output / "per_strength_comparison.csv",
        [{"strength_class": name, **per_strength[name]} for name in ("heavy", "medium", "light")],
    )
    report = {
        "experiment": output.name,
        "stage": "P15 protected-gate paired analysis",
        "local_run": args.local_run,
        "candidate_run": args.candidate_run,
        "previous_candidate_run": args.previous_candidate_run,
        "pairs": len(names),
        "seed": protocol_audit["seed"],
        "protocol_audit": protocol_audit,
        "overall": overall,
        "mean_candidate_minus_previous_g_psnr_db": float(previous_delta.mean()),
        "per_group": per_group,
        "per_strength": per_strength,
        "gate_diagnostics": {
            "gate_vs_teacher_strength_pearson": float(np.corrcoef(gates, teachers)[0, 1]),
            "gate_vs_input_psnr_pearson": float(np.corrcoef(gates, input_psnr)[0, 1]),
            "mean_absolute_gate_teacher_error": float(np.abs(gates - teachers).mean()),
            "gate_min": float(gates.min()),
            "gate_max": float(gates.max()),
        },
        "descriptive_image_bootstrap_95_percent_interval_db": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "predeclared_gates": {
            "overall_threshold_db": 0.02,
            "ssim_floor": 0.0,
            "group_floor_db": -0.01,
            "strength_floor_db": -0.01,
            "minimum_group_delta_db": group_floor,
            "minimum_strength_delta_db": strength_floor,
            "passed": passed,
        },
        "official_test_accessed": False,
        "paper_result": False,
        "note": (
            "Image bootstrap is descriptive because adjacent video frames are correlated. "
            "A second training seed is required before promotion."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
