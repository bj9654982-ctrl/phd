"""P11: conservative three-way mixture for P5/P8/B0.

Only confident ends of the previously calibrated blur-strength gate activate a
specialized branch.  Ambiguous images fall back to B0.  No parameter or
threshold search is performed in this experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GoProPairDataset, recording_group  # noqa: E402
from src.models import (  # noqa: E402
    apply_anchored_residual_correction,
    apply_residual_gate,
    deployable_strength_gate,
)
from scripts.evaluate_p10_complementary_composition import (  # noqa: E402
    DEFAULT_RUNS,
    _load_models,
    _psnr,
    _set_determinism,
    _ssim,
    _strength_class,
)


METHOD_NAMES = ("baseline", "protected", "mixture_local", "mixture_trajectory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate uncertainty-aware P5/P8/B0 mixing.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "GOPRO_Large")
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=DEFAULT_RUNS / "P3_baseline_group_split_v0" / "best.pt",
    )
    parser.add_argument(
        "--scalar-checkpoint",
        type=Path,
        default=DEFAULT_RUNS / "P4_scalar_residual_gate_v1_patch256" / "best_gate.pt",
    )
    parser.add_argument(
        "--spatial-checkpoint",
        type=Path,
        default=DEFAULT_RUNS / "P5_spatial_residual_gate_v1_anchored" / "best_gate.pt",
    )
    parser.add_argument(
        "--local-checkpoint",
        type=Path,
        default=DEFAULT_RUNS / "P8A_e2e_local_control_v0" / "best.pt",
    )
    parser.add_argument(
        "--trajectory-checkpoint",
        type=Path,
        default=DEFAULT_RUNS / "P8B_e2e_trajectory_v0" / "best.pt",
    )
    parser.add_argument(
        "--calibration-summary",
        type=Path,
        default=DEFAULT_RUNS / "P7A_blur_strength_calibration_v1" / "summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUNS / "P11_uncertainty_mixture_v0",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def uncertainty_weights(strength_gate: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return light-specialist, B0, and heavy-specialist convex weights."""
    if strength_gate.ndim != 4 or strength_gate.shape[1:] != (1, 1, 1):
        raise ValueError("strength_gate must have shape (B, 1, 1, 1)")
    centered = 2.0 * strength_gate.float().clamp(0.0, 1.0) - 1.0
    heavy_weight = F.relu(centered).square()
    light_weight = F.relu(-centered).square()
    baseline_weight = 1.0 - heavy_weight - light_weight
    return light_weight, baseline_weight, heavy_weight


@torch.no_grad()
def evaluate(
    models: dict[str, torch.nn.Module],
    loader: DataLoader,
    light_cut_log_l1: float,
    heavy_cut_log_l1: float,
    heavy_max_psnr: float,
    light_min_psnr: float,
) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    for batch in tqdm(loader, desc="P11 uncertainty mixture"):
        blurry = batch["blur"].cuda(non_blocking=True)
        sharp = batch["sharp"].cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            baseline = models["baseline"](blurry)
            local = models["local"](blurry)
            trajectory = models["trajectory"](blurry)
        residual = baseline.float() - blurry.float()
        scalar_alpha = models["scalar"](blurry.float(), residual)
        correction = models["spatial"](blurry.float(), residual)
        spatial_alpha = apply_anchored_residual_correction(scalar_alpha, correction)
        protected = apply_residual_gate(blurry.float(), baseline.float(), spatial_alpha)
        strength_gate, _ = deployable_strength_gate(
            blurry,
            baseline,
            light_cut_log_l1,
            heavy_cut_log_l1,
        )
        light_weight, baseline_weight, heavy_weight = uncertainty_weights(strength_gate)
        mixture_local = (
            baseline_weight * baseline.float()
            + light_weight * protected
            + heavy_weight * local.float()
        )
        mixture_trajectory = (
            baseline_weight * baseline.float()
            + light_weight * protected
            + heavy_weight * trajectory.float()
        )
        outputs = {
            "baseline": baseline,
            "protected": protected,
            "mixture_local": mixture_local,
            "mixture_trajectory": mixture_trajectory,
        }
        for index, name in enumerate(batch["name"]):
            scene = str(name).split("/", maxsplit=1)[0]
            input_psnr = _psnr(blurry[index], sharp[index])
            record: dict[str, float | str] = {
                "recording_group": recording_group(scene),
                "name": str(name),
                "strength_class": _strength_class(input_psnr, heavy_max_psnr, light_min_psnr),
                "input_psnr_db": input_psnr,
                "strength_gate": float(strength_gate[index, 0, 0, 0]),
                "light_branch_weight": float(light_weight[index, 0, 0, 0]),
                "baseline_weight": float(baseline_weight[index, 0, 0, 0]),
                "heavy_branch_weight": float(heavy_weight[index, 0, 0, 0]),
            }
            for method, output in outputs.items():
                record[f"{method}_psnr_db"] = _psnr(output[index], sharp[index])
                record[f"{method}_ssim"] = _ssim(output[index], sharp[index])
            baseline_psnr = float(record["baseline_psnr_db"])
            baseline_ssim = float(record["baseline_ssim"])
            for method in METHOD_NAMES[1:]:
                record[f"{method}_minus_baseline_db"] = (
                    float(record[f"{method}_psnr_db"]) - baseline_psnr
                )
                record[f"{method}_minus_baseline_ssim"] = (
                    float(record[f"{method}_ssim"]) - baseline_ssim
                )
            record["trajectory_minus_local_mixture_db"] = (
                float(record["mixture_trajectory_psnr_db"])
                - float(record["mixture_local_psnr_db"])
            )
            records.append(record)
    return records


def _aggregate(records: list[dict[str, float | str]]) -> dict[str, float | int]:
    result: dict[str, float | int] = {"pairs": len(records)}
    for field in ("strength_gate", "light_branch_weight", "baseline_weight", "heavy_branch_weight"):
        result[field] = statistics.fmean(float(record[field]) for record in records)
    for method in METHOD_NAMES:
        result[f"{method}_psnr_db"] = statistics.fmean(
            float(record[f"{method}_psnr_db"]) for record in records
        )
        result[f"{method}_ssim"] = statistics.fmean(
            float(record[f"{method}_ssim"]) for record in records
        )
        if method != "baseline":
            result[f"{method}_minus_baseline_db"] = statistics.fmean(
                float(record[f"{method}_minus_baseline_db"]) for record in records
            )
            result[f"{method}_minus_baseline_ssim"] = statistics.fmean(
                float(record[f"{method}_minus_baseline_ssim"]) for record in records
            )
    result["trajectory_minus_local_mixture_db"] = statistics.fmean(
        float(record["trajectory_minus_local_mixture_db"]) for record in records
    )
    return result


def _stratify(
    records: list[dict[str, float | str]], field: str
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for record in records:
        grouped[str(record[field])].append(record)
    return {name: _aggregate(items) for name, items in sorted(grouped.items())}


def _write_csv(path: Path, records: list[dict[str, float | str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; P11 intentionally has no CPU fallback")
    if args.batch_size <= 0 or args.num_workers < 0 or args.max_pairs < 0:
        raise ValueError("invalid batch, worker, or max-pairs configuration")
    _set_determinism(args.seed)
    started = time.perf_counter()
    device = torch.device("cuda:0")
    models, metadata = _load_models(args, device)
    calibration_path = args.calibration_summary.expanduser().resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("official_test_accessed") is not False:
        raise ValueError("calibration summary does not preserve the official-test lock")
    gate_calibration = calibration["correction_gate_calibration"]
    development_groups = tuple(calibration["frozen_development_recording_groups"])
    dataset = GoProPairDataset(
        args.data_root,
        split="train",
        patch_size=int(metadata["baseline_args"]["val_patch_size"]),
        augment=False,
        max_pairs=args.max_pairs or None,
        include_recording_groups=development_groups,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    torch.cuda.reset_peak_memory_stats(device)
    records = evaluate(
        models,
        loader,
        float(gate_calibration["light_cut_log_l1"]),
        float(gate_calibration["heavy_cut_log_l1"]),
        float(calibration["heavy_medium_threshold_input_psnr_db"]),
        float(calibration["medium_light_threshold_input_psnr_db"]),
    )
    overall = _aggregate(records)
    per_group = _stratify(records, "recording_group")
    per_strength = _stratify(records, "strength_class")
    method = "mixture_trajectory"
    gain = float(overall[f"{method}_minus_baseline_db"])
    ssim_gain = float(overall[f"{method}_minus_baseline_ssim"])
    group_gains = {
        name: float(group[f"{method}_minus_baseline_db"])
        for name, group in per_group.items()
    }
    strength_gains = {
        name: float(group[f"{method}_minus_baseline_db"])
        for name, group in per_strength.items()
    }
    method_gates = {
        "psnr_gain_at_least_0_075_db": gain >= 0.075,
        "ssim_non_decreasing": ssim_gain >= 0.0,
        "minimum_group_at_least_minus_0_01_db": min(group_gains.values()) >= -0.01,
        "heavy_gain_at_least_0_05_db": strength_gains["heavy"] >= 0.05,
        "medium_gain_at_least_minus_0_01_db": strength_gains["medium"] >= -0.01,
        "light_gain_at_least_minus_0_005_db": strength_gains["light"] >= -0.005,
    }
    method_passed = all(method_gates.values())
    module_margin = float(overall["trajectory_minus_local_mixture_db"])
    module_margin_passed = module_margin >= 0.02
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": output.name,
        "stage": "P11 uncertainty-aware three-way residual mixture",
        "seed": args.seed,
        "checkpoint_paths": metadata["paths"],
        "checkpoint_steps": metadata["steps"],
        "calibration_summary": str(calibration_path),
        "pairs": len(records),
        "validation_recording_groups": development_groups,
        "formula": {
            "centered_strength": "z=2*g-1",
            "light_weight": "relu(-z)^2",
            "heavy_weight": "relu(z)^2",
            "baseline_weight": "1-light_weight-heavy_weight",
            "output": "w0*B0 + w_light*P5 + w_heavy*P8",
        },
        "parameter_or_threshold_search_performed": False,
        "new_trainable_parameters": 0,
        "overall": overall,
        "per_recording_group": per_group,
        "per_strength": per_strength,
        "method_decision_gates": method_gates,
        "method_passed": method_passed,
        "trajectory_minus_local_mixture_db": module_margin,
        "trajectory_module_margin_threshold_db": 0.02,
        "trajectory_module_margin_passed": module_margin_passed,
        "trajectory_claim_passed": method_passed and module_margin_passed,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "wall_seconds": time.perf_counter() - started,
        "uses_sharp_at_inference": False,
        "official_test_accessed": False,
        "paper_result": False,
        "note": (
            "Development-only no-search diagnostic. The added medium-blur safety criterion "
            "was declared after P10 and before this evaluation."
        ),
    }
    _write_csv(output / "per_image_metrics.csv", records)
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "overall": overall,
                "per_group_trajectory_gain": group_gains,
                "per_strength_trajectory_gain": strength_gains,
                "method_decision_gates": method_gates,
                "method_passed": method_passed,
                "trajectory_module_margin_passed": module_margin_passed,
                "trajectory_claim_passed": method_passed and module_margin_passed,
                "official_test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Summary: {output / 'summary.json'}")
    return 0 if method_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
