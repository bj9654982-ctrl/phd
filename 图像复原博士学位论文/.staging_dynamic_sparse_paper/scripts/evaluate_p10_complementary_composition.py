"""P10: zero-training composition of P5 protection and P8 expert residuals."""

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
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GoProPairDataset, recording_group  # noqa: E402
from src.models import (  # noqa: E402
    CompactDeblurUNet,
    ImageResidualGate,
    SpatialResidualCorrection,
    TrajectoryNonlocalBlock,
    apply_anchored_residual_correction,
    apply_residual_gate,
    compose_protected_expert,
    deployable_strength_gate,
)


DEFAULT_RUNS = PROJECT_ROOT / "results" / "runs"
METHOD_NAMES = (
    "baseline",
    "protected",
    "strength_local",
    "strength_trajectory",
    "combined_local",
    "combined_trajectory",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen P5 + P8 composition.")
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
        default=DEFAULT_RUNS / "P10_complementary_composition_v0",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _trajectory_model(checkpoint: dict[str, object], width: int) -> CompactDeblurUNet:
    oracle = dict(checkpoint["oracle_args"])
    training = dict(checkpoint["args"])
    block = TrajectoryNonlocalBlock(
        channels=width * 4,
        samples=int(oracle["trajectory_samples"]),
        max_feature_length=float(oracle["max_feature_length"]),
        fusion_mode=str(oracle["trajectory_fusion"]),
        angle_hypotheses=1,
        direction_residual_scale=float(training.get("direction_residual_scale", 2.0)),
    )
    model = CompactDeblurUNet(width=width, bottleneck_module=block)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def _psnr(image: torch.Tensor, target: torch.Tensor) -> float:
    mse = (image.float().clamp(0.0, 1.0) - target.float().clamp(0.0, 1.0)).square().mean()
    return float(-10.0 * mse.clamp_min(1e-12).log10())


def _ssim(image: torch.Tensor, target: torch.Tensor) -> float:
    image_array = image.float().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    target_array = target.float().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    return float(
        structural_similarity(
            target_array,
            image_array,
            data_range=1.0,
            channel_axis=2,
        )
    )


def _strength_class(input_psnr: float, heavy_max: float, light_min: float) -> str:
    if input_psnr <= heavy_max:
        return "heavy"
    if input_psnr >= light_min:
        return "light"
    return "medium"


def _load_models(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, object]]:
    paths = {
        "baseline": args.baseline_checkpoint.expanduser().resolve(),
        "scalar": args.scalar_checkpoint.expanduser().resolve(),
        "spatial": args.spatial_checkpoint.expanduser().resolve(),
        "local": args.local_checkpoint.expanduser().resolve(),
        "trajectory": args.trajectory_checkpoint.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")
    checkpoints = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    baseline_args = dict(checkpoints["baseline"]["args"])
    width = int(baseline_args["width"])
    baseline = CompactDeblurUNet(width=width)
    baseline.load_state_dict(checkpoints["baseline"]["model"], strict=True)
    local = CompactDeblurUNet(width=width)
    local.load_state_dict(checkpoints["local"]["model"], strict=True)
    trajectory = _trajectory_model(checkpoints["trajectory"], width)

    scalar_args = dict(checkpoints["scalar"]["args"])
    scalar = ImageResidualGate(
        width=int(scalar_args["gate_width"]),
        initial_alpha=float(scalar_args["initial_alpha"]),
    )
    scalar.load_state_dict(checkpoints["scalar"]["gate"], strict=True)
    spatial_args = dict(checkpoints["spatial"]["args"])
    if spatial_args.get("spatial_mode") != "anchored":
        raise ValueError("P10 requires the anchored P5 spatial correction")
    spatial = SpatialResidualCorrection(
        width=int(spatial_args["gate_width"]),
        max_correction=float(spatial_args["max_correction"]),
    )
    spatial.load_state_dict(checkpoints["spatial"]["gate"], strict=True)

    models: dict[str, torch.nn.Module] = {
        "baseline": baseline,
        "local": local,
        "trajectory": trajectory,
        "scalar": scalar,
        "spatial": spatial,
    }
    for model in models.values():
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    metadata = {
        "paths": {name: str(path) for name, path in paths.items()},
        "steps": {name: int(checkpoints[name]["step"]) for name in checkpoints},
        "baseline_args": baseline_args,
    }
    return models, metadata


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
    for batch in tqdm(loader, desc="P10 frozen composition"):
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
        strength_local = baseline.float() + strength_gate * (local.float() - baseline.float())
        strength_trajectory = baseline.float() + strength_gate * (
            trajectory.float() - baseline.float()
        )
        combined_local = compose_protected_expert(
            blurry.float(), baseline.float(), local.float(), spatial_alpha, strength_gate
        )
        combined_trajectory = compose_protected_expert(
            blurry.float(), baseline.float(), trajectory.float(), spatial_alpha, strength_gate
        )
        outputs = {
            "baseline": baseline,
            "protected": protected,
            "strength_local": strength_local,
            "strength_trajectory": strength_trajectory,
            "combined_local": combined_local,
            "combined_trajectory": combined_trajectory,
        }
        for index, name in enumerate(batch["name"]):
            scene = str(name).split("/", maxsplit=1)[0]
            input_psnr = _psnr(blurry[index], sharp[index])
            record: dict[str, float | str] = {
                "recording_group": recording_group(scene),
                "name": str(name),
                "strength_class": _strength_class(input_psnr, heavy_max_psnr, light_min_psnr),
                "input_psnr_db": input_psnr,
                "spatial_alpha_mean": float(spatial_alpha[index].mean()),
                "strength_gate": float(strength_gate[index, 0, 0, 0]),
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
            record["trajectory_minus_local_combined_db"] = (
                float(record["combined_trajectory_psnr_db"])
                - float(record["combined_local_psnr_db"])
            )
            records.append(record)
    return records


def _aggregate(records: list[dict[str, float | str]]) -> dict[str, object]:
    result: dict[str, object] = {"pairs": len(records)}
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
    result["trajectory_minus_local_combined_db"] = statistics.fmean(
        float(record["trajectory_minus_local_combined_db"]) for record in records
    )
    return result


def _stratify(
    records: list[dict[str, float | str]],
    field: str,
) -> dict[str, dict[str, object]]:
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
        raise RuntimeError("CUDA is required; P10 intentionally has no CPU fallback")
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
    development_groups = tuple(calibration["frozen_development_recording_groups"])
    gate_calibration = calibration["correction_gate_calibration"]
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

    gain = float(overall["combined_trajectory_minus_baseline_db"])
    ssim_gain = float(overall["combined_trajectory_minus_baseline_ssim"])
    minimum_group = min(
        float(group["combined_trajectory_minus_baseline_db"])
        for group in per_group.values()
    )
    heavy_gain = float(per_strength["heavy"]["combined_trajectory_minus_baseline_db"])
    light_gain = float(per_strength["light"]["combined_trajectory_minus_baseline_db"])
    module_margin = float(overall["trajectory_minus_local_combined_db"])
    method_gates = {
        "psnr_gain_at_least_0_075_db": gain >= 0.075,
        "ssim_non_decreasing": ssim_gain >= 0.0,
        "minimum_group_at_least_minus_0_01_db": minimum_group >= -0.01,
        "heavy_gain_at_least_0_05_db": heavy_gain >= 0.05,
        "light_gain_at_least_minus_0_005_db": light_gain >= -0.005,
    }
    method_passed = all(method_gates.values())
    trajectory_margin_passed = module_margin >= 0.02
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": output.name,
        "stage": "P10 zero-training complementary residual composition",
        "seed": args.seed,
        "checkpoint_paths": metadata["paths"],
        "checkpoint_steps": metadata["steps"],
        "calibration_summary": str(calibration_path),
        "pairs": len(records),
        "validation_recording_groups": development_groups,
        "formula": (
            "blur + spatial_alpha*(B0-blur) + strength_gate*(expert-B0)"
        ),
        "new_trainable_parameters": 0,
        "overall": overall,
        "per_recording_group": per_group,
        "per_strength": per_strength,
        "minimum_group_combined_trajectory_minus_baseline_db": minimum_group,
        "heavy_combined_trajectory_minus_baseline_db": heavy_gain,
        "light_combined_trajectory_minus_baseline_db": light_gain,
        "trajectory_minus_local_combined_db": module_margin,
        "method_decision_gates": method_gates,
        "method_passed": method_passed,
        "trajectory_module_margin_threshold_db": 0.02,
        "trajectory_module_margin_passed": trajectory_margin_passed,
        "trajectory_claim_passed": method_passed and trajectory_margin_passed,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "wall_seconds": time.perf_counter() - started,
        "uses_sharp_at_inference": False,
        "official_test_accessed": False,
        "paper_result": False,
        "note": (
            "Development-only composition of already frozen modules. A method pass does not "
            "establish a trajectory-specific contribution; that requires the matched local margin."
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
                "per_group_trajectory_gain": {
                    name: group["combined_trajectory_minus_baseline_db"]
                    for name, group in per_group.items()
                },
                "per_strength_trajectory_gain": {
                    name: group["combined_trajectory_minus_baseline_db"]
                    for name, group in per_strength.items()
                },
                "method_passed": method_passed,
                "trajectory_module_margin_passed": trajectory_margin_passed,
                "trajectory_claim_passed": method_passed and trajectory_margin_passed,
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
