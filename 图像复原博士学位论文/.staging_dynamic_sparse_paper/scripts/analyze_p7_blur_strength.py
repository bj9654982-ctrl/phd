"""P7-A: calibrate deployable blurry-only blur-strength proxies.

The paired blurry/sharp PSNR is used only as a training-time teacher.  Every
candidate proxy can be computed from the blurry input (and the frozen B0
prediction), so the selected signal remains available at inference time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GoProPairDataset, recording_group  # noqa: E402
from src.models import CompactDeblurUNet  # noqa: E402


DEFAULT_BASELINE_RUN = PROJECT_ROOT / "results" / "runs" / "P3_baseline_group_split_v0"
POSITIVE_FEATURES = (
    "gradient_rms",
    "laplacian_rms",
    "relative_high_frequency",
    "baseline_correction_rms",
    "baseline_correction_l1",
    "correction_to_gradient",
)
FEATURE_NAMES = (*POSITIVE_FEATURES, "structure_anisotropy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze blurry-only blur-strength proxies on the B0 training partition."
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "GOPRO_Large")
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=DEFAULT_BASELINE_RUN / "best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "P7A_blur_strength_calibration_v0",
    )
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def _psnr_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction.float() - target.float()).square().flatten(1).mean(1).clamp_min(1e-12)
    return -10.0 * mse.log10()


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic handling of ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or float(left.std()) < 1e-12 or float(right.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _correlation(_rank(left), _rank(right))


def _blurry_only_metrics(blurry: torch.Tensor, baseline_output: torch.Tensor) -> dict[str, torch.Tensor]:
    blurry = blurry.float()
    baseline_output = baseline_output.float()
    gray = (
        0.2989 * blurry[:, 0:1]
        + 0.5870 * blurry[:, 1:2]
        + 0.1140 * blurry[:, 2:3]
    )
    gradient_x = F.pad(gray[:, :, :, 2:] - gray[:, :, :, :-2], (1, 1, 0, 0), mode="replicate")
    gradient_y = F.pad(gray[:, :, 2:, :] - gray[:, :, :-2, :], (0, 0, 1, 1), mode="replicate")
    gradient_power = (gradient_x.square() + gradient_y.square()).flatten(1).mean(1)

    laplacian = F.conv2d(
        gray,
        gray.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])[
            None, None
        ],
        padding=1,
    )
    laplacian_power = laplacian.square().flatten(1).mean(1)

    jxx = F.avg_pool2d(gradient_x.square(), 31, stride=1, padding=15)
    jyy = F.avg_pool2d(gradient_y.square(), 31, stride=1, padding=15)
    jxy = F.avg_pool2d(gradient_x * gradient_y, 31, stride=1, padding=15)
    anisotropy = (
        torch.sqrt((jxx - jyy).square() + 4.0 * jxy.square() + 1e-12)
        / (jxx + jyy + 1e-8)
    ).flatten(1).mean(1)

    correction = baseline_output - blurry
    correction_rms = correction.square().flatten(1).mean(1).sqrt()
    correction_l1 = correction.abs().flatten(1).mean(1)
    gradient_rms = gradient_power.sqrt()
    laplacian_rms = laplacian_power.sqrt()
    return {
        "gradient_rms": gradient_rms,
        "laplacian_rms": laplacian_rms,
        "relative_high_frequency": laplacian_rms / (gradient_rms + 1e-8),
        "structure_anisotropy": anisotropy,
        "baseline_correction_rms": correction_rms,
        "baseline_correction_l1": correction_l1,
        "correction_to_gradient": correction_rms / (gradient_rms + 1e-8),
    }


def _design_matrix(records: list[dict[str, float | str]]) -> np.ndarray:
    columns = []
    for name in FEATURE_NAMES:
        values = np.asarray([float(record[name]) for record in records], dtype=np.float64)
        if name in POSITIVE_FEATURES:
            values = np.log(values.clip(min=1e-12))
        columns.append(values)
    return np.column_stack(columns)


def _fit_ridge(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = fit_x.mean(axis=0)
    scale = fit_x.std(axis=0).clip(min=1e-8)
    standardized = (fit_x - mean) / scale
    augmented = np.column_stack((np.ones(len(standardized)), standardized))
    penalty = np.eye(augmented.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ fit_y)
    return coefficients, mean, scale


def _predict_ridge(
    features: np.ndarray,
    coefficients: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    standardized = (features - mean) / scale
    return np.column_stack((np.ones(len(standardized)), standardized)) @ coefficients


def _severity_label(input_psnr: np.ndarray, lower: float, upper: float) -> np.ndarray:
    # 0=heavy blur, 1=medium blur, 2=light blur.
    return np.where(input_psnr <= lower, 0, np.where(input_psnr >= upper, 2, 1))


def _classification_report(target: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    names = ("heavy", "medium", "light")
    confusion = np.zeros((3, 3), dtype=np.int64)
    for actual, estimate in zip(target, predicted, strict=True):
        confusion[int(actual), int(estimate)] += 1
    recalls = []
    for index in range(3):
        denominator = int(confusion[index].sum())
        recalls.append(float(confusion[index, index] / denominator) if denominator else float("nan"))
    return {
        "accuracy": float(np.mean(target == predicted)),
        "balanced_accuracy": float(np.nanmean(recalls)),
        "recall": {name: recalls[index] for index, name in enumerate(names)},
        "confusion_rows_actual_columns_predicted": confusion.tolist(),
    }


def _fit_monotonic_gate_thresholds(
    proxy: np.ndarray,
    target_class: np.ndarray,
) -> tuple[float, float, dict[str, object]]:
    """Fit light/heavy cuts using only the inner-fit partition.

    Larger proxy values mean stronger blur.  The search optimizes three-class
    balanced accuracy, not raw accuracy, so the majority class cannot dominate.
    """
    candidates = np.unique(np.quantile(proxy, np.linspace(0.03, 0.97, 95)))
    best: tuple[float, float, float, float] | None = None
    best_report: dict[str, object] | None = None
    for light_cut in candidates[:-1]:
        for heavy_cut in candidates[1:]:
            if heavy_cut <= light_cut:
                continue
            predicted = np.where(proxy >= heavy_cut, 0, np.where(proxy <= light_cut, 2, 1))
            report = _classification_report(target_class, predicted)
            score = (
                float(report["balanced_accuracy"]),
                float(report["accuracy"]),
                float(heavy_cut - light_cut),
                -float(light_cut),
            )
            if best is None or score > best:
                best = score
                best_report = report
                best_light = float(light_cut)
                best_heavy = float(heavy_cut)
    if best_report is None:
        raise RuntimeError("could not fit monotonic blur-strength thresholds")
    return best_light, best_heavy, best_report


def _soft_gate(proxy: np.ndarray, light_cut: float, heavy_cut: float) -> np.ndarray:
    linear = np.clip((proxy - light_cut) / (heavy_cut - light_cut), 0.0, 1.0)
    return np.square(linear) * (3.0 - 2.0 * linear)


def _write_csv(path: Path, records: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; P7-A intentionally has no CPU fallback")
    if min(args.patch_size, args.batch_size) <= 0:
        raise ValueError("patch-size and batch-size must be positive")
    if args.max_pairs < 0 or args.num_workers < 0:
        raise ValueError("max-pairs and num-workers must be nonnegative")
    if not 0.0 < args.calibration_fraction < 0.5:
        raise ValueError("calibration-fraction must be between 0 and 0.5")
    if args.ridge < 0:
        raise ValueError("ridge must be nonnegative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")

    baseline_path = args.baseline_checkpoint.expanduser().resolve()
    if not baseline_path.is_file():
        raise FileNotFoundError(f"baseline checkpoint not found: {baseline_path}")
    checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=False)
    baseline_args = checkpoint.get("args", {})
    if baseline_args.get("variant") != "local":
        raise ValueError("baseline checkpoint must use variant=local")
    if baseline_args.get("validation_source") != "train_holdout":
        raise ValueError("baseline checkpoint must use validation_source=train_holdout")
    frozen_groups = tuple(baseline_args.get("validation_groups") or ())
    if not frozen_groups:
        raise ValueError("baseline checkpoint does not contain frozen validation groups")

    dataset = GoProPairDataset(
        args.data_root,
        split="train",
        patch_size=args.patch_size,
        augment=False,
        max_pairs=args.max_pairs or None,
        exclude_recording_groups=frozen_groups,
    )
    if set(dataset.recording_groups) & set(frozen_groups):
        raise RuntimeError("frozen development groups leaked into P7-A")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    baseline = CompactDeblurUNet(width=int(baseline_args["width"]))
    baseline.load_state_dict(checkpoint["model"], strict=True)
    baseline = baseline.to(device).eval()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)

    records: list[dict[str, float | str]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="P7-A blur-strength scan"):
            blurry = batch["blur"].to(device, non_blocking=True)
            sharp = batch["sharp"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                baseline_output = baseline(blurry)
            metrics = _blurry_only_metrics(blurry, baseline_output)
            input_psnr = _psnr_per_image(blurry, sharp)
            baseline_psnr = _psnr_per_image(baseline_output, sharp)
            for index, name in enumerate(batch["name"]):
                scene = str(name).split("/", maxsplit=1)[0]
                record: dict[str, float | str] = {
                    "recording_group": recording_group(scene),
                    "scene": scene,
                    "name": str(name),
                    "input_psnr_db": float(input_psnr[index]),
                    "baseline_psnr_db": float(baseline_psnr[index]),
                }
                record.update({key: float(value[index]) for key, value in metrics.items()})
                records.append(record)

    all_groups = sorted({str(record["recording_group"]) for record in records})
    shuffled_groups = all_groups.copy()
    random.Random(args.seed).shuffle(shuffled_groups)
    calibration_count = max(2, int(math.ceil(len(all_groups) * args.calibration_fraction)))
    calibration_groups = tuple(sorted(shuffled_groups[:calibration_count]))
    fit_groups = tuple(sorted(set(all_groups) - set(calibration_groups)))
    fit_records = [record for record in records if record["recording_group"] in fit_groups]
    calibration_records = [
        record for record in records if record["recording_group"] in calibration_groups
    ]
    if not fit_records or not calibration_records:
        raise RuntimeError("inner group split produced an empty partition")

    all_x = _design_matrix(records)
    fit_mask = np.asarray([record["recording_group"] in fit_groups for record in records])
    calibration_mask = ~fit_mask
    fit_x = all_x[fit_mask]
    calibration_x = all_x[calibration_mask]
    fit_input_psnr = np.asarray(
        [float(record["input_psnr_db"]) for record in fit_records], dtype=np.float64
    )
    calibration_input_psnr = np.asarray(
        [float(record["input_psnr_db"]) for record in calibration_records], dtype=np.float64
    )
    coefficients, feature_mean, feature_scale = _fit_ridge(
        fit_x,
        fit_input_psnr,
        args.ridge,
    )
    predicted_fit_psnr = _predict_ridge(fit_x, coefficients, feature_mean, feature_scale)
    predicted_calibration_psnr = _predict_ridge(
        calibration_x,
        coefficients,
        feature_mean,
        feature_scale,
    )
    lower_threshold, upper_threshold = np.quantile(fit_input_psnr, (1.0 / 3.0, 2.0 / 3.0))
    fit_target_class = _severity_label(fit_input_psnr, lower_threshold, upper_threshold)
    calibration_target_class = _severity_label(
        calibration_input_psnr,
        lower_threshold,
        upper_threshold,
    )
    fit_predicted_class = _severity_label(predicted_fit_psnr, lower_threshold, upper_threshold)
    calibration_predicted_class = _severity_label(
        predicted_calibration_psnr,
        lower_threshold,
        upper_threshold,
    )

    correction_index = FEATURE_NAMES.index("baseline_correction_l1")
    fit_correction = fit_x[:, correction_index]
    calibration_correction = calibration_x[:, correction_index]
    gate_light_cut, gate_heavy_cut, gate_fit_classification = _fit_monotonic_gate_thresholds(
        fit_correction,
        fit_target_class,
    )
    gate_calibration_class = np.where(
        calibration_correction >= gate_heavy_cut,
        0,
        np.where(calibration_correction <= gate_light_cut, 2, 1),
    )
    fit_gate = _soft_gate(fit_correction, gate_light_cut, gate_heavy_cut)
    calibration_gate = _soft_gate(
        calibration_correction,
        gate_light_cut,
        gate_heavy_cut,
    )

    def gate_means(gate: np.ndarray, target: np.ndarray) -> dict[str, float]:
        names = ("heavy", "medium", "light")
        return {
            name: float(gate[target == index].mean()) if np.any(target == index) else float("nan")
            for index, name in enumerate(names)
        }

    proxy_reports: dict[str, dict[str, float]] = {}
    for feature_index, name in enumerate(FEATURE_NAMES):
        fit_values = fit_x[:, feature_index]
        calibration_values = calibration_x[:, feature_index]
        slope, intercept = np.polyfit(fit_values, fit_input_psnr, 1)
        univariate_prediction = slope * calibration_values + intercept
        proxy_reports[name] = {
            "fit_pearson_with_input_psnr": _correlation(fit_values, fit_input_psnr),
            "fit_spearman_with_input_psnr": _spearman(fit_values, fit_input_psnr),
            "calibration_pearson_with_input_psnr": _correlation(
                calibration_values, calibration_input_psnr
            ),
            "calibration_spearman_with_input_psnr": _spearman(
                calibration_values, calibration_input_psnr
            ),
            "calibration_univariate_mae_db": float(
                np.mean(np.abs(univariate_prediction - calibration_input_psnr))
            ),
        }

    calibration_by_group: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(calibration_records):
        calibration_by_group[str(record["recording_group"])].append(index)
    per_group = []
    for group, indices in sorted(calibration_by_group.items()):
        selected = np.asarray(indices, dtype=np.int64)
        per_group.append(
            {
                "recording_group": group,
                "pairs": len(indices),
                "teacher_input_psnr_db": float(calibration_input_psnr[selected].mean()),
                "predicted_input_psnr_db": float(predicted_calibration_psnr[selected].mean()),
                "mae_db": float(
                    np.mean(
                        np.abs(
                            predicted_calibration_psnr[selected]
                            - calibration_input_psnr[selected]
                        )
                    )
                ),
                "pearson": _correlation(
                    predicted_calibration_psnr[selected],
                    calibration_input_psnr[selected],
                ),
            }
        )

    for record, prediction in zip(fit_records, predicted_fit_psnr, strict=True):
        record["inner_partition"] = "fit"
        record["predicted_input_psnr_db"] = float(prediction)
    for record, prediction in zip(
        calibration_records, predicted_calibration_psnr, strict=True
    ):
        record["inner_partition"] = "calibration"
        record["predicted_input_psnr_db"] = float(prediction)

    report = {
        "experiment": args.output.name,
        "stage": "P7-A deployable blur-strength calibration",
        "baseline_checkpoint": str(baseline_path),
        "baseline_checkpoint_step": int(checkpoint["step"]),
        "seed": args.seed,
        "pairs": len(records),
        "patch_size": args.patch_size,
        "all_training_recording_groups": all_groups,
        "inner_fit_recording_groups": list(fit_groups),
        "inner_calibration_recording_groups": list(calibration_groups),
        "frozen_development_recording_groups": list(frozen_groups),
        "recording_group_overlap_with_frozen_development": sorted(
            set(all_groups) & set(frozen_groups)
        ),
        "fit_pairs": len(fit_records),
        "calibration_pairs": len(calibration_records),
        "teacher_input_psnr_fit_mean_db": float(fit_input_psnr.mean()),
        "teacher_input_psnr_calibration_mean_db": float(calibration_input_psnr.mean()),
        "heavy_medium_threshold_input_psnr_db": float(lower_threshold),
        "medium_light_threshold_input_psnr_db": float(upper_threshold),
        "proxy_reports": proxy_reports,
        "ridge_feature_names": list(FEATURE_NAMES),
        "ridge_coefficients_intercept_then_standardized_features": coefficients.tolist(),
        "ridge_feature_mean": feature_mean.tolist(),
        "ridge_feature_scale": feature_scale.tolist(),
        "fit_regression": {
            "mae_db": float(np.mean(np.abs(predicted_fit_psnr - fit_input_psnr))),
            "rmse_db": float(np.sqrt(np.mean((predicted_fit_psnr - fit_input_psnr) ** 2))),
            "pearson": _correlation(predicted_fit_psnr, fit_input_psnr),
            "spearman": _spearman(predicted_fit_psnr, fit_input_psnr),
            "classification": _classification_report(fit_target_class, fit_predicted_class),
        },
        "calibration_regression": {
            "mae_db": float(
                np.mean(np.abs(predicted_calibration_psnr - calibration_input_psnr))
            ),
            "rmse_db": float(
                np.sqrt(np.mean((predicted_calibration_psnr - calibration_input_psnr) ** 2))
            ),
            "pearson": _correlation(predicted_calibration_psnr, calibration_input_psnr),
            "spearman": _spearman(predicted_calibration_psnr, calibration_input_psnr),
            "classification": _classification_report(
                calibration_target_class,
                calibration_predicted_class,
            ),
        },
        "selected_deployable_proxy": "log(baseline_correction_l1)",
        "correction_gate_calibration": {
            "light_cut_log_l1": gate_light_cut,
            "heavy_cut_log_l1": gate_heavy_cut,
            "light_cut_l1": float(np.exp(gate_light_cut)),
            "heavy_cut_l1": float(np.exp(gate_heavy_cut)),
            "gate_definition": (
                "smoothstep(clamp((log_correction_l1-light_cut)/(heavy_cut-light_cut),0,1))"
            ),
            "fit_classification": gate_fit_classification,
            "calibration_classification": _classification_report(
                calibration_target_class,
                gate_calibration_class,
            ),
            "fit_mean_gate_by_teacher_class": gate_means(fit_gate, fit_target_class),
            "calibration_mean_gate_by_teacher_class": gate_means(
                calibration_gate,
                calibration_target_class,
            ),
        },
        "inner_calibration_per_group": per_group,
        "wall_seconds": time.perf_counter() - started,
        "teacher_uses_sharp_during_calibration": True,
        "selected_features_use_sharp_at_inference": False,
        "official_test_accessed": False,
        "paper_result": False,
        "note": (
            "Training-partition calibration only. Paired PSNR is a teacher label and is not "
            "available to the deployable strength proxy at inference time."
        ),
    }

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ordered_records = sorted(records, key=lambda item: str(item["name"]))
    _write_csv(output / "per_image_strength.csv", ordered_records)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
