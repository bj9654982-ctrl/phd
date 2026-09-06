"""P9-A: learn when the P8-B trajectory specialist benefits restoration.

Sharp targets supervise the router only on the B0 training partition.  The
deployed router uses pooled statistics from the blurry image, B0 output,
trajectory candidate output, and candidate geometry.  The frozen development
groups are evaluated only after the fit/calibration choice has been fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GoProPairDataset, recording_group  # noqa: E402
from src.models import CompactDeblurUNet, TrajectoryNonlocalBlock  # noqa: E402


DEFAULT_RUNS = PROJECT_ROOT / "results" / "runs"
FEATURE_NAMES = (
    "log_blurry_gradient_rms",
    "log_blurry_laplacian_rms",
    "log_baseline_correction_l1",
    "log_baseline_correction_rms",
    "log_candidate_delta_l1",
    "log_candidate_delta_rms",
    "log_delta_to_correction_rms",
    "delta_correction_cosine",
    "log_baseline_to_blurry_gradient_ratio",
    "log_candidate_to_baseline_gradient_ratio",
    "log_baseline_to_blurry_laplacian_ratio",
    "log_candidate_to_baseline_laplacian_ratio",
    "log_candidate_delta_gradient_rms",
    "blurry_structure_anisotropy",
    "geometry_length_mean",
    "geometry_length_std",
    "geometry_confidence_mean",
    "geometry_confidence_std",
    "geometry_attention_entropy_mean",
    "geometry_attention_entropy_std",
    "geometry_direction_coherence",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate a deployable P8-B benefit router.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "GOPRO_Large")
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=DEFAULT_RUNS / "P3_baseline_group_split_v0" / "best.pt",
    )
    parser.add_argument(
        "--candidate-checkpoint",
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
        default=DEFAULT_RUNS / "P9A_benefit_router_v0",
    )
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--max-fit-pairs", type=int, default=0)
    parser.add_argument("--max-calibration-pairs", type=int, default=0)
    parser.add_argument("--max-development-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _instantiate_candidate(
    checkpoint: dict[str, object],
    width: int,
) -> CompactDeblurUNet:
    oracle_args = dict(checkpoint.get("oracle_args", {}))
    training_args = dict(checkpoint.get("args", {}))
    required = ("trajectory_samples", "max_feature_length", "trajectory_fusion")
    if any(name not in oracle_args for name in required):
        raise ValueError("candidate checkpoint is missing trajectory configuration")
    block = TrajectoryNonlocalBlock(
        channels=width * 4,
        samples=int(oracle_args["trajectory_samples"]),
        max_feature_length=float(oracle_args["max_feature_length"]),
        fusion_mode=str(oracle_args["trajectory_fusion"]),
        angle_hypotheses=1,
        direction_residual_scale=float(training_args.get("direction_residual_scale", 2.0)),
    )
    model = CompactDeblurUNet(width=width, bottleneck_module=block)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def _gray(image: torch.Tensor) -> torch.Tensor:
    return 0.2989 * image[:, 0:1] + 0.5870 * image[:, 1:2] + 0.1140 * image[:, 2:3]


def _gradient_rms(image: torch.Tensor) -> torch.Tensor:
    gray = _gray(image.float())
    dx = F.pad(gray[:, :, :, 2:] - gray[:, :, :, :-2], (1, 1, 0, 0), mode="replicate")
    dy = F.pad(gray[:, :, 2:, :] - gray[:, :, :-2, :], (0, 0, 1, 1), mode="replicate")
    return (dx.square() + dy.square()).flatten(1).mean(1).clamp_min(1e-12).sqrt()


def _laplacian_rms(image: torch.Tensor) -> torch.Tensor:
    gray = _gray(image.float())
    kernel = gray.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    )[None, None]
    response = F.conv2d(gray, kernel, padding=1)
    return response.square().flatten(1).mean(1).clamp_min(1e-12).sqrt()


def _structure_anisotropy(image: torch.Tensor) -> torch.Tensor:
    gray = _gray(image.float())
    dx = F.pad(gray[:, :, :, 2:] - gray[:, :, :, :-2], (1, 1, 0, 0), mode="replicate")
    dy = F.pad(gray[:, :, 2:, :] - gray[:, :, :-2, :], (0, 0, 1, 1), mode="replicate")
    jxx = F.avg_pool2d(dx.square(), 31, stride=1, padding=15)
    jyy = F.avg_pool2d(dy.square(), 31, stride=1, padding=15)
    jxy = F.avg_pool2d(dx * dy, 31, stride=1, padding=15)
    value = torch.sqrt((jxx - jyy).square() + 4.0 * jxy.square() + 1e-12)
    return (value / (jxx + jyy + 1e-8)).flatten(1).mean(1)


def _mean_std(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flattened = tensor.float().flatten(1)
    return flattened.mean(1), flattened.std(1, unbiased=False)


def _deployable_features(
    blurry: torch.Tensor,
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    geometry: dict[str, torch.Tensor],
) -> torch.Tensor:
    eps = 1e-12
    correction = baseline.float() - blurry.float()
    delta = candidate.float() - baseline.float()
    correction_l1 = correction.abs().flatten(1).mean(1).clamp_min(eps)
    correction_rms = correction.square().flatten(1).mean(1).clamp_min(eps).sqrt()
    delta_l1 = delta.abs().flatten(1).mean(1).clamp_min(eps)
    delta_rms = delta.square().flatten(1).mean(1).clamp_min(eps).sqrt()
    cosine = (delta * correction).flatten(1).mean(1) / (delta_rms * correction_rms + eps)

    blurry_gradient = _gradient_rms(blurry)
    baseline_gradient = _gradient_rms(baseline)
    candidate_gradient = _gradient_rms(candidate)
    delta_gradient = _gradient_rms(delta)
    blurry_laplacian = _laplacian_rms(blurry)
    baseline_laplacian = _laplacian_rms(baseline)
    candidate_laplacian = _laplacian_rms(candidate)

    length_mean, length_std = _mean_std(geometry["length"])
    confidence_mean, confidence_std = _mean_std(geometry["confidence"])
    attention = geometry["attention"].float().clamp_min(1e-12)
    entropy = -(attention * attention.log()).sum(1) / math.log(attention.shape[1])
    entropy_mean, entropy_std = _mean_std(entropy)
    double_angle_mean = geometry["double_angle"].float().flatten(2).mean(2)
    direction_coherence = double_angle_mean.square().sum(1).sqrt()

    features = torch.stack(
        (
            blurry_gradient.log(),
            blurry_laplacian.log(),
            correction_l1.log(),
            correction_rms.log(),
            delta_l1.log(),
            delta_rms.log(),
            (delta_rms / correction_rms).clamp_min(eps).log(),
            cosine,
            (baseline_gradient / blurry_gradient).clamp_min(eps).log(),
            (candidate_gradient / baseline_gradient).clamp_min(eps).log(),
            (baseline_laplacian / blurry_laplacian).clamp_min(eps).log(),
            (candidate_laplacian / baseline_laplacian).clamp_min(eps).log(),
            delta_gradient.log(),
            _structure_anisotropy(blurry),
            length_mean,
            length_std,
            confidence_mean,
            confidence_std,
            entropy_mean,
            entropy_std,
            direction_coherence,
        ),
        dim=1,
    )
    if features.shape[1] != len(FEATURE_NAMES) or not torch.isfinite(features).all():
        raise RuntimeError("non-finite or incorrectly sized deployable feature vector")
    return features


def _psnr_from_mse(mse: np.ndarray) -> np.ndarray:
    return -10.0 * np.log10(np.clip(mse, 1e-12, None))


@torch.no_grad()
def _extract(
    baseline: CompactDeblurUNet,
    candidate: CompactDeblurUNet,
    loader: DataLoader,
    heavy_max_psnr_db: float,
    light_min_psnr_db: float,
    description: str,
) -> list[dict[str, object]]:
    baseline.eval()
    candidate.eval()
    records: list[dict[str, object]] = []
    for batch in tqdm(loader, desc=description):
        blurry = batch["blur"].cuda(non_blocking=True)
        sharp = batch["sharp"].cuda(non_blocking=True).float().clamp(0.0, 1.0)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            baseline_output = baseline(blurry)
            candidate_output, geometry = candidate(blurry, return_geometry=True)
        blurry = blurry.float().clamp(0.0, 1.0)
        baseline_output = baseline_output.float().clamp(0.0, 1.0)
        candidate_output = candidate_output.float().clamp(0.0, 1.0)
        features = _deployable_features(blurry, baseline_output, candidate_output, geometry)

        baseline_error = baseline_output - sharp
        delta = candidate_output - baseline_output
        baseline_mse = baseline_error.square().flatten(1).mean(1)
        cross = (baseline_error * delta).flatten(1).mean(1)
        delta_square = delta.square().flatten(1).mean(1)
        optimal_gate = (-cross / delta_square.clamp_min(1e-12)).clamp(0.0, 1.0)
        input_mse = (blurry - sharp).square().flatten(1).mean(1)
        candidate_mse = (candidate_output - sharp).square().flatten(1).mean(1)

        for index, name in enumerate(batch["name"]):
            input_psnr = float(-10.0 * input_mse[index].clamp_min(1e-12).log10())
            strength = (
                "heavy"
                if input_psnr <= heavy_max_psnr_db
                else "light" if input_psnr >= light_min_psnr_db else "medium"
            )
            scene = str(name).split("/", maxsplit=1)[0]
            records.append(
                {
                    "name": str(name),
                    "recording_group": recording_group(scene),
                    "strength_class": strength,
                    "features": features[index].cpu().numpy().astype(np.float64),
                    "input_psnr_db": input_psnr,
                    "baseline_mse": float(baseline_mse[index]),
                    "cross": float(cross[index]),
                    "delta_square": float(delta_square[index]),
                    "candidate_mse": float(candidate_mse[index]),
                    "optimal_gate": float(optimal_gate[index]),
                }
            )
    return records


def _arrays(records: list[dict[str, object]]) -> tuple[np.ndarray, ...]:
    return (
        np.stack([np.asarray(record["features"]) for record in records]),
        np.asarray([float(record["optimal_gate"]) for record in records]),
        np.asarray([float(record["baseline_mse"]) for record in records]),
        np.asarray([float(record["cross"]) for record in records]),
        np.asarray([float(record["delta_square"]) for record in records]),
    )


def _design_fit(features: np.ndarray, degree: int) -> tuple[np.ndarray, dict[str, object]]:
    raw_mean = features.mean(axis=0)
    raw_scale = features.std(axis=0).clip(min=1e-8)
    standardized = (features - raw_mean) / raw_scale
    if degree == 1:
        design = standardized
        pairs: list[tuple[int, int]] = []
    else:
        pairs = [(i, j) for i in range(features.shape[1]) for j in range(i, features.shape[1])]
        products = np.column_stack([standardized[:, i] * standardized[:, j] for i, j in pairs])
        design = np.column_stack((standardized, products))
    design_mean = design.mean(axis=0)
    design_scale = design.std(axis=0).clip(min=1e-8)
    return (design - design_mean) / design_scale, {
        "degree": degree,
        "raw_mean": raw_mean.tolist(),
        "raw_scale": raw_scale.tolist(),
        "quadratic_pairs": pairs,
        "design_mean": design_mean.tolist(),
        "design_scale": design_scale.tolist(),
    }


def _design_apply(features: np.ndarray, transform: dict[str, object]) -> np.ndarray:
    raw_mean = np.asarray(transform["raw_mean"], dtype=np.float64)
    raw_scale = np.asarray(transform["raw_scale"], dtype=np.float64)
    standardized = (features - raw_mean) / raw_scale
    pairs = [tuple(pair) for pair in transform["quadratic_pairs"]]
    if pairs:
        products = np.column_stack([standardized[:, i] * standardized[:, j] for i, j in pairs])
        design = np.column_stack((standardized, products))
    else:
        design = standardized
    mean = np.asarray(transform["design_mean"], dtype=np.float64)
    scale = np.asarray(transform["design_scale"], dtype=np.float64)
    return (design - mean) / scale


def _fit_ridge(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    augmented = np.column_stack((np.ones(len(design)), design))
    penalty = np.eye(augmented.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ target)


def _predict(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(design)), design)) @ coefficients


def _gains(
    records: list[dict[str, object]],
    gates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, baseline_mse, cross, delta_square = _arrays(records)
    final_mse = baseline_mse + 2.0 * gates * cross + np.square(gates) * delta_square
    baseline_psnr = _psnr_from_mse(baseline_mse)
    final_psnr = _psnr_from_mse(final_mse)
    return baseline_psnr, final_psnr, final_psnr - baseline_psnr


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def _group_means(
    records: list[dict[str, object]],
    values: np.ndarray,
    field: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record, value in zip(records, values, strict=True):
        grouped[str(record[field])].append(float(value))
    return {name: statistics.fmean(items) for name, items in sorted(grouped.items())}


def _calibrate_affine_gate(
    records: list[dict[str, object]],
    raw_prediction: np.ndarray,
) -> dict[str, object]:
    best: dict[str, object] | None = None
    for scale in np.linspace(0.0, 2.0, 17):
        for bias in np.linspace(-0.5, 0.5, 21):
            gates = np.clip(scale * raw_prediction + bias, 0.0, 1.0)
            _, _, gains = _gains(records, gates)
            group_gain = _group_means(records, gains, "recording_group")
            strength_gain = _group_means(records, gains, "strength_class")
            feasible = (
                min(group_gain.values()) >= -0.01
                and strength_gain.get("light", 0.0) >= -0.005
            )
            candidate = {
                "scale": float(scale),
                "bias": float(bias),
                "mean_gain_db": _mean(gains),
                "minimum_group_gain_db": min(group_gain.values()),
                "light_gain_db": strength_gain.get("light", float("nan")),
                "feasible": feasible,
            }
            if feasible and (best is None or candidate["mean_gain_db"] > best["mean_gain_db"]):
                best = candidate
    if best is None:
        raise RuntimeError("no feasible affine router; the zero gate should always be feasible")
    return best


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.std() < 1e-12 or right.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
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


def _router_metrics(
    records: list[dict[str, object]],
    raw_prediction: np.ndarray,
    affine: dict[str, object],
) -> dict[str, object]:
    gates = np.clip(
        float(affine["scale"]) * raw_prediction + float(affine["bias"]),
        0.0,
        1.0,
    )
    baseline_psnr, final_psnr, gains = _gains(records, gates)
    candidate_psnr = _psnr_from_mse(
        np.asarray([float(record["candidate_mse"]) for record in records])
    )
    endpoint_benefit = candidate_psnr - baseline_psnr
    optimal_gate = np.asarray([float(record["optimal_gate"]) for record in records])
    oracle_gains = _gains(records, optimal_gate)[2]
    endpoint_oracle = np.maximum(endpoint_benefit, 0.0)
    return {
        "pairs": len(records),
        "baseline_psnr_db": _mean(baseline_psnr),
        "candidate_psnr_db": _mean(candidate_psnr),
        "final_psnr_db": _mean(final_psnr),
        "candidate_minus_baseline_db": _mean(endpoint_benefit),
        "final_minus_baseline_db": _mean(gains),
        "mean_gate": _mean(gates),
        "nonzero_gate_pairs": int(np.sum(gates > 0.0)),
        "full_gate_pairs": int(np.sum(gates >= 1.0)),
        "endpoint_oracle_gain_db": _mean(endpoint_oracle),
        "continuous_oracle_gain_db": _mean(oracle_gains),
        "raw_prediction_vs_optimal_gate_pearson": _correlation(raw_prediction, optimal_gate),
        "raw_prediction_vs_optimal_gate_spearman": _correlation(
            _rank(raw_prediction), _rank(optimal_gate)
        ),
        "per_recording_group_gain_db": _group_means(records, gains, "recording_group"),
        "per_strength_gain_db": _group_means(records, gains, "strength_class"),
        "gates": gates,
        "baseline_psnr": baseline_psnr,
        "candidate_psnr": candidate_psnr,
        "final_psnr": final_psnr,
        "gains": gains,
    }


def _serializable_metrics(metrics: dict[str, object]) -> dict[str, object]:
    hidden = {"gates", "baseline_psnr", "candidate_psnr", "final_psnr", "gains"}
    return {name: value for name, value in metrics.items() if name not in hidden}


@torch.no_grad()
def _development_ssim(
    baseline: CompactDeblurUNet,
    candidate: CompactDeblurUNet,
    loader: DataLoader,
    gate_by_name: dict[str, float],
) -> tuple[float, float, float, list[dict[str, float | str]]]:
    baseline_values: list[float] = []
    candidate_values: list[float] = []
    final_values: list[float] = []
    rows: list[dict[str, float | str]] = []
    for batch in tqdm(loader, desc="P9 development SSIM"):
        blurry = batch["blur"].cuda(non_blocking=True)
        sharp = batch["sharp"].float().clamp(0.0, 1.0)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            baseline_output = baseline(blurry)
            candidate_output = candidate(blurry)
        baseline_output = baseline_output.float().clamp(0.0, 1.0).cpu()
        candidate_output = candidate_output.float().clamp(0.0, 1.0).cpu()
        for index, name in enumerate(batch["name"]):
            gate = gate_by_name[str(name)]
            final_output = baseline_output[index] + gate * (
                candidate_output[index] - baseline_output[index]
            )
            target = sharp[index]

            def score(image: torch.Tensor) -> float:
                return float(
                    structural_similarity(
                        target.permute(1, 2, 0).numpy(),
                        image.permute(1, 2, 0).numpy(),
                        data_range=1.0,
                        channel_axis=2,
                    )
                )

            baseline_ssim = score(baseline_output[index])
            candidate_ssim = score(candidate_output[index])
            final_ssim = score(final_output)
            baseline_values.append(baseline_ssim)
            candidate_values.append(candidate_ssim)
            final_values.append(final_ssim)
            rows.append(
                {
                    "name": str(name),
                    "baseline_ssim": baseline_ssim,
                    "candidate_ssim": candidate_ssim,
                    "final_ssim": final_ssim,
                }
            )
    return (
        statistics.fmean(baseline_values),
        statistics.fmean(candidate_values),
        statistics.fmean(final_values),
        rows,
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; P9-A intentionally has no CPU fallback")
    if min(args.patch_size, args.batch_size) <= 0 or args.num_workers < 0 or args.ridge < 0:
        raise ValueError("invalid patch, batch, worker, or ridge configuration")
    _set_determinism(args.seed)
    started = time.perf_counter()
    device = torch.device("cuda:0")

    baseline_path = args.baseline_checkpoint.expanduser().resolve()
    candidate_path = args.candidate_checkpoint.expanduser().resolve()
    calibration_path = args.calibration_summary.expanduser().resolve()
    for description, path in (
        ("baseline checkpoint", baseline_path),
        ("candidate checkpoint", candidate_path),
        ("calibration summary", calibration_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")

    baseline_checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=False)
    candidate_checkpoint = torch.load(candidate_path, map_location="cpu", weights_only=False)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("official_test_accessed") is not False:
        raise ValueError("calibration summary does not preserve the official-test lock")
    fit_groups = tuple(calibration["inner_fit_recording_groups"])
    calibration_groups = tuple(calibration["inner_calibration_recording_groups"])
    development_groups = tuple(calibration["frozen_development_recording_groups"])
    if (set(fit_groups) & set(calibration_groups)) or (
        (set(fit_groups) | set(calibration_groups)) & set(development_groups)
    ):
        raise RuntimeError("recording-group leakage in P9 partitions")

    baseline_args = dict(baseline_checkpoint["args"])
    width = int(baseline_args["width"])
    baseline = CompactDeblurUNet(width=width)
    baseline.load_state_dict(baseline_checkpoint["model"], strict=True)
    candidate = _instantiate_candidate(candidate_checkpoint, width)
    baseline = baseline.to(device).eval()
    candidate = candidate.to(device).eval()
    for model in (baseline, candidate):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    dataset_options = {
        "root": args.data_root,
        "split": "train",
        "patch_size": args.patch_size,
        "augment": False,
    }
    fit_set = GoProPairDataset(
        **dataset_options,
        include_recording_groups=fit_groups,
        max_pairs=args.max_fit_pairs or None,
    )
    calibration_set = GoProPairDataset(
        **dataset_options,
        include_recording_groups=calibration_groups,
        max_pairs=args.max_calibration_pairs or None,
    )
    development_set = GoProPairDataset(
        **dataset_options,
        include_recording_groups=development_groups,
        max_pairs=args.max_development_pairs or None,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    fit_loader = DataLoader(fit_set, **loader_options)
    calibration_loader = DataLoader(calibration_set, **loader_options)
    development_loader = DataLoader(development_set, **loader_options)
    heavy_max = float(calibration["heavy_medium_threshold_input_psnr_db"])
    light_min = float(calibration["medium_light_threshold_input_psnr_db"])

    torch.cuda.reset_peak_memory_stats(device)
    fit_records = _extract(baseline, candidate, fit_loader, heavy_max, light_min, "P9 fit features")
    calibration_records = _extract(
        baseline, candidate, calibration_loader, heavy_max, light_min, "P9 calibration features"
    )

    fit_x, fit_target, *_ = _arrays(fit_records)
    calibration_x, *_ = _arrays(calibration_records)
    candidates: list[dict[str, object]] = []
    for degree in (1, 2):
        fit_design, transform = _design_fit(fit_x, degree)
        coefficients = _fit_ridge(fit_design, fit_target, args.ridge)
        fit_prediction = _predict(fit_design, coefficients)
        calibration_prediction = _predict(_design_apply(calibration_x, transform), coefficients)
        affine = _calibrate_affine_gate(calibration_records, calibration_prediction)
        fit_metrics = _router_metrics(fit_records, fit_prediction, affine)
        calibration_metrics = _router_metrics(
            calibration_records, calibration_prediction, affine
        )
        candidates.append(
            {
                "degree": degree,
                "transform": transform,
                "coefficients": coefficients.tolist(),
                "affine": affine,
                "fit": _serializable_metrics(fit_metrics),
                "calibration": _serializable_metrics(calibration_metrics),
            }
        )

    selected = max(
        candidates,
        key=lambda item: float(item["calibration"]["final_minus_baseline_db"]),
    )
    development_records = _extract(
        baseline, candidate, development_loader, heavy_max, light_min, "P9 frozen development"
    )
    development_x, *_ = _arrays(development_records)
    development_prediction = _predict(
        _design_apply(development_x, selected["transform"]),
        np.asarray(selected["coefficients"], dtype=np.float64),
    )
    development_metrics = _router_metrics(
        development_records,
        development_prediction,
        selected["affine"],
    )
    gates = np.asarray(development_metrics["gates"])
    gate_by_name = {
        str(record["name"]): float(gate)
        for record, gate in zip(development_records, gates, strict=True)
    }
    baseline_ssim, candidate_ssim, final_ssim, ssim_rows = _development_ssim(
        baseline, candidate, development_loader, gate_by_name
    )

    ssim_by_name = {str(row["name"]): row for row in ssim_rows}
    per_image: list[dict[str, object]] = []
    for index, (record, gate) in enumerate(
        zip(development_records, gates, strict=True)
    ):
        name = str(record["name"])
        ssim = ssim_by_name[name]
        per_image.append(
            {
                "recording_group": record["recording_group"],
                "name": name,
                "strength_class": record["strength_class"],
                "input_psnr_db": record["input_psnr_db"],
                "baseline_psnr_db": float(development_metrics["baseline_psnr"][index]),
                "candidate_psnr_db": float(development_metrics["candidate_psnr"][index]),
                "final_psnr_db": float(development_metrics["final_psnr"][index]),
                "candidate_minus_baseline_db": float(
                    development_metrics["candidate_psnr"][index]
                    - development_metrics["baseline_psnr"][index]
                ),
                "final_minus_baseline_db": float(development_metrics["gains"][index]),
                "optimal_gate_training_diagnostic": record["optimal_gate"],
                "predicted_gate": float(gate),
                "baseline_ssim": ssim["baseline_ssim"],
                "candidate_ssim": ssim["candidate_ssim"],
                "final_ssim": ssim["final_ssim"],
                "final_minus_baseline_ssim": float(ssim["final_ssim"] - ssim["baseline_ssim"]),
            }
        )

    development_public = _serializable_metrics(development_metrics)
    per_group = development_public["per_recording_group_gain_db"]
    per_strength = development_public["per_strength_gain_db"]
    gain = float(development_public["final_minus_baseline_db"])
    ssim_gain = final_ssim - baseline_ssim
    gates_passed = {
        "psnr_gain_at_least_0_075_db": gain >= 0.075,
        "ssim_non_decreasing": ssim_gain >= 0.0,
        "minimum_group_at_least_minus_0_01_db": min(per_group.values()) >= -0.01,
        "heavy_gain_at_least_0_05_db": per_strength.get("heavy", -math.inf) >= 0.05,
        "light_gain_at_least_minus_0_005_db": per_strength.get("light", -math.inf) >= -0.005,
    }
    passed = all(gates_passed.values())
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    router = {
        "feature_names": FEATURE_NAMES,
        "ridge": args.ridge,
        "selected_degree": selected["degree"],
        "transform": selected["transform"],
        "coefficients_intercept_then_design": selected["coefficients"],
        "affine_gate": selected["affine"],
        "uses_sharp_at_inference": False,
    }
    report = {
        "experiment": output.name,
        "stage": "P9-A training-only benefit routing diagnostic",
        "baseline_checkpoint": str(baseline_path),
        "baseline_checkpoint_step": int(baseline_checkpoint["step"]),
        "candidate_checkpoint": str(candidate_path),
        "candidate_checkpoint_step": int(candidate_checkpoint["step"]),
        "calibration_summary": str(calibration_path),
        "seed": args.seed,
        "patch_size": args.patch_size,
        "fit_pairs": len(fit_records),
        "calibration_pairs": len(calibration_records),
        "development_pairs": len(development_records),
        "fit_recording_groups": fit_groups,
        "calibration_recording_groups": calibration_groups,
        "frozen_development_recording_groups": development_groups,
        "candidate_models": candidates,
        "selected_degree": selected["degree"],
        "selected_affine_gate": selected["affine"],
        "development": development_public,
        "baseline_ssim": baseline_ssim,
        "candidate_ssim": candidate_ssim,
        "final_ssim": final_ssim,
        "final_minus_baseline_ssim": ssim_gain,
        "per_recording_group_gain_db": per_group,
        "per_strength_gain_db": per_strength,
        "decision_gates": gates_passed,
        "passed": passed,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "wall_seconds": time.perf_counter() - started,
        "teacher_uses_sharp_during_router_training": True,
        "router_uses_sharp_at_inference": False,
        "candidate_was_trained_on_router_fit_and_calibration_images": True,
        "official_test_accessed": False,
        "paper_result": False,
        "note": (
            "Development diagnostic only. The P8-B candidate was trained on all non-development "
            "training groups, so a positive result still requires an out-of-fold router audit "
            "before the method can be frozen for official testing."
        ),
    }
    _write_csv(output / "per_image_metrics.csv", per_image)
    (output / "router.json").write_text(
        json.dumps(router, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "experiment": report["experiment"],
                "selected_degree": report["selected_degree"],
                "selected_affine_gate": report["selected_affine_gate"],
                "development": report["development"],
                "final_minus_baseline_ssim": report["final_minus_baseline_ssim"],
                "decision_gates": report["decision_gates"],
                "passed": report["passed"],
                "official_test_accessed": report["official_test_accessed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Summary: {output / 'summary.json'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
