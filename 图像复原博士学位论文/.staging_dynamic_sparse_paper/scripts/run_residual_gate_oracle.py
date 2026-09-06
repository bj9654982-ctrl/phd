"""Measure the headroom of degradation-aware residual gating on frozen B0."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GoProPairDataset, recording_group  # noqa: E402
from src.models import CompactDeblurUNet  # noqa: E402


DEFAULT_RUN = PROJECT_ROOT / "results" / "runs" / "P3_baseline_group_split_v0"
AGGREGATE_FIELDS = (
    "input_psnr_db",
    "baseline_psnr_db",
    "global_gate_psnr_db",
    "oracle_gate_psnr_db",
    "global_gain_vs_baseline_db",
    "oracle_gain_vs_baseline_db",
    "input_ssim",
    "baseline_ssim",
    "global_gate_ssim",
    "oracle_gate_ssim",
    "global_ssim_gain_vs_baseline",
    "oracle_ssim_gain_vs_baseline",
    "oracle_alpha",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan residual strength on the frozen training holdout without accessing GoPro test."
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "GOPRO_Large")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN / "best.pt")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN / "residual_gate_oracle")
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 evaluates the complete frozen holdout.")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def _ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction_array = prediction.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    target_array = target.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return float(
        structural_similarity(
            target_array,
            prediction_array,
            data_range=1.0,
            channel_axis=2,
        )
    )


def _write_csv(path: Path, records: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _aggregate(records: list[dict[str, float | str]]) -> dict[str, float | int]:
    return {
        "pairs": len(records),
        **{
            field: statistics.fmean(float(record[field]) for record in records)
            for field in AGGREGATE_FIELDS
        },
    }


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _render_diagnostic(
    output: Path,
    alphas: np.ndarray,
    mean_psnr: np.ndarray,
    input_psnr: np.ndarray,
    oracle_alpha: np.ndarray,
    global_index: int,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    axes[0].plot(alphas, mean_psnr, marker="o", markersize=3)
    axes[0].axvline(alphas[global_index], linestyle="--", linewidth=1)
    axes[0].scatter([alphas[global_index]], [mean_psnr[global_index]], s=45, zorder=3)
    axes[0].set_title("Global residual-strength scan")
    axes[0].set_xlabel("Residual strength alpha")
    axes[0].set_ylabel("Mean validation PSNR (dB)")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(input_psnr, oracle_alpha, s=10, alpha=0.5)
    axes[1].set_title("Per-image oracle strength")
    axes[1].set_xlabel("Blurry-input PSNR (dB)")
    axes[1].set_ylabel("Oracle alpha")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].grid(alpha=0.25)

    counts = np.array([(oracle_alpha == alpha).sum() for alpha in alphas])
    axes[2].bar(alphas, counts, width=max(float(alphas[1] - alphas[0]) * 0.8, 0.01))
    axes[2].set_title("Oracle-strength distribution")
    axes[2].set_xlabel("Oracle alpha")
    axes[2].set_ylabel("Number of images")
    axes[2].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(output / "residual_gate_diagnostic.png", dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script has no CPU fallback")
    if args.max_pairs < 0 or args.num_workers < 0:
        raise ValueError("max-pairs and num-workers must be nonnegative")
    if not 0 < args.alpha_step <= 1:
        raise ValueError("alpha-step must be in (0, 1]")
    alpha_intervals = round(1.0 / args.alpha_step)
    if not np.isclose(alpha_intervals * args.alpha_step, 1.0, atol=1e-8):
        raise ValueError("alpha-step must divide the [0, 1] interval exactly")
    alpha_values = np.linspace(0.0, 1.0, alpha_intervals + 1, dtype=np.float32)

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    device = torch.device("cuda:0")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    if checkpoint_args.get("variant") != "local":
        raise ValueError("This diagnostic accepts only the local B0 baseline checkpoint")
    if checkpoint_args.get("validation_source") != "train_holdout":
        raise ValueError("Checkpoint was not trained with validation_source=train_holdout")
    validation_groups = tuple(checkpoint_args.get("validation_groups") or ())
    if not validation_groups:
        raise ValueError("Checkpoint does not contain frozen validation recording groups")

    dataset = GoProPairDataset(
        args.data_root,
        split="train",
        patch_size=int(checkpoint_args["val_patch_size"]),
        augment=False,
        max_pairs=args.max_pairs or None,
        include_recording_groups=validation_groups,
    )
    if set(dataset.recording_groups) != set(validation_groups):
        raise RuntimeError("Loaded holdout groups do not match the groups frozen in the checkpoint")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    model = CompactDeblurUNet(width=int(checkpoint_args["width"])).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)

    alpha_tensor = torch.from_numpy(alpha_values).to(device=device)[:, None, None, None, None]
    psnr_curves: list[np.ndarray] = []
    identifiers: list[dict[str, str]] = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for index, batch in enumerate(tqdm(loader, desc="Residual gate PSNR scan")):
        blurry = batch["blur"].to(device, non_blocking=True)
        sharp = batch["sharp"].to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            restored = model(blurry)
        candidates = (blurry[None] + alpha_tensor * (restored - blurry)[None]).clamp(0, 1)
        mse = (candidates.float() - sharp.float()[None]).square().mean(dim=(1, 2, 3, 4))
        curve = (-10.0 * torch.log10(mse.clamp_min(torch.finfo(mse.dtype).tiny))).cpu().numpy()
        psnr_curves.append(curve)
        pair = dataset.pairs[index]
        identifiers.append(
            {
                "recording_group": recording_group(pair.scene),
                "scene": pair.scene,
                "name": f"{pair.scene}/{pair.name}",
            }
        )

    curve_matrix = np.stack(psnr_curves)
    mean_psnr_curve = curve_matrix.mean(axis=0)
    global_index = int(np.argmax(mean_psnr_curve))
    oracle_indices = np.argmax(curve_matrix, axis=1)
    oracle_alphas = alpha_values[oracle_indices]
    input_psnr_values = curve_matrix[:, 0]
    baseline_psnr_values = curve_matrix[:, -1]
    if not np.isclose(float(baseline_psnr_values.mean()), float(mean_psnr_curve[-1]), atol=1e-6):
        raise RuntimeError("Baseline endpoint consistency check failed")

    global_alpha = float(alpha_values[global_index])
    records: list[dict[str, float | str]] = []
    for index, batch in enumerate(tqdm(loader, desc="Residual gate SSIM audit")):
        blurry = batch["blur"].to(device, non_blocking=True)
        sharp = batch["sharp"].to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            restored = model(blurry)
        global_candidate = (blurry + global_alpha * (restored - blurry)).clamp(0, 1)
        oracle_alpha = float(oracle_alphas[index])
        oracle_candidate = (blurry + oracle_alpha * (restored - blurry)).clamp(0, 1)
        input_ssim = _ssim(blurry[0], sharp[0])
        baseline_ssim = _ssim(restored[0], sharp[0])
        global_ssim = _ssim(global_candidate[0], sharp[0])
        oracle_ssim = _ssim(oracle_candidate[0], sharp[0])
        baseline_psnr = float(curve_matrix[index, -1])
        global_psnr = float(curve_matrix[index, global_index])
        oracle_psnr = float(curve_matrix[index, oracle_indices[index]])
        records.append(
            {
                **identifiers[index],
                "input_psnr_db": float(curve_matrix[index, 0]),
                "baseline_psnr_db": baseline_psnr,
                "global_gate_psnr_db": global_psnr,
                "oracle_gate_psnr_db": oracle_psnr,
                "global_gain_vs_baseline_db": global_psnr - baseline_psnr,
                "oracle_gain_vs_baseline_db": oracle_psnr - baseline_psnr,
                "input_ssim": input_ssim,
                "baseline_ssim": baseline_ssim,
                "global_gate_ssim": global_ssim,
                "oracle_gate_ssim": oracle_ssim,
                "global_ssim_gain_vs_baseline": global_ssim - baseline_ssim,
                "oracle_ssim_gain_vs_baseline": oracle_ssim - baseline_ssim,
                "oracle_alpha": oracle_alpha,
            }
        )

    grouped_records: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for record in records:
        grouped_records[str(record["recording_group"])].append(record)
    group_summaries: list[dict[str, float | int | str]] = [
        {"recording_group": group_name, **_aggregate(group_records)}
        for group_name, group_records in sorted(grouped_records.items())
    ]
    overall = _aggregate(records)
    oracle_gain = np.array([float(record["oracle_gain_vs_baseline_db"]) for record in records])

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_image_oracle.csv", records)
    _write_csv(output / "per_group_oracle.csv", group_summaries)
    alpha_curve_records: list[dict[str, float | int]] = [
        {
            "alpha": float(alpha),
            "mean_psnr_db": float(mean_psnr_curve[index]),
            "oracle_image_count": int((oracle_alphas == alpha).sum()),
        }
        for index, alpha in enumerate(alpha_values)
    ]
    _write_csv(output / "alpha_curve.csv", alpha_curve_records)
    _render_diagnostic(
        output,
        alpha_values,
        mean_psnr_curve,
        input_psnr_values,
        oracle_alphas,
        global_index,
    )
    summary = {
        "experiment": "P3_baseline_group_split_v0_residual_gate_oracle",
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "data_root": str(args.data_root.expanduser().resolve()),
        "source_split": "train",
        "validation_recording_groups": list(validation_groups),
        "patch_size": int(checkpoint_args["val_patch_size"]),
        "complete_holdout": args.max_pairs == 0,
        "alpha_values": [float(alpha) for alpha in alpha_values],
        "best_global_alpha": global_alpha,
        "mean_psnr_curve_db": [float(value) for value in mean_psnr_curve],
        "overall": overall,
        "per_group": group_summaries,
        "oracle_alpha_median": float(np.median(oracle_alphas)),
        "oracle_alpha_zero_fraction": float((oracle_alphas == 0).mean()),
        "oracle_alpha_at_most_0_25_fraction": float((oracle_alphas <= 0.25).mean()),
        "oracle_alpha_at_least_0_75_fraction": float((oracle_alphas >= 0.75).mean()),
        "oracle_alpha_one_fraction": float((oracle_alphas == 1).mean()),
        "correlation_input_psnr_vs_oracle_alpha": _safe_correlation(input_psnr_values, oracle_alphas),
        "correlation_input_psnr_vs_oracle_gain": _safe_correlation(input_psnr_values, oracle_gain),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "official_test_accessed": False,
        "paper_result": False,
        "note": "Oracle analysis on the frozen train holdout; it measures gating headroom and is not a deployable method result.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(output / "residual_gate_diagnostic.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
