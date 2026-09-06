"""Inference-only intervention audit for the trained P12-D condition branch.

The same trained checkpoint is evaluated four times.  Forward hooks replace
selected condition logits with zero, which gives uniform scale priors and/or a
0.5 local--nonlocal gate without changing any stored parameter.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GoProPairDataset, recording_group  # noqa: E402
from src.models import CompactDeblurUNet, MultiScaleNonlocalBlock  # noqa: E402


INTERVENTIONS = (
    "adaptive_full",
    "no_scale_adaptation",
    "no_fusion_gate",
    "no_scale_or_gate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit P12-D by disabling learned conditioning at inference time."
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "GOPRO_Large")
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "P3_baseline_group_split_v0" / "best.pt",
    )
    parser.add_argument(
        "--adaptive-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "P12D_blur_aware_nonlocal_v0" / "best.pt",
    )
    parser.add_argument(
        "--calibration-summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "P7A_blur_strength_calibration_v1" / "summary.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-pairs", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def psnr_value(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = (prediction.float().clamp(0, 1) - target.float().clamp(0, 1)).square().mean()
    return float(-10.0 * mse.clamp_min(1e-12).log10())


def ssim_value(prediction: torch.Tensor, target: torch.Tensor) -> float:
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


def strength_class(input_psnr_db: float, heavy_max: float, light_min: float) -> str:
    if input_psnr_db <= heavy_max:
        return "heavy"
    if input_psnr_db >= light_min:
        return "light"
    return "medium"


def write_csv(path: Path, records: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


@contextmanager
def apply_condition_intervention(
    block: MultiScaleNonlocalBlock,
    intervention: str,
) -> Iterator[None]:
    """Temporarily replace learned condition logits without changing weights."""
    if intervention not in INTERVENTIONS:
        raise ValueError(f"unknown intervention: {intervention}")
    if intervention == "adaptive_full":
        yield
        return

    scale_count = len(block.scales)
    disable_scale = intervention in {"no_scale_adaptation", "no_scale_or_gate"}
    disable_gate = intervention in {"no_fusion_gate", "no_scale_or_gate"}

    def replace_logits(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        adjusted = output.clone()
        if disable_scale:
            adjusted[:, :scale_count].zero_()  # softmax(0) = uniform scale prior
        if disable_gate:
            adjusted[:, scale_count:].zero_()  # sigmoid(0) = 0.5 fusion gate
        return adjusted

    handle = block.condition_head.register_forward_hook(replace_logits)
    try:
        yield
    finally:
        handle.remove()


def aggregate(records: list[dict[str, float | str]]) -> dict[str, object]:
    result: dict[str, object] = {
        "pairs": len(records),
        "input_psnr_db": statistics.fmean(float(item["input_psnr_db"]) for item in records),
        "interventions": {},
    }
    full_psnr = "adaptive_full_psnr_db"
    full_ssim = "adaptive_full_ssim"
    interventions: dict[str, dict[str, float | int]] = {}
    for name in INTERVENTIONS:
        psnr_key = f"{name}_psnr_db"
        ssim_key = f"{name}_ssim"
        values: dict[str, float | int] = {
            "psnr_db": statistics.fmean(float(item[psnr_key]) for item in records),
            "ssim": statistics.fmean(float(item[ssim_key]) for item in records),
        }
        if name == "adaptive_full":
            values.update(
                {
                    "full_minus_intervention_psnr_db": 0.0,
                    "full_minus_intervention_ssim": 0.0,
                    "images_full_better": 0,
                }
            )
        else:
            values.update(
                {
                    "full_minus_intervention_psnr_db": statistics.fmean(
                        float(item[full_psnr]) - float(item[psnr_key]) for item in records
                    ),
                    "full_minus_intervention_ssim": statistics.fmean(
                        float(item[full_ssim]) - float(item[ssim_key]) for item in records
                    ),
                    "images_full_better": sum(
                        float(item[full_psnr]) > float(item[psnr_key]) for item in records
                    ),
                }
            )
        interventions[name] = values
    result["interventions"] = interventions
    return result


def flatten_aggregate(
    label_name: str,
    label_value: str,
    aggregated: dict[str, object],
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        label_name: label_value,
        "pairs": int(aggregated["pairs"]),
        "input_psnr_db": float(aggregated["input_psnr_db"]),
    }
    variants = aggregated["interventions"]
    assert isinstance(variants, dict)
    for name in INTERVENTIONS:
        metrics = variants[name]
        assert isinstance(metrics, dict)
        for metric_name, value in metrics.items():
            row[f"{name}_{metric_name}"] = value
    return row


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this audit intentionally has no CPU fallback")
    if args.val_pairs < 0 or args.num_workers < 0:
        raise ValueError("val-pairs and num-workers must be nonnegative")

    set_determinism(args.seed)
    device = torch.device("cuda:0")
    baseline_path = args.baseline_checkpoint.expanduser().resolve()
    adaptive_path = args.adaptive_checkpoint.expanduser().resolve()
    calibration_path = args.calibration_summary.expanduser().resolve()
    for path in (baseline_path, adaptive_path, calibration_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    baseline_checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=False)
    baseline_args = dict(baseline_checkpoint.get("args", {}))
    if baseline_args.get("variant") != "local":
        raise ValueError("baseline checkpoint must use variant=local")
    if baseline_args.get("validation_source") != "train_holdout":
        raise ValueError("baseline checkpoint must use validation_source=train_holdout")
    validation_groups = tuple(baseline_args.get("validation_groups") or ())
    if not validation_groups:
        raise ValueError("baseline checkpoint does not contain frozen validation groups")

    adaptive_checkpoint = torch.load(adaptive_path, map_location="cpu", weights_only=False)
    adaptive_args = dict(adaptive_checkpoint.get("args", {}))
    if adaptive_args.get("variant") != "adaptive":
        raise ValueError("adaptive checkpoint must use variant=adaptive")
    if int(adaptive_checkpoint.get("step", -1)) <= 0:
        raise ValueError("adaptive checkpoint has no valid training step")

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("official_test_accessed") is not False:
        raise ValueError("calibration must explicitly state official_test_accessed=false")
    heavy_max = float(calibration["heavy_medium_threshold_input_psnr_db"])
    light_min = float(calibration["medium_light_threshold_input_psnr_db"])

    val_set = GoProPairDataset(
        args.data_root,
        split="train",
        patch_size=int(baseline_args["val_patch_size"]),
        augment=False,
        max_pairs=args.val_pairs or None,
        include_recording_groups=validation_groups,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed + 2),
    )

    scales = tuple(int(value) for value in adaptive_args.get("scales", (1, 3, 7)))
    block = MultiScaleNonlocalBlock(
        channels=int(baseline_args["width"]) * 4,
        scales=scales,
        top_k=int(adaptive_args.get("top_k", 6)),
        statistics_window=int(adaptive_args.get("statistics_window", 5)),
        mode="adaptive",
    )
    model = CompactDeblurUNet(
        width=int(baseline_args["width"]),
        bottleneck_module=block,
    )
    model.load_state_dict(adaptive_checkpoint["model"], strict=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float | str]] = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for batch in tqdm(val_loader, desc="P12 condition interventions"):
        blurry = batch["blur"].to(device, non_blocking=True)
        sharp = batch["sharp"].to(device, non_blocking=True)
        name = str(batch["name"][0])
        scene = name.split("/", maxsplit=1)[0]
        input_psnr = psnr_value(blurry, sharp)
        record: dict[str, float | str] = {
            "recording_group": recording_group(scene),
            "scene": scene,
            "name": name,
            "strength_class": strength_class(input_psnr, heavy_max, light_min),
            "input_psnr_db": input_psnr,
        }
        with torch.inference_mode():
            for intervention in INTERVENTIONS:
                with apply_condition_intervention(block, intervention):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        prediction = model(blurry)
                record[f"{intervention}_psnr_db"] = psnr_value(prediction, sharp)
                record[f"{intervention}_ssim"] = ssim_value(prediction[0], sharp[0])
        records.append(record)

    overall = aggregate(records)
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    stratified: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for record in records:
        grouped[str(record["recording_group"])].append(record)
        stratified[str(record["strength_class"])].append(record)
    per_group = [
        flatten_aggregate("recording_group", name, aggregate(items))
        for name, items in sorted(grouped.items())
    ]
    per_strength = [
        flatten_aggregate("strength_class", name, aggregate(stratified[name]))
        for name in ("heavy", "medium", "light")
        if stratified[name]
    ]
    write_csv(output / "per_image_metrics.csv", records)
    write_csv(output / "per_group_metrics.csv", per_group)
    write_csv(output / "per_strength_metrics.csv", per_strength)

    variants = overall["interventions"]
    assert isinstance(variants, dict)
    scale_drop = float(variants["no_scale_adaptation"]["full_minus_intervention_psnr_db"])
    gate_drop = float(variants["no_fusion_gate"]["full_minus_intervention_psnr_db"])
    both_drop = float(variants["no_scale_or_gate"]["full_minus_intervention_psnr_db"])
    gates = {
        "single_component_threshold_db": 0.005,
        "combined_threshold_db": 0.01,
        "scale_adaptation_supported": scale_drop >= 0.005,
        "fusion_gate_supported": gate_drop >= 0.005,
        "combined_conditioning_supported": both_drop >= 0.01,
    }
    gates["mechanism_intervention_gate_passed"] = all(
        bool(gates[key])
        for key in (
            "scale_adaptation_supported",
            "fusion_gate_supported",
            "combined_conditioning_supported",
        )
    )

    source_summary_path = adaptive_path.parent / "summary.json"
    source_summary_match: bool | None = None
    source_psnr_error_db: float | None = None
    source_ssim_error: float | None = None
    if args.val_pairs == 0 and source_summary_path.is_file():
        source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
        source_psnr_error_db = float(variants["adaptive_full"]["psnr_db"]) - float(
            source_summary["candidate_psnr_db"]
        )
        source_ssim_error = float(variants["adaptive_full"]["ssim"]) - float(
            source_summary["candidate_ssim"]
        )
        source_summary_match = abs(source_psnr_error_db) <= 5e-6 and abs(source_ssim_error) <= 5e-7
        if not source_summary_match:
            raise RuntimeError(
                "full adaptive rerun does not reproduce the frozen P12-D summary: "
                f"psnr_error={source_psnr_error_db}, ssim_error={source_ssim_error}"
            )

    report = {
        "experiment": output.name,
        "stage": "P12 inference-only condition intervention audit",
        "seed": args.seed,
        "checkpoint": str(adaptive_path),
        "checkpoint_step": int(adaptive_checkpoint["step"]),
        "data_root": str(args.data_root.expanduser().resolve()),
        "split": "train-internal frozen development groups",
        "validation_recording_groups": list(validation_groups),
        "pairs": len(records),
        "validation_patch_size": int(baseline_args["val_patch_size"]),
        "intervention_definition": {
            "adaptive_full": "learned scale logits and learned local--nonlocal gate",
            "no_scale_adaptation": "scale logits set to zero (uniform priors); learned gate retained",
            "no_fusion_gate": "gate logit set to zero (gate=0.5); learned scale priors retained",
            "no_scale_or_gate": "uniform scale priors and gate=0.5",
        },
        "overall": overall,
        "per_group": per_group,
        "per_strength": per_strength,
        "gates": gates,
        "source_summary_match": source_summary_match,
        "source_psnr_error_db": source_psnr_error_db,
        "source_ssim_error": source_ssim_error,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "wall_seconds": time.perf_counter() - started,
        "inference_only_intervention": True,
        "off_distribution_intervention": True,
        "official_test_accessed": False,
        "paper_result": False,
        "note": (
            "This audit measures reliance of one trained P12-D checkpoint on its condition "
            "outputs. It does not replace separately trained matched ablations."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
