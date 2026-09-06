"""P7-B: strength-stratified trajectory specialization with B0 protection."""

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
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GoProPairDataset, recording_group  # noqa: E402
from src.models import (  # noqa: E402
    CompactDeblurUNet,
    TrajectoryNonlocalBlock,
    blend_with_baseline,
    deployable_strength_gate,
    input_psnr_per_image,
    teacher_blur_strength,
)


DEFAULT_BASELINE_RUN = PROJECT_ROOT / "results" / "runs" / "P3_baseline_group_split_v0"
DEFAULT_INITIALIZATION_RUN = (
    PROJECT_ROOT / "results" / "runs" / "P6F_aggregation_adaptation_v0"
)
DEFAULT_CALIBRATION_RUN = (
    PROJECT_ROOT / "results" / "runs" / "P7A_blur_strength_calibration_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly adapt trajectory geometry and aggregation with blur-stratified safety."
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "GOPRO_Large")
    parser.add_argument(
        "--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_RUN / "best.pt"
    )
    parser.add_argument(
        "--initialization-checkpoint",
        type=Path,
        default=DEFAULT_INITIALIZATION_RUN / "best.pt",
    )
    parser.add_argument(
        "--calibration-summary",
        type=Path,
        default=DEFAULT_CALIBRATION_RUN / "summary.json",
    )
    parser.add_argument(
        "--strength-manifest",
        type=Path,
        default=DEFAULT_CALIBRATION_RUN / "per_image_strength.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "P7B_strength_stratified_joint_v0",
    )
    parser.add_argument(
        "--candidate-variant",
        choices=("trajectory", "local"),
        default="trajectory",
    )
    parser.add_argument(
        "--trajectory-initialization",
        choices=("checkpoint", "identity_from_baseline"),
        default="checkpoint",
    )
    parser.add_argument(
        "--trainable-scope",
        choices=("bottleneck", "all"),
        default="bottleneck",
    )
    parser.add_argument("--trajectory-samples", type=int, default=5)
    parser.add_argument("--max-feature-length", type=float, default=7.0)
    parser.add_argument(
        "--trajectory-fusion",
        choices=("aggregate", "detail"),
        default="detail",
    )
    parser.add_argument("--direction-residual-scale", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--train-pairs", type=int, default=0)
    parser.add_argument("--val-pairs", type=int, default=0)
    parser.add_argument("--train-patch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--frequency-weight", type=float, default=0.1)
    parser.add_argument("--heavy-endpoint-weight", type=float, default=0.5)
    parser.add_argument("--light-safety-weight", type=float, default=0.5)
    parser.add_argument("--val-interval", type=int, default=375)
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


def _psnr_value(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = (prediction.float().clamp(0, 1) - target.float().clamp(0, 1)).square().mean()
    return float(-10.0 * mse.clamp_min(1e-12).log10())


def _ssim_value(prediction: torch.Tensor, target: torch.Tensor) -> float:
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


def _per_image_restoration_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    frequency_weight: float,
) -> torch.Tensor:
    charbonnier = torch.sqrt((prediction.float() - target.float()).square() + 1e-6)
    charbonnier = charbonnier.flatten(1).mean(1)
    prediction_fft = torch.fft.rfft2(prediction.float(), norm="ortho")
    target_fft = torch.fft.rfft2(target.float(), norm="ortho")
    frequency = (prediction_fft - target_fft).abs().flatten(1).mean(1)
    return charbonnier + frequency_weight * frequency


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.float().clamp_min(0.0)
    return (values.float() * weights).sum() / weights.sum().clamp_min(1e-6)


def _write_csv(path: Path, records: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _load_strength_manifest(path: Path) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        manifest = {str(row["name"]): float(row["input_psnr_db"]) for row in rows}
    if not manifest:
        raise RuntimeError(f"empty strength manifest: {path}")
    return manifest


def _strength_class(input_psnr_db: float, heavy_max: float, light_min: float) -> str:
    if input_psnr_db <= heavy_max:
        return "heavy"
    if input_psnr_db >= light_min:
        return "light"
    return "medium"


def _make_balanced_sampler(
    dataset: GoProPairDataset,
    manifest: dict[str, float],
    heavy_max: float,
    light_min: float,
    seed: int,
) -> tuple[WeightedRandomSampler, dict[str, int]]:
    labels = []
    for pair in dataset.pairs:
        name = f"{pair.scene}/{pair.name}"
        if name not in manifest:
            raise KeyError(f"training pair missing from P7-A strength manifest: {name}")
        labels.append(_strength_class(manifest[name], heavy_max, light_min))
    counts = {name: labels.count(name) for name in ("heavy", "medium", "light")}
    if any(count == 0 for count in counts.values()):
        raise RuntimeError(f"strength stratum is empty: {counts}")
    weights = torch.tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return sampler, counts


def _make_dynamic_model(
    checkpoint: dict[str, object],
    baseline_state: dict[str, torch.Tensor],
    width: int,
    candidate_variant: str,
    trajectory_initialization: str,
    trainable_scope: str,
    trajectory_config: dict[str, object],
) -> tuple[CompactDeblurUNet, dict[str, object]]:
    if candidate_variant == "local":
        if trainable_scope != "all":
            raise ValueError("candidate-variant=local requires trainable-scope=all")
        model = CompactDeblurUNet(width=width)
        model.load_state_dict(baseline_state, strict=True)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return model, {
            "initialization_args": {},
            "oracle_args": {},
            "initialization_source": "baseline_checkpoint",
        }

    initialization_args = dict(checkpoint.get("args", {}))
    oracle_args = dict(checkpoint.get("oracle_args", {}))
    if trajectory_initialization == "checkpoint":
        required = ("trajectory_samples", "max_feature_length", "trajectory_fusion")
        if any(name not in oracle_args for name in required):
            raise ValueError("initialization checkpoint is missing trajectory configuration")
        samples = int(oracle_args["trajectory_samples"])
        max_feature_length = float(oracle_args["max_feature_length"])
        fusion_mode = str(oracle_args["trajectory_fusion"])
        direction_residual_scale = float(
            initialization_args.get("direction_residual_scale", 2.0)
        )
    else:
        samples = int(trajectory_config["trajectory_samples"])
        max_feature_length = float(trajectory_config["max_feature_length"])
        fusion_mode = str(trajectory_config["trajectory_fusion"])
        direction_residual_scale = float(trajectory_config["direction_residual_scale"])
        oracle_args = {
            "trajectory_samples": samples,
            "max_feature_length": max_feature_length,
            "trajectory_fusion": fusion_mode,
        }
    block = TrajectoryNonlocalBlock(
        channels=width * 4,
        samples=samples,
        max_feature_length=max_feature_length,
        fusion_mode=fusion_mode,
        angle_hypotheses=1,
        direction_residual_scale=direction_residual_scale,
    )
    model = CompactDeblurUNet(width=width, bottleneck_module=block)
    if trajectory_initialization == "checkpoint":
        model.load_state_dict(checkpoint["model"], strict=True)
        initialization_source = "trajectory_checkpoint"
    else:
        incompatible = model.load_state_dict(baseline_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"baseline initialization has unexpected tensors: {incompatible.unexpected_keys}"
            )
        expected_missing = set(block.state_dict().keys())
        actual_missing = {
            name.removeprefix("bottleneck_module.") for name in incompatible.missing_keys
        }
        if actual_missing != expected_missing:
            raise RuntimeError(
                "identity trajectory initialization has an unexpected missing-key set"
            )
        initialization_source = "identity_trajectory_on_baseline"

    for parameter in model.parameters():
        parameter.requires_grad_(trainable_scope == "all")
    if trainable_scope == "bottleneck":
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    return model, {
        "initialization_args": initialization_args,
        "oracle_args": oracle_args,
        "initialization_source": initialization_source,
    }


@torch.no_grad()
def evaluate(
    baseline: CompactDeblurUNet,
    model: CompactDeblurUNet,
    loader: DataLoader,
    light_cut_log_l1: float,
    heavy_cut_log_l1: float,
    heavy_max_psnr_db: float,
    light_min_psnr_db: float,
    detailed: bool = False,
) -> tuple[
    dict[str, float | int],
    list[dict[str, float | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    baseline.eval()
    model.eval()
    records: list[dict[str, float | str]] = []
    for batch in loader:
        blurry = batch["blur"].cuda(non_blocking=True)
        sharp = batch["sharp"].cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            baseline_output = baseline(blurry)
            dynamic_output = model(blurry)
            gate, proxy = deployable_strength_gate(
                blurry,
                baseline_output,
                light_cut_log_l1,
                heavy_cut_log_l1,
            )
            final_output = blend_with_baseline(baseline_output, dynamic_output, gate)
        name = str(batch["name"][0])
        scene = name.split("/", maxsplit=1)[0]
        input_psnr = _psnr_value(blurry, sharp)
        baseline_psnr = _psnr_value(baseline_output, sharp)
        dynamic_psnr = _psnr_value(dynamic_output, sharp)
        final_psnr = _psnr_value(final_output, sharp)
        record: dict[str, float | str] = {
            "recording_group": recording_group(scene),
            "scene": scene,
            "name": name,
            "strength_class": _strength_class(
                input_psnr,
                heavy_max_psnr_db,
                light_min_psnr_db,
            ),
            "input_psnr_db": input_psnr,
            "baseline_psnr_db": baseline_psnr,
            "dynamic_psnr_db": dynamic_psnr,
            "final_psnr_db": final_psnr,
            "dynamic_minus_baseline_db": dynamic_psnr - baseline_psnr,
            "final_minus_baseline_db": final_psnr - baseline_psnr,
            "gate": float(gate[0, 0, 0, 0]),
            "baseline_correction_l1": float(proxy[0].exp()),
        }
        if detailed:
            baseline_ssim = _ssim_value(baseline_output[0], sharp[0])
            dynamic_ssim = _ssim_value(dynamic_output[0], sharp[0])
            final_ssim = _ssim_value(final_output[0], sharp[0])
            record.update(
                {
                    "baseline_ssim": baseline_ssim,
                    "dynamic_ssim": dynamic_ssim,
                    "final_ssim": final_ssim,
                    "dynamic_minus_baseline_ssim": dynamic_ssim - baseline_ssim,
                    "final_minus_baseline_ssim": final_ssim - baseline_ssim,
                }
            )
        records.append(record)

    numeric_fields = [
        "input_psnr_db",
        "baseline_psnr_db",
        "dynamic_psnr_db",
        "final_psnr_db",
        "dynamic_minus_baseline_db",
        "final_minus_baseline_db",
        "gate",
        "baseline_correction_l1",
    ]
    if detailed:
        numeric_fields.extend(
            [
                "baseline_ssim",
                "dynamic_ssim",
                "final_ssim",
                "dynamic_minus_baseline_ssim",
                "final_minus_baseline_ssim",
            ]
        )

    def aggregate(items: list[dict[str, float | str]]) -> dict[str, float | int]:
        return {
            "pairs": len(items),
            **{
                field: statistics.fmean(float(item[field]) for item in items)
                for field in numeric_fields
            },
        }

    overall = aggregate(records)
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    stratified: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for record in records:
        grouped[str(record["recording_group"])].append(record)
        stratified[str(record["strength_class"])].append(record)
    per_group = [
        {"recording_group": name, **aggregate(items)}
        for name, items in sorted(grouped.items())
    ]
    per_strength = [
        {"strength_class": name, **aggregate(stratified[name])}
        for name in ("heavy", "medium", "light")
        if stratified[name]
    ]
    return overall, records, per_group, per_strength


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; P7-B intentionally has no CPU fallback")
    if min(args.steps, args.batch_size, args.train_patch_size, args.val_interval) <= 0:
        raise ValueError("steps, batch-size, train-patch-size, and val-interval must be positive")
    if args.train_pairs < 0 or args.val_pairs < 0 or args.num_workers < 0:
        raise ValueError("train-pairs, val-pairs, and num-workers must be nonnegative")
    if min(
        args.learning_rate,
        args.frequency_weight,
        args.heavy_endpoint_weight,
        args.light_safety_weight,
    ) < 0:
        raise ValueError("loss weights and learning rate must be nonnegative")
    if args.candidate_variant == "local" and args.trajectory_initialization != "checkpoint":
        raise ValueError(
            "trajectory-initialization applies only to candidate-variant=trajectory"
        )

    set_determinism(args.seed)
    device = torch.device("cuda:0")
    baseline_path = args.baseline_checkpoint.expanduser().resolve()
    initialization_path = args.initialization_checkpoint.expanduser().resolve()
    calibration_path = args.calibration_summary.expanduser().resolve()
    manifest_path = args.strength_manifest.expanduser().resolve()
    for description, path in (
        ("baseline checkpoint", baseline_path),
        ("initialization checkpoint", initialization_path),
        ("calibration summary", calibration_path),
        ("strength manifest", manifest_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")

    baseline_checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=False)
    initialization_checkpoint = torch.load(
        initialization_path,
        map_location="cpu",
        weights_only=False,
    )
    baseline_args = dict(baseline_checkpoint.get("args", {}))
    if baseline_args.get("variant") != "local":
        raise ValueError("baseline checkpoint must use variant=local")
    if baseline_args.get("validation_source") != "train_holdout":
        raise ValueError("baseline checkpoint must use validation_source=train_holdout")
    validation_groups = tuple(baseline_args.get("validation_groups") or ())
    if not validation_groups:
        raise ValueError("baseline checkpoint does not contain frozen validation groups")

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("official_test_accessed") is not False:
        raise ValueError("P7-A calibration must explicitly state official_test_accessed=false")
    if calibration.get("recording_group_overlap_with_frozen_development"):
        raise ValueError("P7-A calibration reports frozen-development leakage")
    heavy_max_psnr_db = float(calibration["heavy_medium_threshold_input_psnr_db"])
    light_min_psnr_db = float(calibration["medium_light_threshold_input_psnr_db"])
    gate_calibration = calibration["correction_gate_calibration"]
    light_cut_log_l1 = float(gate_calibration["light_cut_log_l1"])
    heavy_cut_log_l1 = float(gate_calibration["heavy_cut_log_l1"])
    manifest = _load_strength_manifest(manifest_path)

    train_set = GoProPairDataset(
        args.data_root,
        split="train",
        patch_size=args.train_patch_size,
        augment=True,
        max_pairs=args.train_pairs or None,
        exclude_recording_groups=validation_groups,
    )
    val_set = GoProPairDataset(
        args.data_root,
        split="train",
        patch_size=int(baseline_args["val_patch_size"]),
        augment=False,
        max_pairs=args.val_pairs or None,
        include_recording_groups=validation_groups,
    )
    if set(train_set.recording_groups) & set(val_set.recording_groups):
        raise RuntimeError("training/frozen-development recording-group leakage detected")
    sampler, strength_counts = _make_balanced_sampler(
        train_set,
        manifest,
        heavy_max_psnr_db,
        light_min_psnr_db,
        args.seed,
    )
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
        **loader_options,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        generator=torch.Generator().manual_seed(args.seed + 2),
        **loader_options,
    )

    width = int(baseline_args["width"])
    baseline = CompactDeblurUNet(width=width)
    baseline.load_state_dict(baseline_checkpoint["model"], strict=True)
    baseline = baseline.to(device).eval()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)
    model, initialization_metadata = _make_dynamic_model(
        initialization_checkpoint,
        baseline_checkpoint["model"],
        width,
        args.candidate_variant,
        args.trajectory_initialization,
        args.trainable_scope,
        {
            "trajectory_samples": args.trajectory_samples,
            "max_feature_length": args.max_feature_length,
            "trajectory_fusion": args.trajectory_fusion,
            "direction_residual_scale": args.direction_residual_scale,
        },
    )
    model = model.to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_parameter_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("no candidate parameters were enabled")
    if args.trainable_scope == "bottleneck" and any(
        not name.startswith("bottleneck_module.") for name in trainable_parameter_names
    ):
        raise RuntimeError("a non-trajectory parameter was accidentally enabled")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    initial, _, initial_groups, initial_strength = evaluate(
        baseline,
        model,
        val_loader,
        light_cut_log_l1,
        heavy_cut_log_l1,
        heavy_max_psnr_db,
        light_min_psnr_db,
    )
    should_start_as_baseline = (
        args.candidate_variant == "local"
        or args.trajectory_initialization == "identity_from_baseline"
    )
    if should_start_as_baseline and abs(float(initial["dynamic_minus_baseline_db"])) > 1e-8:
        raise RuntimeError(
            "P8 fairness audit failed: candidate does not start as the exact B0 function"
        )
    metadata = {
        "args": vars(args),
        "baseline_checkpoint": str(baseline_path),
        "initialization_checkpoint": str(initialization_path),
        "calibration_summary": str(calibration_path),
        "strength_manifest": str(manifest_path),
        **initialization_metadata,
    }
    torch.save({"step": 0, "model": model.state_dict(), **metadata}, output / "best.pt")
    best_psnr = float(initial["final_psnr_db"])
    history: list[dict[str, float | int]] = [
        {
            "step": 0,
            "total_train_loss": float("nan"),
            "gated_train_loss": float("nan"),
            "heavy_endpoint_train_loss": float("nan"),
            "light_safety_train_loss": float("nan"),
            "mean_teacher_strength": float("nan"),
            "mean_deployable_gate": float("nan"),
            "baseline_psnr_db": float(initial["baseline_psnr_db"]),
            "dynamic_psnr_db": float(initial["dynamic_psnr_db"]),
            "final_psnr_db": float(initial["final_psnr_db"]),
            "final_minus_baseline_db": float(initial["final_minus_baseline_db"]),
            "minimum_group_final_minus_baseline_db": min(
                float(group["final_minus_baseline_db"]) for group in initial_groups
            ),
            "learning_rate": args.learning_rate,
        }
    ]

    train_iterator = iter(train_loader)
    last_losses = {
        "total": float("nan"),
        "gated": float("nan"),
        "heavy": float("nan"),
        "safety": float("nan"),
        "teacher_strength": float("nan"),
        "gate": float("nan"),
    }
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    progress = tqdm(
        range(1, args.steps + 1),
        desc=f"strength-stratified {args.candidate_variant}/{args.trainable_scope}",
    )
    for step in progress:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        blurry = batch["blur"].to(device, non_blocking=True)
        sharp = batch["sharp"].to(device, non_blocking=True)
        baseline.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            baseline_output = baseline(blurry)
            gate, _ = deployable_strength_gate(
                blurry,
                baseline_output,
                light_cut_log_l1,
                heavy_cut_log_l1,
            )
            paired_psnr = input_psnr_per_image(blurry, sharp)
            teacher_strength = teacher_blur_strength(
                paired_psnr,
                heavy_max_psnr_db,
                light_min_psnr_db,
            )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            dynamic_output = model(blurry)
            final_output = blend_with_baseline(baseline_output, dynamic_output, gate)
            gated_per_image = _per_image_restoration_loss(
                final_output,
                sharp,
                args.frequency_weight,
            )
            dynamic_per_image = _per_image_restoration_loss(
                dynamic_output,
                sharp,
                args.frequency_weight,
            )
            safety_per_image = (dynamic_output.float() - baseline_output.float()).abs().flatten(1).mean(1)
            gated_loss = gated_per_image.mean()
            heavy_endpoint_loss = _weighted_mean(dynamic_per_image, teacher_strength)
            light_safety_loss = _weighted_mean(safety_per_image, 1.0 - teacher_strength)
            total = (
                gated_loss
                + args.heavy_endpoint_weight * heavy_endpoint_loss
                + args.light_safety_weight * light_safety_loss
            )
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        last_losses = {
            "total": float(total.detach()),
            "gated": float(gated_loss.detach()),
            "heavy": float(heavy_endpoint_loss.detach()),
            "safety": float(light_safety_loss.detach()),
            "teacher_strength": float(teacher_strength.mean()),
            "gate": float(gate.mean()),
        }
        if step == 1 or step % 20 == 0:
            progress.set_postfix(
                gated=f"{last_losses['gated']:.4f}",
                heavy=f"{last_losses['heavy']:.4f}",
                safe=f"{last_losses['safety']:.4f}",
            )

        if step % args.val_interval == 0 or step == args.steps:
            validation, _, groups, _ = evaluate(
                baseline,
                model,
                val_loader,
                light_cut_log_l1,
                heavy_cut_log_l1,
                heavy_max_psnr_db,
                light_min_psnr_db,
            )
            history.append(
                {
                    "step": step,
                    "total_train_loss": last_losses["total"],
                    "gated_train_loss": last_losses["gated"],
                    "heavy_endpoint_train_loss": last_losses["heavy"],
                    "light_safety_train_loss": last_losses["safety"],
                    "mean_teacher_strength": last_losses["teacher_strength"],
                    "mean_deployable_gate": last_losses["gate"],
                    "baseline_psnr_db": float(validation["baseline_psnr_db"]),
                    "dynamic_psnr_db": float(validation["dynamic_psnr_db"]),
                    "final_psnr_db": float(validation["final_psnr_db"]),
                    "final_minus_baseline_db": float(validation["final_minus_baseline_db"]),
                    "minimum_group_final_minus_baseline_db": min(
                        float(group["final_minus_baseline_db"]) for group in groups
                    ),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            print(
                f"\nstep={step:05d} B0={validation['baseline_psnr_db']:.4f} dB "
                f"dynamic={validation['dynamic_psnr_db']:.4f} dB "
                f"final={validation['final_psnr_db']:.4f} dB "
                f"final_vs_B0={validation['final_minus_baseline_db']:+.4f} dB "
                f"gate={validation['gate']:.3f}"
            )
            if float(validation["final_psnr_db"]) > best_psnr:
                best_psnr = float(validation["final_psnr_db"])
                torch.save(
                    {"step": step, "model": model.state_dict(), **metadata},
                    output / "best.pt",
                )

    _write_csv(output / "metrics.csv", history)
    torch.save({"step": args.steps, "model": model.state_dict(), **metadata}, output / "last.pt")
    best_checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"], strict=True)
    final, per_image, per_group, per_strength = evaluate(
        baseline,
        model,
        val_loader,
        light_cut_log_l1,
        heavy_cut_log_l1,
        heavy_max_psnr_db,
        light_min_psnr_db,
        detailed=True,
    )
    _write_csv(output / "per_image_metrics.csv", per_image)
    _write_csv(output / "per_group_metrics.csv", per_group)
    _write_csv(output / "per_strength_metrics.csv", per_strength)

    minimum_group_delta = min(float(group["final_minus_baseline_db"]) for group in per_group)
    strength_lookup = {str(row["strength_class"]): row for row in per_strength}
    heavy_delta = float(strength_lookup["heavy"]["final_minus_baseline_db"])
    light_delta = float(strength_lookup["light"]["final_minus_baseline_db"])
    psnr_passed = float(final["final_minus_baseline_db"]) >= 0.075
    ssim_passed = float(final["final_minus_baseline_ssim"]) >= 0.0
    group_passed = minimum_group_delta >= -0.01
    heavy_passed = heavy_delta >= 0.05
    light_protection_passed = light_delta >= -0.005
    report = {
        "experiment": output.name,
        "stage": (
            "P8 end-to-end matched local control"
            if args.candidate_variant == "local"
            else (
                "P8 end-to-end identity-initialized trajectory candidate"
                if args.trainable_scope == "all"
                else "P7 strength-stratified frozen-backbone trajectory specialization"
            )
        ),
        "baseline_checkpoint": str(baseline_path),
        "baseline_checkpoint_step": int(baseline_checkpoint["step"]),
        "initialization_checkpoint": str(initialization_path),
        "initialization_checkpoint_step": int(initialization_checkpoint["step"]),
        "calibration_summary": str(calibration_path),
        "candidate_variant": args.candidate_variant,
        "trajectory_initialization": args.trajectory_initialization,
        "trainable_scope": args.trainable_scope,
        "starts_as_exact_baseline": should_start_as_baseline,
        "seed": args.seed,
        "steps": args.steps,
        "best_checkpoint_step": int(best_checkpoint["step"]),
        "train_pairs": len(train_set),
        "val_pairs": len(val_set),
        "train_patch_size": args.train_patch_size,
        "validation_patch_size": int(baseline_args["val_patch_size"]),
        "training_strength_counts_before_balancing": strength_counts,
        "sampler": "inverse-frequency three-stratum weighted sampling with replacement",
        "heavy_max_input_psnr_db": heavy_max_psnr_db,
        "light_min_input_psnr_db": light_min_psnr_db,
        "light_cut_log_correction_l1": light_cut_log_l1,
        "heavy_cut_log_correction_l1": heavy_cut_log_l1,
        "heavy_endpoint_weight": args.heavy_endpoint_weight,
        "light_safety_weight": args.light_safety_weight,
        "frequency_weight": args.frequency_weight,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
        "trainable_parameter_names": trainable_parameter_names,
        **final,
        "per_group": per_group,
        "per_strength": per_strength,
        "minimum_group_final_minus_baseline_db": minimum_group_delta,
        "heavy_final_minus_baseline_db": heavy_delta,
        "light_final_minus_baseline_db": light_delta,
        "images_final_better_than_baseline": sum(
            float(record["final_minus_baseline_db"]) > 0 for record in per_image
        ),
        "predicted_over_baseline_threshold_db": 0.075,
        "minimum_group_threshold_db": -0.01,
        "heavy_improvement_threshold_db": 0.05,
        "light_protection_threshold_db": -0.005,
        "psnr_gate_passed": psnr_passed,
        "ssim_gate_passed": ssim_passed,
        "group_gate_passed": group_passed,
        "heavy_gate_passed": heavy_passed,
        "light_protection_gate_passed": light_protection_passed,
        "passed": (
            psnr_passed
            and ssim_passed
            and group_passed
            and heavy_passed
            and light_protection_passed
        ),
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "wall_seconds": time.perf_counter() - started,
        "teacher_uses_sharp_during_training": True,
        "final_branch_uses_sharp_at_inference": False,
        "official_test_accessed": False,
        "paper_result": False,
        "note": (
            "Development-holdout screening experiment. Sharp images define training-only "
            "severity supervision; deployed gating uses only blurry input and frozen B0 output."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
