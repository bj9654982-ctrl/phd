"""P12 matched screening for fixed, Top-K, and blur-aware nonlocal features."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GoProPairDataset, recording_group  # noqa: E402
from src.losses import restoration_loss  # noqa: E402
from src.models import (  # noqa: E402
    CompactDeblurUNet,
    MultiScaleNonlocalBlock,
    input_psnr_per_image,
    teacher_blur_strength,
)


DEFAULT_BASELINE_RUN = PROJECT_ROOT / "results" / "runs" / "P3_baseline_group_split_v0"
DEFAULT_CALIBRATION_RUN = PROJECT_ROOT / "results" / "runs" / "P7A_blur_strength_calibration_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one P12 variant under a common GoPro protocol."
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "GOPRO_Large")
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=DEFAULT_BASELINE_RUN / "best.pt",
    )
    parser.add_argument(
        "--calibration-summary",
        type=Path,
        default=DEFAULT_CALIBRATION_RUN / "summary.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "local",
            "fixed",
            "topk",
            "scale_adaptive",
            "gate_adaptive",
            "protected_gate_adaptive",
            "adaptive",
        ),
        required=True,
    )
    parser.add_argument("--scales", type=int, nargs="+", default=(1, 3, 7))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--statistics-window", type=int, default=5)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--train-pairs", type=int, default=0)
    parser.add_argument("--val-pairs", type=int, default=0)
    parser.add_argument("--train-patch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--frequency-weight", type=float, default=0.1)
    parser.add_argument(
        "--gate-supervision-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only paired blur-strength supervision for "
            "gate_adaptive or protected_gate_adaptive; deployment remains blurry-only"
        ),
    )
    parser.add_argument("--val-interval", type=int, default=375)
    parser.add_argument("--teacher-mode", choices=("paired", "shuffled"), default="paired")
    parser.add_argument("--teacher-seed", type=int, default=None,
                        help="Shuffled-only private RNG seed; defaults to training seed + 19000")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--sampling",
        choices=("image_uniform", "group_balanced"),
        default="image_uniform",
        help=(
            "image_uniform shuffles all pairs equally; group_balanced samples recording "
            "groups equally and then samples pairs uniformly inside each group"
        ),
    )
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def validate_gate_supervision(variant: str, weight: float) -> None:
    """Keep the outer residual gate and training-only teacher independent."""
    if weight < 0:
        raise ValueError("gate-supervision-weight must be nonnegative")
    if weight > 0 and variant not in {"gate_adaptive", "protected_gate_adaptive"}:
        raise ValueError("gate supervision requires a learned nonlocal gate")


def compute_gate_supervision_loss(
    geometry: dict[str, torch.Tensor],
    blurry: torch.Tensor,
    sharp: torch.Tensor,
    heavy_max_psnr_db: float,
    light_min_psnr_db: float,
    *,
    teacher_mode: str = "paired",
    teacher_generator: torch.Generator | None = None,
    diagnostics: dict | None = None,
) -> torch.Tensor:
    teacher = teacher_blur_strength(
        input_psnr_per_image(blurry, sharp), heavy_max_psnr_db, light_min_psnr_db
    )
    target, shift = assign_teacher_targets(teacher, teacher_mode, teacher_generator)
    # intervention_gate is constant one without the outer gate; supervise the
    # learned inner gate in both modes. In protected mode these are the same g.
    predicted = geometry["nonlocal_gate"].float().mean(dim=(1, 2, 3))
    loss = F.mse_loss(predicted, target)
    if diagnostics is not None:
        values = teacher.detach().cpu().tolist()
        targets = target.detach().cpu().tolist()
        differences = [abs(a - b) for a, b in zip(values, targets)]
        diagnostics.update({
            "shift": shift,
            "target_changed_fraction": statistics.fmean(a != b for a, b in zip(values, targets)),
            "target_mean_absolute_change": statistics.fmean(differences),
            "teacher_mean": statistics.fmean(values),
            "teacher_std": statistics.pstdev(values),
            "teacher_unique_values": len(set(values)),
            "teacher_endpoint_fraction": statistics.fmean(t in (0.0, 1.0) for t in values),
            **{f"teacher_{i}": value for i, value in enumerate(values)},
        })
    return loss


def assign_teacher_targets(
    teacher: torch.Tensor, mode: str, generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, int]:
    """Preserve target values; shuffle only their batch positions, never sharp images."""
    if mode == "paired":
        return teacher, 0
    if mode != "shuffled":
        raise ValueError("teacher mode must be paired or shuffled")
    if teacher.ndim != 1 or teacher.numel() < 2:
        raise ValueError("shuffled teacher needs at least two scalar batch targets")
    if generator is None:
        raise ValueError("shuffled teacher requires its own explicit generator")
    shift = int(torch.randint(1, teacher.numel(), (1,), generator=generator,
                              device=generator.device).item())
    # Negative roll implements target[i] = teacher[(i + shift) mod B].
    return teacher.roll(-shift, dims=0), shift


def compute_training_objective(
    candidate, blurry, sharp, heavy_max, light_min, frequency_weight, supervision_weight,
    *, teacher_mode="paired", teacher_generator=None, diagnostics=None,
):
    """Historical objective with an opt-in target-assignment negative control."""
    if supervision_weight > 0:
        prediction, geometry = candidate(blurry, return_geometry=True)
        gate_loss = compute_gate_supervision_loss(
            geometry, blurry, sharp, heavy_max, light_min, teacher_mode=teacher_mode,
            teacher_generator=teacher_generator, diagnostics=diagnostics,
        )
    else:
        prediction = candidate(blurry)
        gate_loss = prediction.new_zeros(())
    restoration_objective = restoration_loss(prediction, sharp, frequency_weight=frequency_weight)
    loss = restoration_objective + supervision_weight * gate_loss
    return loss, restoration_objective, gate_loss


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


def _strength_class(input_psnr_db: float, heavy_max: float, light_min: float) -> str:
    if input_psnr_db <= heavy_max:
        return "heavy"
    if input_psnr_db >= light_min:
        return "light"
    return "medium"


def _write_csv(path: Path, records: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _make_group_balanced_sampler(
    dataset: GoProPairDataset,
    seed: int,
) -> tuple[WeightedRandomSampler, dict[str, int]]:
    """Sample recording groups equally without changing the epoch length.

    A pair from group ``g`` receives weight ``1 / n_g``.  Consequently the
    total probability mass of every recording group is identical, while pairs
    remain uniform within a group.  The dedicated generator makes the sampled
    index sequence reproducible and identical across matched variants.
    """
    group_names = [recording_group(pair.scene) for pair in dataset.pairs]
    counts = Counter(group_names)
    if not counts:
        raise RuntimeError("cannot balance an empty training dataset")
    weights = torch.tensor(
        [1.0 / counts[group_name] for group_name in group_names],
        dtype=torch.double,
    )
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return sampler, dict(sorted(counts.items()))


def _make_candidate(
    baseline_state: dict[str, torch.Tensor],
    width: int,
    variant: str,
    scales: tuple[int, ...],
    top_k: int,
    statistics_window: int,
) -> CompactDeblurUNet:
    if variant == "local":
        model = CompactDeblurUNet(width=width)
        model.load_state_dict(baseline_state, strict=True)
        return model

    block = MultiScaleNonlocalBlock(
        channels=width * 4,
        scales=scales,
        top_k=top_k,
        statistics_window=statistics_window,
        mode=variant,
    )
    model = CompactDeblurUNet(width=width, bottleneck_module=block)
    incompatible = model.load_state_dict(baseline_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"baseline initialization has unexpected tensors: {incompatible.unexpected_keys}"
        )
    expected_missing = {
        f"bottleneck_module.{name}" for name in block.state_dict().keys()
    }
    actual_missing = set(incompatible.missing_keys)
    if actual_missing != expected_missing:
        raise RuntimeError(
            "identity nonlocal initialization has an unexpected missing-key set: "
            f"missing={sorted(actual_missing ^ expected_missing)}"
        )
    return model


@torch.no_grad()
def _audit_exact_initialization(
    baseline: CompactDeblurUNet,
    candidate: CompactDeblurUNet,
    loader: DataLoader,
    device: torch.device,
) -> float:
    batch = next(iter(loader))
    blurry = batch["blur"].to(device, non_blocking=True)
    baseline_output = baseline(blurry)
    candidate_output = candidate(blurry)
    maximum_error = float((candidate_output - baseline_output).abs().max())
    if maximum_error != 0.0:
        raise RuntimeError(
            "P12 fairness audit failed: candidate does not start as the exact B0 function; "
            f"maximum error={maximum_error}"
        )
    return maximum_error


@torch.no_grad()
def evaluate(
    baseline: CompactDeblurUNet,
    candidate: CompactDeblurUNet,
    loader: DataLoader,
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
    candidate.eval()
    records: list[dict[str, float | str]] = []
    for batch in loader:
        blurry = batch["blur"].cuda(non_blocking=True)
        sharp = batch["sharp"].cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            baseline_output = baseline(blurry)
            candidate_output = candidate(blurry)
        name = str(batch["name"][0])
        scene = name.split("/", maxsplit=1)[0]
        input_psnr = _psnr_value(blurry, sharp)
        baseline_psnr = _psnr_value(baseline_output, sharp)
        candidate_psnr = _psnr_value(candidate_output, sharp)
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
            "candidate_psnr_db": candidate_psnr,
            "candidate_minus_baseline_db": candidate_psnr - baseline_psnr,
        }
        if detailed:
            baseline_ssim = _ssim_value(baseline_output[0], sharp[0])
            candidate_ssim = _ssim_value(candidate_output[0], sharp[0])
            record.update(
                {
                    "baseline_ssim": baseline_ssim,
                    "candidate_ssim": candidate_ssim,
                    "candidate_minus_baseline_ssim": candidate_ssim - baseline_ssim,
                }
            )
        records.append(record)

    numeric_fields = [
        "input_psnr_db",
        "baseline_psnr_db",
        "candidate_psnr_db",
        "candidate_minus_baseline_db",
    ]
    if detailed:
        numeric_fields.extend(
            [
                "baseline_ssim",
                "candidate_ssim",
                "candidate_minus_baseline_ssim",
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
        raise RuntimeError("CUDA is required; P12 intentionally has no CPU fallback")
    if min(args.steps, args.batch_size, args.train_patch_size, args.val_interval) <= 0:
        raise ValueError("steps, batch-size, train-patch-size, and val-interval must be positive")
    if args.train_pairs < 0 or args.val_pairs < 0 or args.num_workers < 0:
        raise ValueError("train-pairs, val-pairs, and num-workers must be nonnegative")
    if args.learning_rate <= 0 or min(args.frequency_weight, args.gate_supervision_weight) < 0:
        raise ValueError("learning-rate must be positive and loss weights nonnegative")
    validate_gate_supervision(args.variant, args.gate_supervision_weight)
    if args.teacher_mode == "shuffled":
        if args.gate_supervision_weight <= 0 or args.batch_size < 2:
            raise ValueError("shuffled teacher requires positive supervision and batch size >= 2")
        args.teacher_seed = args.seed + 19000 if args.teacher_seed is None else args.teacher_seed
    elif args.teacher_seed is not None:
        raise ValueError("teacher-seed is only used in shuffled mode")

    set_determinism(args.seed)
    device = torch.device("cuda:0")
    baseline_path = args.baseline_checkpoint.expanduser().resolve()
    calibration_path = args.calibration_summary.expanduser().resolve()
    if not baseline_path.is_file():
        raise FileNotFoundError(f"baseline checkpoint not found: {baseline_path}")
    if not calibration_path.is_file():
        raise FileNotFoundError(f"calibration summary not found: {calibration_path}")

    baseline_checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=False)
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
        raise ValueError("calibration must explicitly state official_test_accessed=false")
    heavy_max_psnr_db = float(calibration["heavy_medium_threshold_input_psnr_db"])
    light_min_psnr_db = float(calibration["medium_light_threshold_input_psnr_db"])

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
    overlap = sorted(set(train_set.recording_groups) & set(val_set.recording_groups))
    if overlap:
        raise RuntimeError(f"training/frozen-development leakage detected: {overlap}")

    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_sampler = None
    training_group_counts = dict(
        sorted(
            Counter(recording_group(pair.scene) for pair in train_set.pairs).items()
        )
    )
    if args.sampling == "group_balanced":
        train_sampler, training_group_counts = _make_group_balanced_sampler(
            train_set,
            seed=args.seed + 3,
        )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
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

    scales = tuple(args.scales)
    candidate = _make_candidate(
        baseline_checkpoint["model"],
        width,
        args.variant,
        scales,
        args.top_k,
        args.statistics_window,
    ).to(device)
    for parameter in candidate.parameters():
        parameter.requires_grad_(True)
    exact_initialization_max_abs_error = _audit_exact_initialization(
        baseline,
        candidate,
        val_loader,
        device,
    )

    optimizer = torch.optim.AdamW(candidate.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda")
    teacher_generator = (
        torch.Generator().manual_seed(args.teacher_seed)
        if args.teacher_mode == "shuffled" else None
    )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    history: list[dict[str, float | int]] = []
    teacher_history: list[dict] = []
    best_psnr = float("-inf")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    train_iterator = iter(train_loader)
    progress = tqdm(range(1, args.steps + 1), desc=f"P12 matched {args.variant}")
    for step in progress:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        blurry = batch["blur"].to(device, non_blocking=True)
        sharp = batch["sharp"].to(device, non_blocking=True)
        candidate.train()
        optimizer.zero_grad(set_to_none=True)
        teacher_diagnostics = {} if args.teacher_mode == "shuffled" else None
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, restoration_objective, gate_supervision_loss = compute_training_objective(
                candidate, blurry, sharp, heavy_max_psnr_db, light_min_psnr_db,
                args.frequency_weight, args.gate_supervision_weight,
                teacher_mode=args.teacher_mode, teacher_generator=teacher_generator,
                diagnostics=teacher_diagnostics,
            )
        if teacher_diagnostics is not None:
            teacher_history.append({"step": step, **teacher_diagnostics})
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_value = float(loss.detach())
        if step == 1 or step % 20 == 0:
            progress.set_postfix(loss=f"{loss_value:.4f}")
        if step == 1 or step % args.val_interval == 0 or step == args.steps:
            validation, _, groups, _ = evaluate(
                baseline,
                candidate,
                val_loader,
                heavy_max_psnr_db,
                light_min_psnr_db,
            )
            record = {
                "step": step,
                "train_loss": loss_value,
                "train_restoration_loss": float(restoration_objective.detach()),
                "train_gate_supervision_loss": float(gate_supervision_loss.detach()),
                "baseline_psnr_db": float(validation["baseline_psnr_db"]),
                "candidate_psnr_db": float(validation["candidate_psnr_db"]),
                "candidate_minus_baseline_db": float(
                    validation["candidate_minus_baseline_db"]
                ),
                "minimum_group_candidate_minus_baseline_db": min(
                    float(group["candidate_minus_baseline_db"]) for group in groups
                ),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            print(
                f"\nstep={step:05d} B0={validation['baseline_psnr_db']:.4f} dB "
                f"candidate={validation['candidate_psnr_db']:.4f} dB "
                f"delta={validation['candidate_minus_baseline_db']:+.4f} dB"
            )
            if float(validation["candidate_psnr_db"]) > best_psnr:
                best_psnr = float(validation["candidate_psnr_db"])
                torch.save(
                    {
                        "step": step,
                        "model": candidate.state_dict(),
                        "args": vars(args),
                        "baseline_checkpoint": str(baseline_path),
                    },
                    output / "best.pt",
                )

    _write_csv(output / "metrics.csv", history)
    if teacher_history:
        _write_csv(output / "teacher_diagnostics.csv", teacher_history)
    torch.save(
        {
            "step": args.steps,
            "model": candidate.state_dict(),
            "args": vars(args),
            "baseline_checkpoint": str(baseline_path),
        },
        output / "last.pt",
    )
    best_checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    candidate.load_state_dict(best_checkpoint["model"], strict=True)
    final, per_image, per_group, per_strength = evaluate(
        baseline,
        candidate,
        val_loader,
        heavy_max_psnr_db,
        light_min_psnr_db,
        detailed=True,
    )
    _write_csv(output / "per_image_metrics.csv", per_image)
    _write_csv(output / "per_group_metrics.csv", per_group)
    _write_csv(output / "per_strength_metrics.csv", per_strength)

    # B0 is frozen for evaluation, so ``count_parameters`` would report zero.
    # Capacity comparison must count all stored model parameters, independent
    # of their current ``requires_grad`` state.
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    minimum_group_delta = min(
        float(group["candidate_minus_baseline_db"]) for group in per_group
    )
    absolute_psnr_gate = float(final["candidate_minus_baseline_db"]) >= 0.02
    ssim_gate = float(final["candidate_minus_baseline_ssim"]) >= 0.0
    group_gate = minimum_group_delta >= -0.01
    report = {
        "experiment": output.name,
        "stage": "P12 matched nonlocal screening",
        "variant": args.variant,
        "seed": args.seed,
        "steps": args.steps,
        "best_checkpoint_step": int(best_checkpoint["step"]),
        "baseline_checkpoint": str(baseline_path),
        "baseline_checkpoint_step": int(baseline_checkpoint["step"]),
        "starts_as_exact_baseline": True,
        "exact_initialization_max_abs_error": exact_initialization_max_abs_error,
        "data_root": str(args.data_root.expanduser().resolve()),
        "train_pairs": len(train_set),
        "val_pairs": len(val_set),
        "sampling_strategy": args.sampling,
        "training_group_counts_before_sampling": training_group_counts,
        "target_recording_group_probability": (
            1.0 / len(training_group_counts)
            if args.sampling == "group_balanced"
            else None
        ),
        "train_recording_groups": list(train_set.recording_groups),
        "validation_recording_groups": list(val_set.recording_groups),
        "recording_group_overlap": overlap,
        "train_patch_size": args.train_patch_size,
        "validation_patch_size": int(baseline_args["val_patch_size"]),
        "scales": list(scales) if args.variant != "local" else None,
        "candidate_count": 8 * len(scales) if args.variant != "local" else None,
        "top_k": args.top_k if args.variant != "local" and args.variant != "fixed" else None,
        "statistics_window": (
            args.statistics_window
            if args.variant
            in {
                "scale_adaptive",
                "gate_adaptive",
                "protected_gate_adaptive",
                "adaptive",
            }
            else None
        ),
        "gate_supervision_weight": args.gate_supervision_weight,
        "teacher_mode": args.teacher_mode,
        "teacher_seed": args.teacher_seed,
        "teacher_assignment_diagnostics": ({
            "steps_recorded": len(teacher_history),
            "mean_target_changed_fraction": statistics.fmean(
                row["target_changed_fraction"] for row in teacher_history),
            "mean_target_absolute_change": statistics.fmean(
                row["target_mean_absolute_change"] for row in teacher_history),
            "mean_within_batch_teacher_std": statistics.fmean(
                row["teacher_std"] for row in teacher_history),
            "mean_teacher_endpoint_fraction": statistics.fmean(
                row["teacher_endpoint_fraction"] for row in teacher_history),
        } if teacher_history else None),
        "outer_gate_enabled": args.variant == "protected_gate_adaptive",
        "gate_supervision_target": (
            "nonlocal_gate_spatial_mean" if args.gate_supervision_weight > 0 else None
        ),
        "gate_supervision_uses_sharp_only_during_training": (
            args.gate_supervision_weight > 0
        ),
        "inference_requires_sharp": False,
        "sparsity_means_runtime_reduction": False,
        "baseline_parameters": baseline_parameters,
        "candidate_parameters": candidate_parameters,
        "parameter_overhead_percent": 100.0 * (
            candidate_parameters / baseline_parameters - 1.0
        ),
        **final,
        "per_group": per_group,
        "per_strength": per_strength,
        "minimum_group_candidate_minus_baseline_db": minimum_group_delta,
        "psnr_threshold_db": 0.02,
        "minimum_group_threshold_db": -0.01,
        "absolute_psnr_gate_passed": absolute_psnr_gate,
        "ssim_gate_passed": ssim_gate,
        "group_gate_passed": group_gate,
        "absolute_gate_passed": absolute_psnr_gate and ssim_gate and group_gate,
        "cross_variant_gate_pending": True,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "wall_seconds": time.perf_counter() - started,
        "official_test_accessed": False,
        "paper_result": False,
        "note": (
            "Train-internal frozen-development screening. The official GoPro test split "
            "was not loaded. B3 must also beat B1 and B2 before promotion."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["absolute_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
