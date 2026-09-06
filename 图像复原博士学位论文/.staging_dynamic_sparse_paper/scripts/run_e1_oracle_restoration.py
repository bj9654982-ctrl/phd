"""E1 oracle gate: test the trajectory operator with exact synthetic blur geometry."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_e0_geometry import fixed_evaluation_batch  # noqa: E402
from src.data import make_motion_batch  # noqa: E402
from src.losses import restoration_loss  # noqa: E402
from src.models import CompactDeblurUNet, TrajectoryNonlocalBlock  # noqa: E402
from src.models.compact_unet import count_parameters  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local B0 with oracle-geometry MT-NLR on synthetic blur.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--frequency-weight", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--trajectory-samples", type=int, default=5)
    parser.add_argument("--max-feature-length", type=float, default=7.0)
    parser.add_argument("--fusion", choices=("aggregate", "detail"), default="detail")
    parser.add_argument("--geometry-mode", choices=("oracle", "predicted"), default="oracle")
    parser.add_argument("--geometry-checkpoint", type=Path)
    parser.add_argument("--statistics-window", type=int, default=51)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "E1_oracle_restoration",
    )
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def oracle_geometry(direction: torch.Tensor, input_length: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "direction": direction,
        "length": 1.0 + (input_length - 1.0) / 4.0,
        "confidence": torch.ones_like(input_length),
    }


@torch.no_grad()
def conditioning_geometry(
    mode: str,
    geometry_probe: CompactDeblurUNet | None,
    blurry: torch.Tensor,
    direction: torch.Tensor,
    input_length: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if mode == "oracle":
        return oracle_geometry(direction, input_length)
    if geometry_probe is None:
        raise RuntimeError("predicted geometry mode requires a frozen geometry probe")
    prediction = geometry_probe.predict_geometry(blurry)
    return {
        "direction": prediction["direction"].detach(),
        "length": prediction["length"].detach(),
        "confidence": prediction["confidence"].detach(),
    }


@torch.no_grad()
def batch_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)
    mse = (prediction - target).square().flatten(1).mean(dim=1).clamp_min(1e-12)
    return float((-10.0 * mse.log10()).mean())


@torch.no_grad()
def evaluate(
    local: CompactDeblurUNet,
    oracle: CompactDeblurUNet,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    geometry_mode: str,
    geometry_probe: CompactDeblurUNet | None,
) -> tuple[float, float, float]:
    blurry, sharp, direction, input_length = batch
    local.eval()
    oracle.eval()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        local_output = local(blurry)
        geometry = conditioning_geometry(geometry_mode, geometry_probe, blurry, direction, input_length)
        oracle_output = oracle(blurry, geometry_override=geometry)
    return batch_psnr(blurry, sharp), batch_psnr(local_output, sharp), batch_psnr(oracle_output, sharp)


def train_step(
    model: CompactDeblurUNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    blurry: torch.Tensor,
    sharp: torch.Tensor,
    frequency_weight: float,
    geometry: dict[str, torch.Tensor] | None = None,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = model(blurry, geometry_override=geometry)
        loss = restoration_loss(prediction, sharp, frequency_weight=frequency_weight)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach())


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script has no CPU fallback")
    if args.steps <= 0 or args.eval_interval <= 0:
        raise ValueError("steps and eval-interval must be positive")
    if args.geometry_mode == "predicted" and args.geometry_checkpoint is None:
        raise ValueError("--geometry-checkpoint is required for predicted geometry mode")

    set_determinism(args.seed)
    device = torch.device("cuda:0")
    common = CompactDeblurUNet(width=args.width)
    local = copy.deepcopy(common)
    oracle = copy.deepcopy(common)
    oracle.bottleneck_module = TrajectoryNonlocalBlock(
        channels=args.width * 4,
        samples=args.trajectory_samples,
        max_feature_length=args.max_feature_length,
        fusion_mode=args.fusion,
    )
    local = local.to(device)
    oracle = oracle.to(device)

    geometry_probe = None
    if args.geometry_mode == "predicted":
        geometry_probe = CompactDeblurUNet(width=args.width)
        geometry_probe.bottleneck_module = TrajectoryNonlocalBlock(
            channels=args.width * 4,
            samples=args.trajectory_samples,
            max_feature_length=args.max_feature_length,
            statistics_window=args.statistics_window,
            fusion_mode=args.fusion,
        )
        checkpoint_path = args.geometry_checkpoint.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"geometry checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        geometry_probe.load_state_dict(checkpoint["model"])
        geometry_probe = geometry_probe.to(device).eval()
        for parameter in geometry_probe.parameters():
            parameter.requires_grad_(False)

    local_optimizer = torch.optim.AdamW(local.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    oracle_optimizer = torch.optim.AdamW(oracle.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    local_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(local_optimizer, T_max=args.steps)
    oracle_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(oracle_optimizer, T_max=args.steps)
    local_scaler = torch.amp.GradScaler("cuda")
    oracle_scaler = torch.amp.GradScaler("cuda")

    evaluation_batch = fixed_evaluation_batch(args.size, args.seed + 100_000)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_local = float("-inf")
    best_oracle = float("-inf")
    best_paired_delta = float("-inf")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    initial_input, initial_local, initial_oracle = evaluate(
        local,
        oracle,
        evaluation_batch,
        args.geometry_mode,
        geometry_probe,
    )
    if initial_local != initial_oracle:
        raise RuntimeError("identity initialization failed: local and oracle outputs differ before training")

    progress = tqdm(range(1, args.steps + 1), desc=f"E1 oracle {args.fusion}")
    for step in progress:
        blurry, sharp, direction, input_length = make_motion_batch(
            batch_size=args.batch_size,
            size=args.size,
            seed=args.seed + step,
        )
        local_loss = train_step(
            local,
            local_optimizer,
            local_scaler,
            blurry,
            sharp,
            args.frequency_weight,
        )
        oracle_loss = train_step(
            oracle,
            oracle_optimizer,
            oracle_scaler,
            blurry,
            sharp,
            args.frequency_weight,
            conditioning_geometry(
                args.geometry_mode,
                geometry_probe,
                blurry,
                direction,
                input_length,
            ),
        )
        local_scheduler.step()
        oracle_scheduler.step()

        if step == 1 or step % 20 == 0:
            progress.set_postfix(local=f"{local_loss:.4f}", oracle=f"{oracle_loss:.4f}")
        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            input_value, local_value, oracle_value = evaluate(
                local,
                oracle,
                evaluation_batch,
                args.geometry_mode,
                geometry_probe,
            )
            delta = oracle_value - local_value
            record = {
                "step": step,
                "local_loss": local_loss,
                "oracle_loss": oracle_loss,
                "input_psnr_db": input_value,
                "local_psnr_db": local_value,
                "oracle_psnr_db": oracle_value,
                "oracle_minus_local_db": delta,
                "learning_rate": local_optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            print(
                f"\nstep={step:05d} input={input_value:.3f} dB local={local_value:.3f} dB "
                f"oracle={oracle_value:.3f} dB delta={delta:+.3f} dB"
            )
            if local_value > best_local:
                best_local = local_value
                torch.save({"step": step, "model": local.state_dict(), "args": vars(args)}, output / "best_local.pt")
            if oracle_value > best_oracle:
                best_oracle = oracle_value
                torch.save({"step": step, "model": oracle.state_dict(), "args": vars(args)}, output / "best_oracle.pt")
            best_paired_delta = max(best_paired_delta, delta)

    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    tail = history[-min(3, len(history)) :]
    tail_mean_delta = float(np.mean([float(item["oracle_minus_local_db"]) for item in tail]))
    best_model_delta = best_oracle - best_local
    passed = tail_mean_delta >= 0.1 and best_model_delta >= 0.1
    report = {
        "experiment": f"E1_{args.geometry_mode}_restoration",
        "geometry_mode": args.geometry_mode,
        "fusion": args.fusion,
        "seed": args.seed,
        "steps": args.steps,
        "input_psnr_db": initial_input,
        "initial_local_psnr_db": initial_local,
        "initial_oracle_psnr_db": initial_oracle,
        "local_parameters": count_parameters(local),
        "oracle_parameters": count_parameters(oracle),
        "best_local_psnr_db": best_local,
        "best_oracle_psnr_db": best_oracle,
        "best_model_delta_db": best_model_delta,
        "best_paired_delta_db": best_paired_delta,
        "tail_mean_delta_db": tail_mean_delta,
        "pass_threshold_db": 0.1,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "wall_seconds": time.perf_counter() - started,
        "passed": passed,
        "paper_result": False,
        "note": (
            "Oracle diagnostic only; exact synthetic geometry is unavailable at real inference time."
            if args.geometry_mode == "oracle"
            else "Frozen E0 geometry diagnostic on synthetic blur; this is not a real-data paper result."
        ),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
